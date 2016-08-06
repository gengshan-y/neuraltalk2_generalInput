####
 # Filename:        prepro.py
 # Date:            Jul 09 2016
 # Last Edited by:  Gengshan Yang
 # Description:     Gathering files(features.h5, captions) indicated by a .json
 #                  Then create a .h5 file and a .json file.
 #                  Input: json file that has the form
 #                  [{file_path: 'path/img.jpg', captions: ['a caption', ...] 
 #                  }, ...]
 ####

"""
This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file, an hdf5 file and a wordCount.txt file
The hdf5 file contains several fields:
/feats is (N,4096) uint8 array of feature data
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'feats_info' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize
from preproLib import *

def assign_splits(featPairs, params):
  num_val = params['num_val']
  num_test = params['num_test']

  for i,featPair in enumerate(featPairs):
      if i < num_val:
        featPair['split'] = 'val'
      elif i < num_val + num_test: 
        featPair['split'] = 'test'
      else: 
        featPair['split'] = 'train'

  print 'assigned %d to val, %d to test.' % (num_val, num_test)

def encode_captions(imgs, params, wtoi):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = params['max_length']
  N = len(imgs)
  M = sum(len(img['final_captions']) for img in imgs) # total number of captions

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  caption_counter = 0
  counter = 1
  for i,img in enumerate(imgs):
    n = len(img['final_captions'])
    assert n > 0, 'error: some image has no captions'

    Li = np.zeros((n, max_length), dtype='uint32')
    for j,s in enumerate(img['final_captions']):
      label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
      caption_counter += 1
      for k,w in enumerate(s):
        if k < max_length:
          Li[j,k] = wtoi[w]

    # note: word indices are 1-indexed, and captions are padded with zeros
    label_arrays.append(Li)
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1
    
    counter += n
  
  L = np.concatenate(label_arrays, axis=0) # put all the labels together
  assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
  assert np.all(label_length > 0), 'error: some caption had no words?'

  print 'encoded captions to array of size ', `L.shape`
  return L, label_start_ix, label_end_ix, label_length

def main(params):
  featPairs = json.load(open(params['input_json'], 'r'))
  seed(123) # make reproducible
  shuffle(featPairs) # shuffle the order

  ''' tokenization and preprocessing '''
  prepro_captions(featPairs)

  ''' create the vocab '''
  vocab = build_vocab(featPairs, params)
  itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

  ''' assign the splits '''
  assign_splits(featPairs, params)
  
  ''' encode captions in large arrays, ready to ship to hdf5 file '''
  L, label_start_ix, label_end_ix, label_length = encode_captions(featPairs, params, wtoi)

  ''' create output h5 file '''
  N = len(featPairs)
  f = h5py.File(params['output_h5'], "w")
  f.create_dataset("labels", dtype='uint32', data=L)
  f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  f.create_dataset("label_length", dtype='uint32', data=label_length)

  ''' load the feature '''
  featSet = f.create_dataset("feats", (N, 4096), dtype='float32')
  for i,featPair in enumerate(featPairs):
    tmpFeatSet = h5py.File(featPair['file_path'],'r')
    featSet[i] = tmpFeatSet['mp']  # change here to be consistent with your h5

    if i % 1000 == 0:
      print 'processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N)
  f.close()
  print 'wrote ', params['output_h5']

  ''' create output json file '''
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['feats_info'] = []
  for i,featPair in enumerate(featPairs):    
    jfeat = {}
    jfeat['split'] = featPair['split']
    jfeat['captions'] = featPair['captions']
    if 'file_path' in featPair: 
      jfeat['file_path'] = featPair['file_path']  # copy path over, might need
    out['feats_info'].append(jfeat)

  json.dump(out, open(params['output_json'], 'w'))
  print 'wrote ', params['output_json']

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
  parser.add_argument('--num_val', required=True, type=int, help='number of images to assign to validation data (for CV etc)')
  parser.add_argument('--output_json', default='data/data.json', help='output json file')
  parser.add_argument('--output_h5', default='data/data.h5', help='output h5 file')
  
  # options
  parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed input parameters:'
  print json.dumps(params, indent = 2)
  main(params)
