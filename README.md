# neuraltalk2_generalInput
This is a simplified version of [Neuraltalk2](https://github.com/karpathy/neuraltalk2). 
It takes general-features/sentence pairs as input, and uses them to train a language model.

## Data Preparation
0. `$ mkdir output` to make a dictionary for word counting file.
1. Provide path-to-feature/caption pair list in the following json format:  
[{file_path: 'path/feature1.h5', captions: ['a caption', ...]}, ...]

2. Then call prepro.py to gather training data in an .h5 file and a .json file.  
**Usage**: python prepro.py --input_json path-to-list --num_val max-num-splited-to-val  
**e.g.**   python prepro.py --input_json data/test_input.json --num_val 2 

#### output json format
{  
'ix_to_word'[vocab-index mapping form {ix:'word'}, where ix is 1-indexed]:  
'feats_info'[a list of pairs stores the split info and feature path]:  
}

#### output h5 format
/feats is (N,4096) float32 array of feature data, N is the number of images  
/labels is (M,max_length) uint32 array of encoded labels, zero padded, M is the number of captions

## Train the network
Then run the training code:  
**Usage:** th train.lua -input_h5 training-h5 -input_json training-json - checkpoint_path output-checkpoint-path -id task-id  
**e.g.**   th train.lua -input_h5 data/data.h5 -input_json data/data.json -checkpoint_history_path output/ -id 1

## Play around with the code
#### prepro.py
  1. It preprocesses the corpus, and assigns the train-val split.
  
  2. It also generates a wordCount.txt file in output/ folder that counts the vacab frequency.

  3. Change line 190-194 to modify feature-dimension and feature-name(as stored in hdf5 file)


#### misc/DataLoader.lua
  1. It fetches batches of data before forward pass.

  2. Uncomment line 142-149 to check the correctess of input sentences.

#### train.lua
  1. It  manages the whole process of training and validation.

  2. In line 176-181, we allocated memory for feature vectors, which avoids memory
     reallocation after fetching batches of features.

  3. We defined a build_fc() function in "misc/net_utils.lua", and called it in line 130 of train,
  to build up the connection layer between features LSTM part, instead of a whole cnn model.
  
  4. We modified the code to sample sentences during training and valication phase in line 221-225
  and line 302-312.

