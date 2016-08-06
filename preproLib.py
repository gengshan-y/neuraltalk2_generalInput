import string

''' Preprocess all the captions (punctuation, lowercase, special symbols) '''
def prepro_captions(featPairs):
  print 'example processed tokens:'
  for i,featPair in enumerate(featPairs):
    featPair['processed_tokens'] = []
    for j,s in enumerate(featPair['captions']):
      txt = str(s.encode("utf-8")).lower().translate(None, string.punctuation).strip().split()
      featPair['processed_tokens'].append(txt)
      if i < 10 and j == 0: print txt

''' Build vocabulary and generate some statistics '''
def build_vocab(featPairs, params):
  count_thr = params['word_count_threshold']

  # count up the number of words
  counts = {}
  for featPair in featPairs:
    for txt in featPair['processed_tokens']:
      for w in txt:
        counts[w] = counts.get(w, 0) + 1
  cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
  print 'top words and their counts:'
  print '\n'.join(map(str,cw[:20]))

  # write an extra file of wordCount 
  wordCount = '\n'.join(map(str,cw));
  with open('wordCount.txt', 'w') as tmpFile:
    tmpFile.write(wordCount)

  # print some stats
  total_words = sum(counts.itervalues())
  print 'total words:', total_words
  bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
  vocab = [w for w,n in counts.iteritems() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
  print 'number of words in vocab would be %d' % (len(vocab), )
  print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

  # lets look at the distribution of lengths as well
  sent_lengths = {}
  for featPair in featPairs:
    for txt in featPair['processed_tokens']:
      nw = len(txt)
      sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print 'max length sentence in raw data: ', max_len
  sum_len = sum(sent_lengths.values())
  print sum_len, ' sentences in total'
  print total_words / sum_len, ' words per sentece in average' 
  print 'sentence length distribution (count, number of words):'
  for i in xrange(max_len+1):
    print '%2d| %10d|  %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len)

  # lets now produce the final annotations
  if bad_count > 0:
    # additional special UNK token we will use below to map infrequent words to
    vocab.append('UNK')
    print 'special UNK token inserted'

  for featPair in featPairs:
    featPair['final_captions'] = []
    for txt in featPair['processed_tokens']:
      caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
      featPair['final_captions'].append(caption)
  return vocab
