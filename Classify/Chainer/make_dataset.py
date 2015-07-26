#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse

from settings import Settings

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--infile', '-i', type=str, default='yelp_all_features.txt')
  parser.add_argument('--outdir', '-o', type=str, default=Settings.DATA_DIR)
  args = parser.parse_args()
  print args

  if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

  data = []
  labels = []

  filename = os.path.join(args.outdir, args.infile)
  RATE = 0.8
  for features in ['text_only', 'topics_only', 'text_topics']:
    print features

    tlist = []
    if features == 'text_only':
      tlist = range(13)
      for i in range(3):
        tlist.append(12)
    elif features == 'topics_only':
      tlist = range(13, 63) # 50 features
      for i in range(14):
        tlist.append(62)
    else:
      tlist = range(63)
      tlist.append(62)

    tlist.append(65)
    usecols = tuple(tlist)

    all_data = np.loadtxt(filename, delimiter=' ', usecols=usecols)
    N, F = all_data.shape
    
    for i in range(N):
      tmp = int(all_data[i,F-1] / 0.2)
      if tmp == 5:
        tmp = 4
      all_data[i, F-1] = tmp

    # Random split into 20 train-test data pairs
    for i in range(Settings.NUM_TEST):
      N_train = int(N*RATE)
      N_test = N - N_train
      p_train = np.random.permutation(N)[:N_train]
      p_test = np.random.permutation(N)[N_train:]

      train_data = all_data[p_train, 0:(F-1)]
      train_label = all_data[p_train, F-1]

      test_data = all_data[p_test, 0:(F-1)]
      test_label = all_data[p_test, F-1]

      psize = int(np.sqrt(F-1))
      assert(F-1 == psize*psize)

      train_data = train_data.reshape((len(p_train), 1, psize, psize)).astype(np.float32)
      train_label = train_label.astype(np.int32)

      test_data = test_data.reshape((len(p_test), 1, psize, psize)).astype(np.float32)
      test_label = test_label.astype(np.int32)

      idx = args.infile.find('_')
      term = args.infile[:idx] + '_' + features + '_' + str(i)

      print term
      print 'train_data', train_data.shape
      print 'train_label', train_label.shape
      
      print 'test_data', test_data.shape
      print 'test_label', test_label.shape

      np.save('%s/train_data_%s' % (args.outdir, term), train_data)
      np.save('%s/train_label_%s' % (args.outdir, term), train_label)

      np.save('%s/test_data_%s' % (args.outdir, term), test_data)
      np.save('%s/test_label_%s' % (args.outdir, term), test_label)

    