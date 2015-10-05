#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Make dataset for 2-channel text_topics features
"""
import os
import numpy as np
import argparse

from settings import Settings

def enlarge(nparr):
  W = int( np.sqrt(nparr.shape[0]) )
  assert( nparr.shape[0] == W*W )
  nparr = nparr.reshape((W, W))
  rsarr = np.zeros((2*W, 2*W), dtype=nparr.dtype)
  for i in range(W):
    for j in range(W):
      rsarr[2*i, 2*j] = nparr[i, j]
      rsarr[2*i+1, 2*j+1] = nparr[i, j]
  rsarr = rsarr.reshape((4*W*W, ))
  return rsarr

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--outdir', '-o', type=str, default=Settings.FEATURES_DIR)
  parser.add_argument('--train_rate', type=float, default=0.8)
  parser.add_argument('--test_rate', type=float, default=0.2)
  args = parser.parse_args()
  print args

  if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

  data = []
  labels = []

  for site in ['yelp', 'tripadvisor']:
    for num_features in [64]:
      filename = site + '_topics_' + str(num_features) + '_features.txt'
      filepath = os.path.join(args.outdir, filename)
      tlist = range(num_features)
      tlist.append(num_features + 2)
      usecols = tuple(tlist)

      all_data = np.loadtxt(filepath, delimiter=' ', usecols=usecols)
      print 'Topics data shape', all_data.shape

      # Read text features
      text_path = os.path.join(args.outdir, site + '_text_features.txt')
      usetextcols = tuple(range(13))
      text_data = np.loadtxt(text_path, delimiter=' ', usecols=usetextcols)
      text_data = np.hstack(( text_data[:, 0:3], text_data ))
      en_text_data = np.zeros((text_data.shape[0], 4*text_data.shape[1]), dtype=np.float32)

      for i in range(en_text_data.shape[0]):
        en_text_data[i] = enlarge(text_data[i])
      print 'Text data shape', en_text_data.shape

      all_data = np.hstack((en_text_data, all_data))

      N, F = all_data.shape
      print N, F
      psize = int( np.sqrt((F-1) / 2) )
      assert(F-1 == 2 * psize * psize)

      N_train = int(N * args.train_rate)
      N_test = int(N * args.test_rate)

      assert(N_train + N_test <= N)

      for i in range(N):
        tmp = int(all_data[i,F-1] / 0.2)
        if tmp == 5:
          tmp = 4
        all_data[i, F-1] = tmp

      # Random split into 10 train-test data pairs
      for i in range(10):
        p_train = np.random.permutation(N)[:N_train]
        p_test = np.random.permutation(N)[N_train:(N_train + N_test)]

        train_data = all_data[p_train, 0:(F-1)]
        train_label = all_data[p_train, F-1]

        test_data = all_data[p_test, 0:(F-1)]
        test_label = all_data[p_test, F-1]

        train_data = train_data.reshape((len(p_train), 2, psize, psize)).astype(np.float32)
        train_label = train_label.astype(np.int32)

        test_data = test_data.reshape((len(p_test), 2, psize, psize)).astype(np.float32)
        test_label = test_label.astype(np.int32)

        term = site + '_text_topics_2_channel_' + str(num_features)
        term += '_features_order_' + str(i)

        print term
        print 'train_data', train_data.shape
        print 'train_label', train_label.shape
        
        print 'test_data', test_data.shape
        print 'test_label', test_label.shape

        np.save('%s/train_data_%s' % (args.outdir, term), train_data)
        np.save('%s/train_label_%s' % (args.outdir, term), train_label)

        np.save('%s/test_data_%s' % (args.outdir, term), test_data)
        np.save('%s/test_label_%s' % (args.outdir, term), test_label)

    