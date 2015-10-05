#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse

from settings import Settings

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

  for features in ['text_topics']:
    for site in ['yelp', 'tripadvisor']:
      for num_features in [64, 100, 144, 196, 256]:
        filename = site + '_topics_' + str(num_features) + '_features.txt'
        filepath = os.path.join(args.outdir, filename)
        tlist = range(num_features)
        tlist.append(num_features + 2)
        usecols = tuple(tlist)

        all_data = np.loadtxt(filepath, delimiter=' ', usecols=usecols)
        print all_data.shape
        if features == 'text_topics':
          text_path = os.path.join(args.outdir, site + '_text_features.txt')
          usetextcols = tuple(range(13))
          text_data = np.loadtxt(text_path, delimiter=' ', usecols=usetextcols)
          print text_data.shape
          all_data = np.hstack((text_data, all_data))

        N, F = all_data.shape
        print N, F
        psize = int(np.sqrt(F-1))
        if (F-1 != psize*psize):
          apd = (psize+1)*(psize+1) - (F-1)
          first = all_data[:, 0:(F-1)]
          second = all_data[:, (F-1-apd):(F-1)]
          end = all_data[:, F-1]
          end = end.reshape((end.shape[0], 1))
          all_data = np.hstack((first, second, end))

        N, F = all_data.shape
        print N, F
        psize = int(np.sqrt(F-1))
        assert(F-1 == psize * psize)

        N_train = int(N * args.train_rate)
        N_test = int(N * args.test_rate)

        assert(N_train + N_test <= N)

        for i in range(N):
          tmp = int(all_data[i,F-1] / 0.2)
          if tmp == 5:
            tmp = 4
          all_data[i, F-1] = tmp

        # Random split into 20 train-test data pairs
        for i in range(Settings.NUM_TEST):
          p_train = np.random.permutation(N)[:N_train]
          p_test = np.random.permutation(N)[N_train:(N_train + N_test)]

          train_data = all_data[p_train, 0:(F-1)]
          train_label = all_data[p_train, F-1]

          test_data = all_data[p_test, 0:(F-1)]
          test_label = all_data[p_test, F-1]

          train_data = train_data.reshape((len(p_train), 1, psize, psize)).astype(np.float32)
          train_label = train_label.astype(np.int32)

          test_data = test_data.reshape((len(p_test), 1, psize, psize)).astype(np.float32)
          test_label = test_label.astype(np.int32)

          idx = filename.find('.txt')
          term = filename[:idx] + '_order_' + str(i)
          term = term.replace('_topics_', '_' + features + '_')

          print term
          print 'train_data', train_data.shape
          print 'train_label', train_label.shape
          
          print 'test_data', test_data.shape
          print 'test_label', test_label.shape

          np.save('%s/train_data_%s' % (args.outdir, term), train_data)
          np.save('%s/train_label_%s' % (args.outdir, term), train_label)

          np.save('%s/test_data_%s' % (args.outdir, term), test_data)
          np.save('%s/test_label_%s' % (args.outdir, term), test_label)

    