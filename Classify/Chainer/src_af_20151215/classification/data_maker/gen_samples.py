#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Make dataset for 2-channel text_topics features
"""
import os, sys
import numpy as np
import argparse
sys.path.append(r'../utils')
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

def split_dataset(site, train_rate, outdir):
  """
  Random split dataset into training/test dataset
  """
  all_data = np.load(os.path.join(outdir, site + '_combined_data.npy'))
  print all_data.shape                        

  # Extract by label
  L = 5 # number of labels

  tmp_map = dict()
  for i in range(L):
    tmp_map[i] = all_data[all_data[:,-1] == i]
    print i, tmp_map[i].shape

  # Peform data spli
  N = 20 # number of splitation
  for j in range(N):
    train_data = []
    test_data = []
    
    for i in range(L):
      fsize = tmp_map[i].shape[0]
      trsize = int(fsize * train_rate)
      perm = np.random.permutation(tmp_map[i].shape[0])
      train_data.append(tmp_map[i][perm[:trsize]] )
      test_data.append(tmp_map[i][perm[trsize:]] )

    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)
    
    print j, train_data.shape, test_data.shape

    # # Divide data for prediction exp
    # np.save('%s/%s_train_data_split_predict_%d' % (outdir, site, j), train_data)
    # np.save('%s/%s_test_data_split_predict_%d'  % (outdir, site, j), test_data)
    
    # Divide data for classification exp
    np.save('%s/%s_train_data_class_predict_%d' % (outdir, site, j), train_data)
    np.save('%s/%s_test_data_class_predict_%d'  % (outdir, site, j), test_data)

    print 'Saved data'

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
      # tlist = range(num_features)
      # tlist.append(num_features + 2)
      # usecols = tuple(tlist)
      usecols = None
      all_data = np.loadtxt(filepath, delimiter=' ', usecols=usecols)
      print site, 'Topics data shape', all_data.shape

      # Read text features
      text_path = os.path.join(args.outdir, site + '_text_features.txt')
      usetextcols = tuple(range(13))
      text_data = np.loadtxt(text_path, delimiter=' ', usecols=usetextcols)
      all_data = np.hstack((text_data, all_data))
      print site, 'Combined text data shape', all_data.shape

      (N, F) = all_data.shape
      labels = np.zeros((N,1), dtype=all_data.dtype)
      for i in range(N):
        tmp = int(all_data[i,F-1] / 0.2)
        if tmp == 5:
          tmp = 4
        labels[i, 0] = tmp

      all_data = np.hstack((all_data, labels))
      print site, 'Combined label, data shape', all_data.shape
      np.save('%s/%s_combined_data' % (args.outdir, site), all_data)
      print 'Saved data'

  #split_dataset('yelp', 0.8, Settings.FEATURES_DIR)
 