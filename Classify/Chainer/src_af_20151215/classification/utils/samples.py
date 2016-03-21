#!env python
# -*- coding: utf-8 -*-
"""
Load samples
"""

import numpy as np
import glob
import os, sys, time
import pickle

class SampleCreator(object):
    def __init__(self):
        pass

class RVTopicsSampleCreator(SampleCreator):
    def __init__(self):
        pass
    
    def load_samples(self, data_dir, site, index, train_down_rate, test_down_rate):
      """
      Load samples from dataset
      """
      train_filename = site + '_train_data_split_class_' + str(index) + '.npy'
      train_data = np.load(os.path.join(data_dir, train_filename))
      num_samples = min(train_data.shape[0] * train_down_rate, train_data.shape[0])
      train_label = np.array(train_data[:num_samples,-1], dtype=np.int32)
      train_data = np.array(train_data[:num_samples, 13:77], dtype=np.float32).reshape(num_samples, 1, 8, 8)

      test_filename = site + '_test_data_split_class_' + str(index) + '.npy'
      test_data = np.load(os.path.join(data_dir, test_filename))
      num_samples = min(test_data.shape[0] * test_down_rate, test_data.shape[0])
      test_label = np.array(test_data[:num_samples,-1], dtype=np.int32)
      test_data = np.array(test_data[:num_samples, 13:77], dtype=np.float32).reshape(num_samples, 1, 8, 8)
      return train_data, train_label, test_data, test_label

class RVForCheckSampleCreator(SampleCreator):
    def __init__(self):
        pass
    
    def load_samples(self, data_dir, site, index, train_down_rate, test_down_rate):
      """
      Load samples from dataset
      """
      train_dat_filename = 'train_data_' + site + '_topics_64_features_order_' + str(index) + '.npy'
      train_lb_filename = 'train_label_' + site + '_topics_64_features_order_' + str(index) + '.npy'

      train_data = np.load(os.path.join(data_dir, train_dat_filename))
      train_label = np.load(os.path.join(data_dir, train_lb_filename))

      test_dat_filename = 'test_data_' + site + '_topics_64_features_order_' + str(index) + '.npy'
      test_lb_filename  = 'test_label_' + site + '_topics_64_features_order_' + str(index) + '.npy'

      test_data = np.load(os.path.join(data_dir, test_dat_filename))
      test_label = np.load(os.path.join(data_dir, test_lb_filename))

      return train_data, train_label, test_data, test_label

class RVSampleLoader(SampleCreator):
  def __init__(self):
    pass

  def load_samples(self, data_dir, site, ftype, index):
    """
    Load samples from dataset
    """
    # Get partition of exp
    exp_file = '%s_partition.pickle' % site
    with open(os.path.join(data_dir, exp_file), 'rb') as handle:
      part = pickle.load(handle)

    # Load data file
    data_file = '%s_%s_features.npy' % (site, ftype)
    data = np.load(os.path.join(data_dir, data_file))
    train_index = part[index]['train']
    test_index = part[index]['test']

    x_train = data[train_index, 0:(-3)]
    x_test = data[test_index, 0:(-3)]

    y_train = data[train_index, -1] / 0.2
    y_train = y_train.astype(np.int32)
    y_train[y_train == 5] = 4

    y_test  = data[test_index, -1] / 0.2
    y_test = y_test.astype(np.int32)
    y_test[y_test == 5] = 4

    # if ftype == 'TOPICS_64':
    #   x_train = x_train.reshape(len(train_index), 1, 8, 8)
    #   x_test = x_test.reshape(len(test_index), 1, 8, 8)
    # if ftype == 'tfidf':
    #   x_train = x_train.reshape(len(train_index), 1, 32, 32)
    #   x_test = x_test.reshape(len(test_index), 1, 32, 32)
    
    if ftype == 'TOPICS_MATRIX_64':
      num_train = len(train_index)
      num_test = len(test_index)
      x_train = x_train[:num_train, :3*64*64].reshape(num_train, 3, 64, 64)
      x_test = x_test[:num_test, :3*64*64].reshape(num_test, 3, 64, 64)
      y_train = y_train[:num_train]
      y_test = y_test[:num_test]
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    creator = RVTopicsSampleCreator()
    creator.load_samples(Settings.FEATURES_DIR, 'yelp', 0)