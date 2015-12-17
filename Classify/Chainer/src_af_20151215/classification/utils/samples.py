#!env python
# -*- coding: utf-8 -*-
"""
Load samples
"""

import numpy as np
import glob
import os, sys, time

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

if __name__ == '__main__':
    creator = RVTopicsSampleCreator()
    creator.load_samples(Settings.FEATURES_DIR, 'yelp', 0)