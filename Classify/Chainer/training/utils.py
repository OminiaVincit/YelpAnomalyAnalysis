#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from settings import Settings

def load_dataset(term, datadir=Settings.DATA_DIR):
  u'''Load data for training'''
  train_data = np.load('%s/train_data_%s.npy' % (datadir, term))
  train_labels = np.load('%s/train_label_%s.npy' % (datadir, term))
  test_data = np.load('%s/test_data_%s.npy' % (datadir, term))
  test_labels = np.load('%s/test_label_%s.npy' % (datadir, term))

  return train_data, train_labels, test_data, test_labels

def global_contrast_norm(x):
    if not x.dtype == np.float32:
        x = x.astype(np.float32)
    # local contrast normalization
    for ch in range(x.shape[2]):
        im = x[:, :, ch]
        im = (im - np.mean(im)) / \
            (np.std(im) + np.finfo(np.float32).eps)
        x[:, :, ch] = im

    return x