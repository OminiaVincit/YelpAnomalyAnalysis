#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import imp
import cPickle as pickle
#from chainer import cuda
import chainer.functions as F
import numpy as np
import argparse

import sys
sys.path.append('../training')
from utils import global_contrast_norm, load_dataset

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

def draw_patchs(x_batch, ans, recogs, msize=8):
  wsize = int(np.sqrt(x_batch.shape[0]))
  for i in range(wsize*wsize):
    Z = x_batch[i].reshape(msize, msize)
    plt.subplot(wsize, wsize, i)
    Z = Z[::-1,:]
    plt.xlim(0, msize-1)
    plt.ylim(0, msize-1)
    plt.pcolor(Z)
    if ans[i] != recogs[i]:
      tc = 'r'
    else:
      tc = 'black'  
    plt.title('ans=%d, recog=%d'%(ans[i], recogs[i]), size=8, color=tc)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')

def validate(test_data, test_labels, N_test, model, args):
  """
  Validate
  """
  sum_accuracy = 0
  sum_loss = 0
  for i in range(0, N_test, args.batchsize):
    high = min(N_test, i+args.batchsize)
    x_batch = test_data[i:high]
    y_batch = test_labels[i:high]

    if args.norm == 1:
      x_batch = np.asarray(map(global_contrast_norm, x_batch))

    if args.gpu >= 0:
      with cupy.cuda.Device(args.gpu):
          x_batch = cupy.array(x_batch.astype(np.float32))
          y_batch = cupy.array(y_batch.astype(np.int32))

    loss, acc, pred = model.forward(x_batch, y_batch, train=False)

    # New code for cupy
    if args.gpu >= 0:
        tmp_loss = float(loss.data.get()) 
        sum_loss += tmp_loss * (high - i)
    else:
        tmp_loss = float(loss.data)
        sum_loss += tmp_loss * (high - i)
    
    if args.gpu >= 0:
        tmp = float(acc.data.get())
        sum_accuracy += tmp * (high - i)
    else:
        tmp = float(acc.data)  
        sum_accuracy += tmp * (high - i)

    msg = 'validate {}%: loss={}, accuracy={}'.format(
      int(high*100 / N_test),
      sum_loss / high, sum_accuracy / high )
    print('%s' % msg)

  return sum_loss, sum_accuracy

def visual_ans(model, test_data, test_labels):
  """
  Visualize answers
  """
  # Visualize input data
  N_test = test_data.shape[0]
  assert(N_test > 0)
  N_display = 100
  plt.figure(figsize=(10, 10))
  perm = np.random.permutation(N_test)[:N_display]
  x_batch = test_data[perm]
  y_batch = test_labels[perm]

  x_batch = np.asarray(map(global_contrast_norm, x_batch))
  pred = model.deploy(x_batch)
  pred = pred.data
  pred = np.argmax(pred, axis=1)
  print 'Accuracy = ', np.sum(pred==y_batch) / float(N_display)
  draw_patchs(x_batch, pred, y_batch)
  plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='../deploy/NetModel.py')
  parser.add_argument('--param', type=str, default='../params/yelp_topics_NetModel_dt_0_epoch_2400.chainermodel')
  parser.add_argument('--norm', type=int, default=1)
  parser.add_argument('--batchsize', type=int, default=128)
  parser.add_argument('--gpu', type=int, default=-1)
  parser.add_argument('--label', type=int, default=0)
  parser.add_argument('--datadir', type=str, default='../../../../Dataset/Features')
  parser.add_argument('--data_index', type=int, default=0)
  parser.add_argument('--site', type=str, default='yelp',
                      choices=['yelp', 'tripadvisor'])
  parser.add_argument('--features', type=str, default='topics',
                      choices=['text_only', 'topics', 'text_topics'])
  args = parser.parse_args()
  model_n = os.path.basename(args.model).split('.')[0]
  module = imp.load_source(model_n, args.model)
  model = pickle.load(open(args.param, 'rb'))
  if args.gpu >= 0:
      model.to_gpu()
  else:
      model.to_cpu()

  term = args.site + '_' + args.features + '_64_features_order_' + str(args.data_index)
  _, _, test_data, test_labels = load_dataset(term, args.datadir)
    # Extract by label
  if args.label >= 0:
    test_data = test_data[test_labels == args.label]
    test_labels = test_labels[test_labels == args.label]
  N_test = test_data.shape[0]
  print 'Number of test sample ', N_test
  sum_loss, sum_accuracy = validate(test_data, test_labels, N_test, model, args)
  print('test mean loss={}, accuracy={}'.format(
         sum_loss / N_test, sum_accuracy / N_test))

  visual_ans(model, test_data, test_labels)




