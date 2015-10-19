#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import imp
import cPickle as pickle
from chainer import cuda
import chainer.functions as F
import numpy as np
import argparse
from utils import global_contrast_norm, load_dataset
from settings import Settings

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
  '''Validate test dataset
  '''
  sum_accuracy = 0
  sum_loss = 0
  sum_correct = 0
  for i in range(0, N_test, args.batchsize):
    x_batch = test_data[i:(i+args.batchsize)]
    y_batch = test_labels[i:(i+args.batchsize)]

    if args.norm == 1:
      x_batch = np.asarray(map(global_contrast_norm, x_batch))

    if args.gpu >= 0:
      x_batch = cuda.to_gpu(x_batch.astype(np.float32))
      y_batch = cuda.to_gpu(y_batch.astype(np.int32))

    loss, acc, pred = model.forward(x_batch, y_batch, train=False)
    pred = cuda.to_cpu(F.softmax(pred).data)
    pred = np.argmax(pred, axis=1)
    label = cuda.to_cpu(y_batch)

    sum_correct += np.sum(np.array(pred==label))
    sum_loss += float(cuda.to_cpu(loss.data)) * args.batchsize
    sum_accuracy += float(cuda.to_cpu(acc.data)) * args.batchsize
    pt = int((i+args.batchsize)*100/N_test)
    print 'Validate(', pt,'%), mean loss= ', sum_loss / (i+args.batchsize),
    print ',mean accuracy= ', sum_accuracy / (i+args.batchsize)

  print('correct num:{}\t# of test images:{}'.format(sum_correct, N_test))
  print('correct rate:{}'.format(float(sum_correct) / float(N_test)))

  return sum_loss, sum_accuracy

def visual_ans(model, test_data, test_labels):
  # Visualize input data
  N_test = test_data.shape[0]
  plt.figure(figsize=(10, 10))
  perm = np.random.permutation(N_test)[:100]
  x_batch = test_data[perm]
  y_batch = test_labels[perm]

  x_batch = np.asarray(map(global_contrast_norm, x_batch))
  _, acc, pred = model.forward(x_batch, y_batch, train=False)
  pred = cuda.to_cpu(F.softmax(pred).data)
  pred = np.argmax(pred, axis=1)
  label = cuda.to_cpu(y_batch)
  print 'Accuracy = ', acc.data
  draw_patchs(x_batch, pred, label)
  plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='models/NetModel.py')
  parser.add_argument('--param', type=str)
  parser.add_argument('--norm', type=int, default=1)
  parser.add_argument('--batchsize', type=int, default=128)
  parser.add_argument('--gpu', type=int, default=-1)
  parser.add_argument('--datadir', type=str, default=Settings.DATA_DIR)
  parser.add_argument('--data_index', type=int, default=0)
  parser.add_argument('--site', type=str, default='yelp',
                      choices=['yelp', 'tripadvisor', 'movies'])
  parser.add_argument('--features', type=str, default='text_topics',
                      choices=['text_only', 'topics_only', 'text_topics'])
  args = parser.parse_args()

  if args.gpu >= 0:
      cuda.init()
  model_n = os.path.basename(args.model).split('.')[0]
  module = imp.load_source(model_n, args.model)
  model = pickle.load(open(args.param, 'rb'))
  if args.gpu >= 0:
      model.to_gpu()
  else:
      model.to_cpu()

  term = args.site + '_' + args.features + '_' + str(args.data_index)
  _, _, test_data, test_labels = load_dataset(term, args.datadir)

  # N_test = test_data.shape[0]
  # sum_loss, sum_accuracy = validate(test_data, test_labels, N_test, model, args)
  # print('test mean loss={}, accuracy={}'.format(
  #       sum_loss / N_test, sum_accuracy / N_test))

  visual_ans(model, test_data, test_labels)




