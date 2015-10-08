#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import logging
import time
import os
import imp
import shutil
import numpy as np
from chainer import optimizers
from utils import load_dataset, global_contrast_norm
from settings import Settings
from transform import Transform
import pickle
from draw_loss import draw_loss_curve
#from progressbar import ProgressBar
#from multiprocessing import Process, Queue

def create_result_dir(args):
  u'''Create log file'''
  result_dir =  os.path.join(args.result_dir, os.path.basename(args.model).split('.')[0])
  result_dir += '_' + args.site + '_' + args.features + '_' + str(args.num_features)
  term = args.opt
  if args.opt == 'Adam':
    term += '_alpha_' + str(args.alpha)
  elif args.opt == 'AdaDelta':
    term += '_rho_' + str(args.rho) + '_eps_' + str(args.eps)
  else:
    term += '_lr_' + str(args.lr)

  result_dir += '_' + term + '_data_' + str(args.data_index)
  result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
  result_dir += str(time.time()).replace('.', '')
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  log_fn = '%s/log_%s.txt' % (result_dir, str(args.epoch_offset))
  logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=log_fn, level=logging.DEBUG)
  logging.info(args)

  return log_fn, result_dir


def get_model_optimizer(result_dir, args):
  model_fn = os.path.basename(args.model)
  model_name = model_fn.split('.')[0]
  module = imp.load_source(model_fn.split('.')[0], args.model)
  Net = getattr(module, model_name)

  dst = '%s/%s' % (result_dir, model_fn)
  if not os.path.exists(dst):
    shutil.copy(args.model, dst)

  dst = '%s/%s' % (result_dir, os.path.basename(__file__))
  if not os.path.exists(dst):
    shutil.copy(__file__, dst)

  # prepare model
  model = Net()
  if args.restart_from is not None:
    model = pickle.load(open(args.restart_from, 'rb'))

  # prepare optimizer
  if args.opt == 'MomentumSGD':
    optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
  elif args.opt == 'Adam':
    optimizer = optimizers.Adam(alpha=args.alpha)
  elif args.opt == 'AdaGrad':
    optimizer = optimizers.AdaGrad(lr=args.lr)
  elif args.opt == 'AdaDelta':
    optimizer = optimizers.AdaDelta(rho=args.rho, eps=args.eps)
  elif args.opt == 'RMSprop':
    optimizer = optimizer.RMSprop(lr=args.lr)
  else:
    raise Exception('No optimizer is selected')
  optimizer.setup(model.collect_parameters())

  return model, optimizer

def augmentation(x_batch_queue, aug_x_queue, trans):
  while True:
    x_batch = x_batch_queue.get()
    if x_batch is None:
      break

    aug_x = []
    for x in x_batch:
      aug = trans.transform(x.transpose((1, 2, 0))).transpose((2, 0, 1))
      aug_x.append(aug)
    aug_x_queue.put(np.asarray(aug_x))

def train(train_data, train_labels, N, model, optimizer, args, trans):
  u'''For training'''
  # training
  # pbar = ProgressBar(N)
  perm = np.random.permutation(N)
  sum_accuracy = 0
  sum_loss = 0

  for i in range(0, N, args.batchsize):
    high = min(N, i+args.batchsize)
    x_batch = train_data[perm[i:high]]
    y_batch = train_labels[perm[i:high]]
    
    if args.norm == 1:
      x_batch = np.asarray(map(global_contrast_norm, x_batch))

    optimizer.zero_grads()
    loss, acc = model.forward(x_batch, y_batch, train=True)
    sum_loss += float(loss.data) * (high - i)
    sum_accuracy += float(acc.data) * (high - i)
    
    msg = 'train {}% : loss={}, accuracy={}'.format(
      int(high*100 / N),
      sum_loss / high, sum_accuracy / high )
    print('%s' % msg)

    # Check inf and nan loss when backward
    inf = float("inf")
    if ( loss.data != loss.data or loss.data == inf ):
      continue
    loss.backward()
    if args.opt in ['AdaGrad', 'MomentumSGD']:
      optimizer.weight_decay(decay=args.weight_decay)
    optimizer.update()

    #pbar.update(i + args.batchsize if (i + args.batchsize) < N else N)

  return sum_loss, sum_accuracy

def validate(test_data, test_labels, N_test, model, args):
  u'''Validate'''
  #pbar = ProgressBar(N_test)
  sum_accuracy = 0
  sum_loss = 0
  for i in range(0, N_test, args.batchsize):
    high = min(N_test, i+args.batchsize)
    x_batch = test_data[i:high]
    y_batch = test_labels[i:high]

    if args.norm == 1:
      x_batch = np.asarray(map(global_contrast_norm, x_batch))

    loss, acc, pred = model.forward(x_batch, y_batch, train=False)
    sum_loss += float(loss.data) * args.batchsize
    sum_accuracy += float(acc.data) * args.batchsize

    msg = 'validate {}%: loss={}, accuracy={}'.format(
      int(high*100 / N_test),
      sum_loss / high, sum_accuracy / high )
    print('%s' % msg)

    #pbar.update(i + args.batchsize 
    #           if (i + args.batchsize) < N_test else N_test)

  return sum_loss, sum_accuracy

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str,
                      default='models/NetModelCh.py')
  parser.add_argument('--gpu', type=int, default=-1)
  parser.add_argument('--epoch', type=int, default=100)
  parser.add_argument('--batchsize', type=int, default=128)
  parser.add_argument('--prefix', type=str,
                      default='NetModelCh')
  parser.add_argument('--snapshot', type=int, default=100)
  parser.add_argument('--restart_from', type=str)
  parser.add_argument('--epoch_offset', type=int, default=0)
  parser.add_argument('--datadir', type=str, default=Settings.FEATURES_DIR)
  parser.add_argument('--result_dir', type=str, default=Settings.RESULT_DIR)
  parser.add_argument('--norm', type=int, default=1)
  parser.add_argument('--opt', type=str, default='Adam',
                      choices=['MomentumSGD', 'Adam', 'AdaGrad', 'AdaDelta', 'RMSprop'])
  parser.add_argument('--weight_decay', type=float, default=0.0005)
  parser.add_argument('--alpha', type=float, default=0.001)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--lr_decay_freq', type=int, default=10)
  parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
  parser.add_argument('--rho', type=float, default=0.95)
  parser.add_argument('--eps', type=float, default=1e-6)
  parser.add_argument('--seed', type=int, default=2728)
  parser.add_argument('--data_index', type=int, default=0)
  parser.add_argument('--num_features', type=int, default=64)
  parser.add_argument('--site', type=str, default='yelp',
                      choices=['yelp', 'tripadvisor', 'movies'])
  parser.add_argument('--features', type=str, default='text_topics',
                      choices=['text_topics'])
  args = parser.parse_args()
  np.random.seed(args.seed)

  # create result dir
  log_fn, result_dir = create_result_dir(args)

  # create model and optimizer
  model, optimizer = get_model_optimizer(result_dir, args)

  # load data
  term = args.site + '_text_topics_2_channel_64_features_order_' + str(args.data_index)
  print (term)

  dataset = load_dataset(term, args.datadir)
  train_data, train_labels, test_data, test_labels = dataset

  N = train_data.shape[0]
  N_test = test_data.shape[0]

  # augmentation setting
  _norm = bool(args.norm)
  logging.info('norm:{}'.format(_norm))
  trans = Transform(norm=_norm)
  
  logging.info('start training...')

  # learning loop
  n_epoch = args.epoch
  batchsize = args.batchsize
  for epoch in range(1, n_epoch + 1):
      # train
      if args.opt == 'MomentumSGD':
          print('learning rate:', optimizer.lr)
          if epoch % args.lr_decay_freq == 0:
              optimizer.lr *= args.lr_decay_ratio

      sum_loss, sum_accuracy = train(train_data, train_labels, N,
                                     model, optimizer, args, trans)
      msg = 'epoch:{:02d}\ttrain mean loss={}, accuracy={}'.format(
          epoch + args.epoch_offset, sum_loss / N, sum_accuracy / N)
      logging.info(msg)
      print('\n%s' % msg)

      # validate
      sum_loss, sum_accuracy = validate(test_data, test_labels, N_test,
                                        model, args)
      msg = 'epoch:{:02d}\ttest mean loss={}, accuracy={}'.format(
          epoch + args.epoch_offset, sum_loss / N_test, sum_accuracy / N_test)
      logging.info(msg)
      print('\n%s' % msg)

      if epoch == 1 or epoch % args.snapshot == 0:
          model_fn = '%s/%s_%s_%s_dt_%s_epoch_%d.chainermodel' % (
              result_dir, args.site, args.features, args.prefix, 
              str(args.data_index), epoch + args.epoch_offset)
          #model.to_cpu()
          pickle.dump(model, open(model_fn, 'wb'), -1)

      draw_loss_curve(log_fn, '%s/%s_%s_%s_dt_%s_log.jpg' % (result_dir, args.site, 
                      args.features, args.prefix, str(args.data_index) ) )
