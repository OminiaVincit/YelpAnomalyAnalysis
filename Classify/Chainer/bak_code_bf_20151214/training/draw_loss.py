#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import matplotlib
if sys.platform in ['linux', 'linux2']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

def draw_loss_curve(logfile, outfile):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for line in open(logfile):
        line = line.strip()
        if 'Namespace' in line:
            batchsize = int(re.search('batchsize=([0-9]+)', line).groups()[0])
            alpha = float(re.search('alpha=([0-9\.]+)', line).groups()[0])
            eps = float(re.search('eps=([0-9\.e-]+)', line).groups()[0])
            rho = float(re.search('rho=([0-9\.]+)', line).groups()[0])

            lr = float(re.search('lr=([0-9\.]+)', line).groups()[0])
            lr_decay_freq = float(re.search('lr_decay_freq=([0-9]+)', line).groups()[0])
            lr_decay_ratio = float(re.search('lr_decay_ratio=([0-9\.]+)', line).groups()[0])
            weight_decay = float(re.search('weight_decay=([0-9\.]+)', line).groups()[0])

            seed = int(re.search('seed=([0-9]+)', line).groups()[0])
            opt = re.search('opt=\'([a-zA-Z]+)', line).groups()[0]
            model = re.search('model=\'([./_a-zA-Z]+)', line).groups()[0]

            site = re.search('site=\'([./_a-zA-Z]+)', line).groups()[0]
            features = re.search('features=\'([./_a-zA-Z]+)', line).groups()[0]
            
            data_index = int(re.search('data_index=([0-9]+)', line).groups()[0])
            norm = int(re.search('norm=([0-9]+)', line).groups()[0])
            num_features = int(re.search('num_features=([0-9]+)', line).groups()[0])
            gpu = int(re.search('gpu=([0-9/-]+)', line).groups()[0])
            
        if 'Number of samples' in line:
            train_samples = int(re.search('train:([0-9]+)', line).groups()[0])
            test_samples = int(re.search('test:([0-9]+)', line).groups()[0])

        if not 'epoch:' in line:
            continue
        epoch = int(re.search('epoch:([0-9]+)', line).groups()[0])
        if 'train' in line:
            tr_l = float(re.search('loss=(.+),', line).groups()[0])
            tr_a = float(re.search('accuracy=([0-9\.]+)', line).groups()[0])
            train_loss.append([epoch, tr_l])
            train_acc.append([epoch, tr_a])
        if 'test' in line:
            te_l = float(re.search('loss=(.+),', line).groups()[0])
            te_a = float(re.search('accuracy=([0-9\.]+)', line).groups()[0])
            test_loss.append([epoch, te_l])
            test_acc.append([epoch, te_a])

    train_loss = np.asarray(train_loss)
    test_loss = np.asarray(test_loss)
    train_acc = np.asarray(train_acc)
    test_acc = np.asarray(test_acc)

    if not len(train_acc) > 2:
        return

    N = train_acc.shape[0]
    
    fig, ax1 = plt.subplots()
    p1, = ax1.plot(train_loss[:, 0], train_loss[:, 1], label='training loss')
    p2, = ax1.plot(test_loss[:, 0], test_loss[:, 1], label='test loss')
    ax1.set_xlim([train_acc[0, 0], train_acc[N-1, 0]])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    ax2 = ax1.twinx()
    p3, = ax2.plot(train_acc[:, 0], train_acc[:, 1], label='training accuracy', c='r')
    p4, = ax2.plot(test_acc[:, 0], test_acc[:, 1], label='test accuracy', c='c')
    ax2.set_xlim([train_acc[0, 0], train_acc[N-1, 0]])
    ax2.set_ylim([0, 1.01])
    ax2.set_ylabel('accuracy')

    ax1.legend(bbox_to_anchor=(0.25, -0.1), loc=9)
    ax2.legend(bbox_to_anchor=(0.75, -0.1), loc=9)

    # Draw results at last change point
    ep_iter = train_acc[N-1, 0] + int( train_acc[N-1, 0] / 8 )
    
    ax2.text(ep_iter, 0.70, 'Site=%s' % site, size=10)
    ax2.text(ep_iter, 0.65, 'Features=%s, num_features=%d' % (features, num_features), size=10)
    ax2.text(ep_iter, 0.60, 'Norm=%d, GPU=%d' % (norm, gpu), size=10)

    ax2.text(ep_iter, 0.55, 'Model=%s' % model, size=10)
    ax2.text(ep_iter, 0.50, 'Batchsize=%d, seed=%d' % (batchsize, seed), size=10)
    ax2.text(ep_iter, 0.45, 'Data_index=%s' % data_index, size=10)
    ax2.text(ep_iter, 0.40, 'Num samples: train=%d, test=%d' % (train_samples, test_samples), size=10)

    ax2.text(ep_iter, 0.35, 'Opt=%s, lr=%.4f' % (opt, lr), size=10)
    ax2.text(ep_iter, 0.30, 'lr_decay_freq=%d, lr_decay_ratio=%.2f' % (lr_decay_freq, lr_decay_ratio), size=10)
    ax2.text(ep_iter, 0.25, 'weight_decay=%.4f, alpha=%.4f' % (weight_decay, alpha), size=10)
    ax2.text(ep_iter, 0.20, 'eps=%f, rho=%.3f' % (eps, rho), size=10)

    ax2.text(ep_iter, 0.80, 'Train loss=%.4f' % train_loss[ N-1, 1], color=p1.get_color(), size=10)
    ax2.text(ep_iter, 0.85, 'Test loss=%.4f' % test_loss[ N-1, 1], color=p2.get_color(), size=10)
    ax2.text(ep_iter, 0.90, 'Test acc=%.4f' %  test_acc[ N-1, 1], color=p4.get_color(), size=10)
    ax2.text(ep_iter, 0.95, 'Train acc=%.4f' % train_acc[ N-1, 1], color=p3.get_color(), size=10)
    ax2.text(ep_iter, 1.0, 'Epoch=%d' % N, size=10)

    plt.savefig(outfile, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', '-f', type=str)
    parser.add_argument('--outfile', '-o', type=str)
    args = parser.parse_args()
    print(args)

    draw_loss_curve(args.logfile, args.outfile)