#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model_util import *

class NetModel_FC(ClassificationModelBase):

    def __init__(self):
        n_units = 256
        super(NetModel_FC, self).__init__(
            l1 = F.Linear(64, 128),
            l2 = F.Linear(128, 256),
            lf = F.Linear(256, 5)
        )
        self.name = 'NetModel_FC'

    def apply(self, x_data, train, enable_dropout=False, finetune=False, verbose=False):
        def dropout(ratio):
            if ratio == 0.0:
                return lambda v: v
            return lambda v: F.dropout(v, train=train or enable_dropout, ratio=ratio)

        def _print_macro(desc, shape, verbose=verbose):
            if verbose:
                print desc, shape

        param = dict(test=not train, finetune=finetune)

        x = Variable(x_data, volatile=not train)

        h = F.relu(self.l1(x))
        #h = dropout(0.25)(h)

        h = F.relu(self.l2(h))
        #h = dropout(0.25)(h)
        
        h = self.lf(h)
        return h

    def start_finetuning(self):
        """
        Run batch normalization in finetuning mode
        it computes moving averages of mean and variance for evaluation
        during training, and normalizes the input using statistics
        """
        # self.bn1_1.start_finetuning()
        # self.bn1_2.start_finetuning()
        # self.bn2_1.start_finetuning()
        # self.bn2_2.start_finetuning()
        # return True

        return False
        
class NetModel_BN(ClassificationModelBase):

    def __init__(self):
        n_units = 256
        super(NetModel_BN, self).__init__(
            conv1=F.Convolution2D(1, 32, 2, stride=1, pad=1),
            bn1=F.BatchNormalization(32),
            conv2=F.Convolution2D(32, 32, 2, stride=1, pad=1),
            bn2=F.BatchNormalization(32),
            conv3=F.Convolution2D(32, 64, 2, stride=1, pad=1),
            lf=F.Linear(n_units, 5)
        )
        self.name = 'NetModel_BN'

    def apply(self, x_data, train, enable_dropout=False, finetune=False, verbose=False):
        def dropout(ratio):
            if ratio == 0.0:
                return lambda v: v
            return lambda v: F.dropout(v, train=train or enable_dropout, ratio=ratio)

        def _print_macro(desc, shape, verbose=verbose):
            if verbose:
                print desc, shape

        param = dict(test=not train, finetune=finetune)

        x = Variable(x_data, volatile=not train)

        h = F.relu(self.bn1(self.conv1(x), test = not train))
        h = F.max_pooling_2d(h, 2, stride=2)
#        h = dropout(0.25)(h)

        h = F.relu(self.bn2(self.conv2(h), test = not train))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.lf(h)
        return h

    def start_finetuning(self):
        """
        Run batch normalization in finetuning mode
        it computes moving averages of mean and variance for evaluation
        during training, and normalizes the input using statistics
        """
        # self.bn1_1.start_finetuning()
        # self.bn1_2.start_finetuning()
        # self.bn2_1.start_finetuning()
        # self.bn2_2.start_finetuning()
        # return True

        return False

