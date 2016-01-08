#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model_util import *

class NetModel_FC_nodrop_512_1024(ClassificationModelBase):

    def __init__(self):
        super(NetModel_FC_nodrop_512_1024, self).__init__(
            l1 = F.Linear(256, 512),
            l2 = F.Linear(512, 1024),
            lf = F.Linear(1024, 5)
        )
        self.name = 'NetModel_FC_nodrop_512_1024'

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
        h = F.relu(self.l2(h))
        
        h = self.lf(h)
        return h

    def start_finetuning(self):
        """
        Run batch normalization in finetuning mode
        it computes moving averages of mean and variance for evaluation
        during training, and normalizes the input using statistics
        """
        return False

class NetModel_FC_drop_512_1024(ClassificationModelBase):

    def __init__(self):
        super(NetModel_FC_drop_512_1024, self).__init__(
            l1 = F.Linear(256, 512),
            l2 = F.Linear(512, 1024),
            lf = F.Linear(1024, 5)
        )
        self.name = 'NetModel_FC_drop_512_1024'

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
        h = dropout(0.5)(h)

        h = F.relu(self.l2(h))
        h = dropout(0.5)(h)

        h = self.lf(h)
        return h

    def start_finetuning(self):
        """
        Run batch normalization in finetuning mode
        it computes moving averages of mean and variance for evaluation
        during training, and normalizes the input using statistics
        """
        return False

class NetModel_BN_more_linear(ClassificationModelBase):

    def __init__(self):
        n_units = 256
        m_units = 512
        super(NetModel_BN_more_linear, self).__init__(
            conv1=F.Convolution2D(1, 32, 2, stride=1, pad=1),
            bn1=F.BatchNormalization(32),
            conv2=F.Convolution2D(32, 32, 2, stride=1, pad=1),
            bn2=F.BatchNormalization(32),
            conv3=F.Convolution2D(32, 64, 2, stride=1, pad=1),
            bn3=F.BatchNormalization(64),
            l1 = F.Linear(n_units, m_units),
            l2 = F.Linear(m_units, m_units),
            lf=F.Linear(m_units, 5)
        )
        self.name = 'NetModel_BN_more_linear'

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

        h = F.relu(self.bn2(self.conv2(h), test = not train))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bn3(self.conv3(h), test = not train))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.l1(h))
        h = dropout(0.5)(h)

        h = F.relu(self.l2(h))
        h = dropout(0.5)(h)

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

class NetModel_5_linears(ClassificationModelBase):

    def __init__(self):
        n_features = 256
        super(NetModel_5_linears, self).__init__(
            l1 = F.Linear(n_features, 1200),
            bn1=F.BatchNormalization(1200),   
            l2 = F.Linear(1200, 600, nobias=True),
            bn2=F.BatchNormalization(600),
            l3 = F.Linear(600, 300, nobias=True),
            bn3=F.BatchNormalization(300),
            l4 = F.Linear(300, 150, nobias=True),
            bn4=F.BatchNormalization(150),
            lf=F.Linear(150, 5),
            bn5=F.BatchNormalization(5),
        )
        self.name = 'NetModel_5_linears'

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
        h = x
        h = F.relu(self.bn1(self.l1(h), test = not train))
        h = F.relu(self.bn2(self.l2(h), test = not train))
        h = F.relu(self.bn3(self.l3(h), test = not train))
        h = F.relu(self.bn4(self.l4(h), test = not train))
        h = F.relu(self.bn5(self.lf(h), test = not train))
        
        #h = dropout(0.5)(h)
        #h = self.lf(h)
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