#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model_util import *

class NetModel_FC_drop_128_256(ClassificationModelBase):

    def __init__(self):
        super(NetModel_FC_drop_128_256, self).__init__(
            l1 = F.Linear(64, 128),
            l2 = F.Linear(128, 256),
            lf = F.Linear(256, 5)
        )
        self.name = 'NetModel_FC_drop_128_256'

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
        h = dropout(0.25)(h)

        h = F.relu(self.l2(h))
        h = dropout(0.25)(h)
        
        h = self.lf(h)
        return h

    def start_finetuning(self):
        """
        Run batch normalization in finetuning mode
        it computes moving averages of mean and variance for evaluation
        during training, and normalizes the input using statistics
        """
        return False

class NetModel_FC_tfidf(ClassificationModelBase):

    def __init__(self):
        super(NetModel_FC_tfidf, self).__init__(
            l1 = F.Linear(1024, 256),
            l2 = F.Linear(256, 512),
            lf = F.Linear(512, 5)
        )
        self.name = 'NetModel_FC_tfidf'

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

class NetModel_FC_topics(ClassificationModelBase):

    def __init__(self):
        super(NetModel_FC_topics, self).__init__(
            l1 = F.Linear(64, 256),
            l2 = F.Linear(256, 512),
            lf = F.Linear(512, 5)
        )
        self.name = 'NetModel_FC_topics'

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

class NetModel_FC_STR(ClassificationModelBase):

    def __init__(self):
        super(NetModel_FC_STR, self).__init__(
            l1 = F.Linear(13, 256),
            l2 = F.Linear(256, 512),
            lf = F.Linear(512, 5)
        )
        self.name = 'NetModel_FC_STR'

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

class NetModel_FC_LIWC(ClassificationModelBase):

    def __init__(self):
        super(NetModel_FC_LIWC, self).__init__(
            l1 = F.Linear(64, 256),
            l2 = F.Linear(256, 512),
            lf = F.Linear(512, 5)
        )
        self.name = 'NetModel_FC_LIWC'

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

class NetModel_FC_INQUIRER(ClassificationModelBase):

    def __init__(self):
        super(NetModel_FC_INQUIRER, self).__init__(
            l1 = F.Linear(182, 256),
            l2 = F.Linear(256, 512),
            lf = F.Linear(512, 5)
        )
        self.name = 'NetModel_FC_INQUIRER'

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

class NetModel_FC_GALC(ClassificationModelBase):

    def __init__(self):
        super(NetModel_FC_GALC, self).__init__(
            l1 = F.Linear(39, 256),
            l2 = F.Linear(256, 512),
            lf = F.Linear(512, 5)
        )
        self.name = 'NetModel_FC_GALC'

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