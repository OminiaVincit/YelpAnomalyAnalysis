#!env python
import time

from colorama import Fore, Back, Style
import numpy as np
from chainer import cuda

import model_util
from nn_tools import setup_gpu
from training_logdb import log_entry, create_norm, create_cost, MeasureType, DataType, LogType

class Accumlator(object):
    u'''accumlate values. provides access to the last value and the mean.'''
    def __init__(self):
        self.last_value = 0.0
        self.accum_value = 0.0
        self.size = 0
    def add(self, val, size):
        assert size > 0
        self.last_value = val
        self.accum_value += val * size
        self.size += size
    def mean(self):
        if self.size > 0:
            return self.accum_value / self.size
        return None
    def last(self):
        if self.size > 0:
            return self.last_value
        return None


class ModelTrainer(object):
    class IterType(object):
        TRAIN_DROPOUT, TRAIN_APPLY, TEST_DROPOUT, TEST_APPLY, FINETUNE = 0, 1, 2, 3, 4

    def __init__(self, model, optimizer, weight_decay, max_grad_norm, batchsize, verbose, calc_test_dropout, backup_weights, is_regression, use_gpu, batch_log, epoch_log):
        self.model = model
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.batchsize = batchsize
        self.verbose = verbose
        self.calc_test_dropout = calc_test_dropout
        self.backup_weights = backup_weights
        self.is_regression = is_regression
        self.iter_callback = None
        self.iter_callback_interval = 0
        self.batch_log = batch_log
        self.epoch_log = epoch_log
        self.use_gpu = use_gpu
        self.iepoch = -1

    def set_regression_tolerance(self, tol):
        self.regression_tolerance_2 = tol*tol

    def set_iter_callback(self, interval, callback):
        self.iter_callback_interval = interval
        self.iter_callback = callback

    def training(self, x_train, y_train):
        u'''optimizer for the whole dataset as an epoch.'''
        xp = setup_gpu(self.use_gpu)
        n_train = len(y_train)
        perm = np.random.permutation(n_train)
        t = time.time()
        accum_err  = Accumlator()
        accum_acc  = Accumlator()
        accum_grad = Accumlator()
        accum_norm = Accumlator()
        accum_ratio = Accumlator()
        n_batch = n_train/self.batchsize
        if self.backup_weights:
            sane_params = self.model.parameters
        batchidxs = range(0, n_train, self.batchsize)
        n_minibatch_in_epoch = len(batchidxs)
        batch_logs_in_epoch = []
        for ibatch, i in enumerate(batchidxs):

            x_batch = xp.asarray(x_train[perm[i:i+self.batchsize]], dtype=xp.float32)
            y_batch = xp.asarray(y_train[perm[i:i+self.batchsize]], dtype=xp.int32)

            #print y_batch.shape
            #print x_batch.dtype, y_batch.dtype, type(x_batch), type(y_batch), x_batch.ctypes, y_batch.ctypes

            self.optimizer.zero_grads()
            loss, acc, _ = self.model.forward(x_batch, y_batch, True)
            err = float(loss.data)

            if self.backup_weights:
                if not np.isfinite(err):
                    print 'infinite loss detected. skipping minibatch.'
                    self.model.copy_parameters_from(sane_params)
                    continue
                else:
                    sane_params = self.model.parameters

            loss.backward()
            if self.max_grad_norm > 0:
                self.optimizer.clip_grads(self.max_grad_norm*self.optimizer.tuples[0][1].size) # clip_grads accept unnormalized norm
            g_norm_val = self.optimizer.compute_grads_norm()
            accum_grad.add(g_norm_val, len(x_batch))
            w_norm_val = model_util.compute_weight_norm(self.model)
            accum_norm.add(w_norm_val, len(x_batch))
            ratio_val = g_norm_val / w_norm_val
            accum_ratio.add(ratio_val, len(x_batch))
            if self.weight_decay > 0:
                self.optimizer.weight_decay(self.weight_decay)
            self.optimizer.update()

            accum_err.add(err, len(x_batch))
            acc_val = float(acc.data)
            accum_acc.add(acc_val, len(x_batch))

            if self.verbose > 0 and ibatch % self.verbose == 0 and i > 0:
                self._print_iter_progress(ModelTrainer.IterType.TRAIN_DROPOUT, ibatch, n_batch, accum_err.last(), accum_grad.last(), accum_norm.last(), accum_acc.last())

            if self.iter_callback is not None and ibatch % self.iter_callback_interval == 0:
                self._per_iter_hook()

            # logging
            if self.iepoch >= 0:
                norm = create_norm(grad=g_norm_val, weight=w_norm_val, grad_ratio=ratio_val)
                cost = create_cost(loss=err, accuracy=acc_val)
                batch_logs_in_epoch.append(log_entry(total_epoch=self.iepoch, inepoch_minibatch=ibatch, n_minibatch_in_epoch=n_minibatch_in_epoch,
                    log_type=LogType.minibatch, data_type=DataType.train, measure_type=MeasureType.training,
                    cost=cost, norm=norm,
                    ))

            self.model.increment_minibatch(self.batchsize)

        dt = time.time() - t

        self.batch_log.add_logs(batch_logs_in_epoch) # I/O at once.

        self._print_epoch_progress(ModelTrainer.IterType.TRAIN_DROPOUT, dt, n_train, accum_err.mean(), accum_grad.mean(), accum_norm.mean(), accum_acc.mean())
        return accum_err.mean(), accum_acc.mean(), accum_grad.mean()


    def validation(self, x_test, y_test, iter_type):
        """validation process (not dropouts, gradient calculation, batch normalization updates, etc.) for {training,test} set."""
        xp = setup_gpu(self.use_gpu)
        n_test = len(y_test)
        t = time.time()
        accum_err  = Accumlator()
        accum_acc  = Accumlator()
        n_batch = n_test/self.batchsize
        if iter_type == ModelTrainer.IterType.TRAIN_DROPOUT:
            data_type    = DataType.train
            measure_type = MeasureType.training
        if iter_type == ModelTrainer.IterType.TRAIN_APPLY:
            data_type    = DataType.train
            measure_type = MeasureType.validation
        if iter_type == ModelTrainer.IterType.TEST_DROPOUT:
            data_type    = DataType.test
            measure_type = MeasureType.training
        if iter_type == ModelTrainer.IterType.TEST_APPLY:
            data_type    = DataType.test
            measure_type = MeasureType.validation
        batchidxs = range(0, n_test, self.batchsize)
        n_minibatch_in_epoch = len(batchidxs)
        batch_logs_in_epoch = []
        for ibatch, i in enumerate(batchidxs):
            x_batch = xp.asarray(x_test[i:i+self.batchsize], dtype=xp.float32)
            y_batch = xp.asarray(y_test[i:i+self.batchsize], dtype=xp.int32)
            loss, acc, y_pred = self.model.forward(x_batch, y_batch, train=False, enable_dropout=(iter_type == ModelTrainer.IterType.TEST_DROPOUT))
            err = float(loss.data)

            if not np.isfinite(err):
                print 'infinite loss detected. skipping minibatch.'
                continue

            accum_err.add(err, len(x_batch))
            acc_val = float(acc.data)
            accum_acc.add(acc_val, len(x_batch))

            # logging
            cost = create_cost(loss=err, accuracy=acc_val)
            batch_logs_in_epoch.append(log_entry(total_epoch=self.iepoch, inepoch_minibatch=ibatch, n_minibatch_in_epoch=n_minibatch_in_epoch,
                log_type=LogType.minibatch, data_type=data_type, measure_type=measure_type,
                cost=cost,
                ))

            if self.verbose > 0 and ibatch % self.verbose == 0 and i > 0:
                self._print_iter_progress(iter_type, ibatch, n_batch, accum_err.last(), None, None, accum_acc.last())

            if self.iter_callback is not None and ibatch % self.iter_callback_interval == 0:
                self._per_iter_hook()

        dt = time.time() - t
        self.batch_log.add_logs(batch_logs_in_epoch) # I/O at once.
        self._print_epoch_progress(iter_type, dt, n_test, accum_err.mean(), None, None, accum_acc.mean())
        return accum_err.mean(), accum_acc.mean()

    def finetune_batch_normalization(self, x_train):
        u'''fine-tune batch normalization.'''
        try:
            if not self.model.start_finetuning(): return
        except NameError:
            return
        xp = setup_gpu(args.use_gpu)
        n_train = len(x_train)
        perm = np.random.permutation(n_train)
        t = time.time()
        for i in xrange(0, n_train, self.batchsize):
            x_batch = xp.asarray(x_train[perm[i:i+self.batchsize]])
            y_pred = self.model.apply(x_batch, False, enable_dropout=False, finetune=True)

        dt = time.time() - t
        self._print_epoch_progress(ModelTrainer.IterType.FINETUNE, dt, n_train, None, None, None, None)


    def epoch(self, iepoch, x_train, y_train, x_test, y_test):
        u'''training & validation set for as an epoch.'''
        self.iepoch = iepoch
        dropout_train_err, dropout_train_acc, train_grad = self.training(x_train, y_train)
        # finetune the model.
        self.finetune_batch_normalization(x_train)
        # feed TRAIN data to the validation process to compare the results with real validation.
        train_err, train_acc = self.validation(x_train, y_train, ModelTrainer.IterType.TRAIN_APPLY)
        # someone may need dropout loss on validation set.
        if self.calc_test_dropout:
            dropout_test_err, dropout_test_acc = self.validation(x_test, y_test, ModelTrainer.IterType.TEST_DROPOUT)
        # usual validation.
        test_err, test_acc = self.validation(x_test, y_test, ModelTrainer.IterType.TEST_APPLY)

        self.epoch_log.add_log(log_entry(total_epoch=iepoch,
            log_type=LogType.epoch, data_type=DataType.train, measure_type=MeasureType.validation,
            cost=create_cost(loss=train_err, accuracy=train_acc),
            ))
        self.epoch_log.add_log(log_entry(total_epoch=iepoch,
            log_type=LogType.epoch, data_type=DataType.test, measure_type=MeasureType.validation,
            cost=create_cost(loss=test_err, accuracy=test_acc),
            ))
        self.epoch_log.flush()
        self.batch_log.flush()

        res = dict(train_err=train_err, train_grad=train_grad, test_err=test_err)
        if train_acc is not None:
            res.update(train_acc=train_acc)
        if test_acc is not None:
            res.update(test_acc=test_acc)

        self.model.increment_epoch()
        return res

    def _print_style(self, iter_type):
        if iter_type == ModelTrainer.IterType.TRAIN_DROPOUT:
            color = Fore.CYAN
            title = 'train (w/ dropout)'
        elif iter_type == ModelTrainer.IterType.TRAIN_APPLY:
            color = Fore.GREEN
            title = 'train (apply)'
        elif iter_type == ModelTrainer.IterType.TEST_DROPOUT:
            color = Fore.YELLOW
            title = 'test  (w/ dropout)'
        elif iter_type == ModelTrainer.IterType.TEST_APPLY:
            color = Fore.RED
            title = 'test  (apply)'
        elif iter_type == ModelTrainer.IterType.FINETUNE:
            color = Fore.WHITE
            title = 'finetune'
        return color, title

    def _per_iter_hook(self):
        if self.iter_callback:
            self.iter_callback()

    def _print_iter_progress(self, iter_type, ibatch, n_batch, err, optional_grad=None, optional_norm=None, optional_acc=None):
        color, title = self._print_style(iter_type)
        s = [color]
        s.append('{:18} batch={:4}/{:4}: err={:.5f}'.format(title, ibatch, n_batch, err))
        if optional_acc is not None:
            s.append('acc={:.5f}'.format(optional_acc))
        if optional_grad is not None:
            s.append('grad={:.3e}'.format(optional_grad))
        if optional_norm is not None:
            s.append('norm={:.3e}'.format(optional_norm))
        s.append(Fore.RESET + Back.RESET + Style.RESET_ALL)
        print ' '.join(s)

    def _print_epoch_progress(self, iter_type, dt, n_samples, optional_err=None, optional_grad=None, optional_norm=None, optional_acc=None):
        color, title = self._print_style(iter_type)
        color += Style.BRIGHT
        s = [color]
        s.append('{:32}:'.format(title))
        if optional_err is not None:
            s.append('err={:.5f}'.format(optional_err))
        if optional_acc is not None:
            s.append('acc={:.5f}'.format(optional_acc))
        if optional_grad is not None:
            s.append('grad={:.3e}'.format(optional_grad))
        if optional_norm is not None:
            s.append('norm={:.3e}'.format(optional_norm))
        s.append('(elapsed {:.2f}s = {:.2f}ms/sample)'.format(dt, 1.0e3*dt/n_samples))
        s.append(Fore.RESET + Back.RESET + Style.RESET_ALL)
        print ' '.join(s)
        print '-'*18

def adapt_batchsize(create_trainer, optimizer, model, x_train, y_train, sizes):
    """
    create_trainer = lambda size: return trainer
    sizes = [batchsize]
    """

    print 'Find best batchsize..'
    multiplier = 2
    perf = []
    optimizer.setup(model)
    for sz in sizes:
        if sz > model.max_batchsize: break
        tmp_trainer = create_trainer(sz)
        t = time.time()
        tmp_trainer.training(x_train[:sz*multiplier], y_train[:sz*multiplier])
        time_per_sample = (time.time() - t) / (sz*multiplier)
        perf.append((time_per_sample, sz))
    perf.sort()
    batchsize = perf[0][1]
    print 'Found the best batchsize: %d (training time: %.2f ms/sample)' % (perf[0][1], perf[0][0]*1.0e3)
    return batchsize


