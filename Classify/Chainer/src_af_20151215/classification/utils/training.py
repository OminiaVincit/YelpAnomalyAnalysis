#!env python
import os
import sys

import imp
import datetime
import glob
import argparse
import shutil
import collections
import time
import copy
# import atexit
import scipy as sp
# import scipy.signal
# import skimage.transform
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import colorama

import training_logdb
import nn_tools

from model_util import *
import model_viewer
import experiment_util
import model_trainer
from preprocess import global_contrast_norm, row_contrast_norm

def load_or_create_experiment(task_name, result_base_dir, resume_dir=None, model_desc_str=None, term=''):
    u'''load (if resume_dir is not None) or create a new experiment'''
    if resume_dir is not None:
        # load preious result directory
        experiment = experiment_util.ExperimentResult.load_dir(resume_dir)
        model_desc = experiment.get_model_desc()
        print 'Loaded experiment directory:', experiment.path
    else:
        # create result directory
        model_desc = experiment_util.ModelDescription.from_desc_str(model_desc_str)
        experiment = experiment_util.ExperimentResult(os.path.join(result_base_dir, '{}_{}_{}_{}'.format(
            task_name, model_desc.class_name, term, experiment_util.now().strftime('%Y%m%d_%H%M%S'))))
        shutil.copyfile(model_desc.file_path, experiment.model_structure_path)
        print 'Created experiment directory:', experiment.path
        experiment.set_model_desc(model_desc)
    return experiment


def create_optimizer_from_args(args):
    if args.optimizer == 'Adam':
        return optimizers.Adam(args.alpha)
    if args.optimizer == 'AdaDelta':
        return optimizers.AdaDelta()
    if args.optimizer == 'AdaGrad':
        return optimizers.AdaGrad()
    if args.optimizer == 'MomentumSGD':
        return optimizers.MomentumSGD(lr=args.lr, momentum=args.momentum)
    raise ValueError('unknown optimizer {}'.format(args.optimizer))


def main():
    # arguments
    parser = argparse.ArgumentParser(description='IVC Segmentation')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--training', '-t', default=100, type=int,
                        help='training epoch (<=0 value indicates no training)')
    parser.add_argument('--save-interval', '-s', default=50, type=int,
                        help='save the model after every <save_interval> epoch.')
    parser.add_argument('--plot-interval', '-p', default=50, type=int,
                        help='plot the model after every <plot_interval> epoch.')
    parser.add_argument('--backup-interval', '-b', default=50, type=int,
                        help='backup the model after every <backup_interval> epoch.')
    parser.add_argument('--verbose', '-v', default=-1, type=int,
                        help='verbose output of training progress on every <verbose> minibatch')
    parser.add_argument('--resume', '-r', default=None, type=str,
                        help='directory path of previous experiment')
    parser.add_argument('--continue', '-c', default=False, action='store_true',
                        help='continue the last experiment.')
    parser.add_argument('--site', default='yelp', type=str,
                        help='site for data')
    parser.add_argument('--model', default=None, type=str,
                        help='initial model description. e.g.) models.rv_classification_models@NetModel_BN')
    parser.add_argument('--optimizer', default='Adam', type=str,
                        help='optimizer')
    parser.add_argument('--alpha', default=0.001, type=float,
                        help='alpha of Adam')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=None, type=float,
                        help='weight decay')
    parser.add_argument('--task', default='rv_topics', type=str, required=True,
                        help='task name')
    parser.add_argument('--lr-decay', default=0.90, type=float,
                        help='lr decay rate per 10000 sample')
    parser.add_argument('--index', '-idx', default=0, type=int,
                    help='data index')
    parser.add_argument('--ftype', '-ft', default='TOPICS_64', type=str,
                    help='features type')
    parser.add_argument('--seed', default=1, type=int,
                    help='seed for random')
    parser.add_argument('--train-down-rate', default=1.0, type=float,
                    help='train downsampling rate')
    parser.add_argument('--test-down-rate', default=1.0, type=float,
                    help='test downsampling rate')
    parser.add_argument('--norm', type=int, default=-1)
    parser.add_argument('--result-dir', default=None, type=str,
                    help='result directory')

    args = parser.parse_args()

    # learning settings
    max_grad_norm = -1
    backup_weights = False
    calc_test_dropout = False
    np.random.seed(args.seed)
    
    # GPU
    xp = nn_tools.setup_gpu(args.gpu)
    
    n_epoch = args.training
    do_training = n_epoch > 0

    # --------------------------------------
    # task
    import tasks
    task = tasks.get_task_settings(args.task)
    task.show()

    # --------------------------------------
    # initialize experiment
    term = '{}_{}'.format(args.site, args.index)
    model_desc_str = args.model if args.model is not None else task.default_model_desc
    if args.result_dir is not None:
        task.result_dir = os.path.join(task.result_dir, args.result_dir)
    if do_training:
        resume_dir = args.resume
        if getattr(args, 'continue'):
            resume_dir = experiment_util.get_last_experiment()
        experiment = load_or_create_experiment(args.task, task.result_dir, resume_dir, model_desc_str, term)
        model_desc = experiment.get_model_desc()
    else:
        model_desc = experiment_util.ModelDescription.from_desc_str(model_desc_str)

    # prepare model module
    model_desc.import_module_global()
    model_type = model_desc.get_model_class()
    print 'Loaded model file: %s, model class: %s' % (model_desc.file_path, model_desc.class_name)

    # --------------------------------------
    # load or create model object
    model_file = experiment.model_parameters_path
    training_log_file = experiment.training_log_path
    if os.path.isfile(model_file):
        print 'Loading model object..'
        model = misc_util.retrying_pickle_load(model_file)
    else:
        print 'Creating model object..'
        model = model_type()
        do_training = True
    experiment.model = model

    if args.gpu >= 0:
        model.to_gpu()

    # default settings and overriding.
    if args.weight_decay is not None:
        task.settings['weight_decay'] = args.weight_decay

    # --------------------------------------
    # create or open log
    if do_training:
        log = training_logdb.TrainingLog()
        log.open(training_log_file, task_type='regression' if task.is_regression else 'classification')
        experiment.log = log
        log.store_value('task', task)
        log.store_value('args', args)
        log.store_value('model_desc', model_desc.class_name)

    # --------------------------------------
    # load samples
    if do_training:

        # Dataset setting
        sample_creator = task.sample_creator_func()
        # x_train, y_train, x_test, y_test = \
        #     sample_creator.load_samples(task.dataset_dir, args.site, args.index, 
        #                                 args.train_down_rate, args.test_down_rate)

        x_train, y_train, x_test, y_test = \
            sample_creator.load_samples(task.dataset_dir, args.site, args.ftype, args.index)

        print 'x_train', x_train.shape, x_train.nbytes/1024/1024, 'Mbytes ', 'y_train', y_train.shape, y_train.nbytes/1024, 'Kbs'
        print 'x_test',  x_test.shape,  x_test.nbytes/1024/1024,  'Mbytes ', 'y_test',  y_test.shape,  y_test.nbytes/1024,  'Kbs'

        # Peform preprocess
        if args.norm > 0:
            x_train = np.asarray(map(global_contrast_norm, x_train))
            x_test = np.asarray(map(global_contrast_norm, x_test))
        elif args.norm == 0:
            x_train = np.asarray(map(row_contrast_norm, x_train))
            x_test = np.asarray(map(row_contrast_norm, x_test))
        print 'Finished preprocess with norm = ', args.norm        

        log.store_value('test_samples', len(y_test))
        log.store_value('train_samples', len(y_train))
        
    # --------------------------------------
    # training
    if do_training:
        # Setup optimizer
        optimizer = create_optimizer_from_args(args)
        optimizer.setup(model)
        print 'Optimizer:', optimizer.__class__.__name__

        # trainer
        def create_trainer(mod, sz):
            t = model_trainer.ModelTrainer(mod, optimizer,
                    task.settings['weight_decay'], max_grad_norm, sz, args.verbose, calc_test_dropout, backup_weights, task.is_regression,
                    args.gpu, None, log)  # off the bachlog file
            return t

        batchsize = model.max_batchsize
        print 'batch size:', batchsize
        trainer = create_trainer(model, batchsize)

        # start
        print colorama.Back.RED + 'Training with model: {}'.format(os.path.basename(model_desc.module_name)) + colorama.Back.RESET

        # Learning loop
        epoch_range = (model.total_epoch + 1, model.total_epoch + n_epoch + 1)
        lr = args.lr
        lr_scale = 1.0
        sample_count = 0
        print 'Epoch range: [%d, %d)' % epoch_range

        for epoch in xrange(*epoch_range):
            print 'epoch={}/[{}, {}) batchsize={} lr_scale={} lr={} weight_decay={}'.format(
                    epoch, epoch_range[0], epoch_range[1], batchsize, lr_scale, lr, task.settings['weight_decay'])
            tepoch = time.time()

            # training & validation
            epoch_res = trainer.epoch(epoch, x_train, y_train, x_test, y_test)

            print 'epoch {} done in {:.2f}s'.format(epoch, time.time() - tepoch)

            # post-processing
            experiment.touch_epoch_file()
            if epoch % args.save_interval == 0 or epoch % args.backup_interval == 0 or epoch % args.plot_interval == 0:
                cpu_model = copy_model_to_cpu(model)
            else:
                cpu_model = None
            if epoch % args.save_interval == 0:
                print 'save model'
                nn_tools.retrying_pickle_dump(cpu_model, model_file)
            if epoch % args.backup_interval == 0:
                print 'backup model'
                nn_tools.retrying_pickle_dump(cpu_model, experiment.get_model_backup_path(cpu_model))
            if epoch % args.plot_interval == 0:
                print 'plot model'
                model_viewer.plot_model_layer_images_and_close(model_file, cpu_model)

            # learning rate decay for MomentumSGD
            if isinstance(optimizer, optimizers.MomentumSGD):
                sample_count += len(x_train)
                while sample_count > 10000:
                    lr_scale *= args.lr_decay
                    lr = args.lr*lr_scale
                    sample_count -= 10000
                    optimizer.lr = lr

        # finally save and plot the model.
        cpu_model = copy_model_to_cpu(model)
        nn_tools.retrying_pickle_dump(cpu_model, model_file)
        model_viewer.plot_model_layer_images_and_close(model_file, cpu_model)

if __name__=='__main__':
    main()

