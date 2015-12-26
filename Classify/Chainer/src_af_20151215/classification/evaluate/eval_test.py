from __future__ import print_function
import os, sys
import imp
import time
import pickle
import numpy as np
sys.path.append(r'../utils')
import samples
from settings import Settings
import model_trainer
from nn_tools import setup_gpu
from preprocess import global_contrast_norm
import argparse

def load_model(model_dir, param, use_gpu=0):
    """
    Load model
    """
    model_fn = os.path.basename(model_dir)
    model_name = model_fn.split('.')[0]
    print (model_name, model_dir)
    module = imp.load_source(model_name, model_dir)
    model = pickle.load(open(os.path.join(model_dir, param), 'rb'))
    if use_gpu >= 0:
        model.to_gpu()
    else:
        model.to_cpu()
    return model

def validation(model, x_test, y_test, batchsize=128, use_gpu=0):
    """
    Validation
    """
    xp = setup_gpu(use_gpu)
    n_test = len(y_test)
    t = time.time()
    accum_err  = model_trainer.Accumlator()
    accum_acc  = model_trainer.Accumlator()
    n_batch = n_test/batchsize
    batchidxs = range(0, n_test, batchsize)
    n_minibatch_in_epoch = len(batchidxs)
    batch_logs_in_epoch = []
    for ibatch, i in enumerate(batchidxs):
        x_batch = xp.asarray(x_test[i:i+batchsize], dtype=xp.float32)
        y_batch = xp.asarray(y_test[i:i+batchsize], dtype=xp.int32)
        loss, acc, y_pred = model.forward(x_batch, y_batch, train=False, enable_dropout=False)
        err = float(loss.data)

        if not np.isfinite(err):
            print ('infinite loss detected. skipping minibatch.')
            continue

        accum_err.add(err, len(x_batch))
        acc_val = float(acc.data)
        accum_acc.add(acc_val, len(x_batch))

    dt = time.time() - t
    return accum_err.mean(), accum_acc.mean(), dt


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(description='RV Classification')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--index', '-index', default=0, type=int,
                        help='index of data')
    parser.add_argument('--label', '-lb', default=-1, type=int,
                    help='test label data')
    parser.add_argument('--norm', '-n', default=0, type=int,
                help='normalize data or not')
    parser.add_argument('--data-dir', '-dir', 
                        default=r'/home/zoro/work/Dataset/Features_bak_20151214', 
                        type=str, help='directory path of data')
    parser.add_argument('--model-dir', '-mdir', 
                    default=r'../models/rv_classification_models.py', 
                    type=str, help='model file')
    parser.add_argument('--site', default='yelp', type=str,
                        help='site for data')
    parser.add_argument('--param', default=None, type=str,
                        help='Trained param file')
    args = parser.parse_args()

    if args.param is None:
        args.param = r'/home/zoro/work/classify_result/old_divider_yelp_right_norm/rv_check_NetModel_FC_no_dropout_20151223_214444/model.parameters.pickle'
    #sample_creator_func = samples.RVForCheckSampleCreator()
    sample_creator_func = samples.RVTopicsSampleCreator()
    model = load_model(args.model_dir, args.param, args.gpu)
    
    test_data, test_labels, _, _ = sample_creator_func.load_samples(args.data_dir, args.site, args.index, 1, 1)
    if args.label >= 0:
        test_data = test_data[test_labels==args.label]
        test_labels = test_labels[test_labels==args.label]
    print ('label = ', test_data.shape)
    
    # Peform preprocess
    if args.norm >= 0:
        test_data = np.asarray(map(global_contrast_norm, test_data))
    print ('Finished preprocess with norm = ', args.norm)

    err, acc, dt = validation(model, test_data, test_labels, use_gpu=args.gpu)
    print (args.label, test_data.shape, err, acc, dt/test_labels.shape[0])
    test_labels, test_data = None, None
        
