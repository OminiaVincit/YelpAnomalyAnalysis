#!env python
#print 'model viewer: usage: model_viewer.py model_xxx.pickle'

import os
import sys

import argparse
import collections
import time
import cPickle as pickle
import scipy as sp
import scipy.signal
import skimage.transform
import numpy as np
import matplotlib
if sys.platform in ['linux', 'linux2']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import imp

# import preprocess

def plot_1x1_convolution_layer(W):
    u'''plot 1x1 conv layer in a special format'''
    out_ch, in_ch, y, x = W.shape
    assert y*x == 1

    W = W.reshape(out_ch, 1, y*x, in_ch) 
    return plot_filters(W, normalize='row' if in_ch > 1 else None)

def plot_1ch_input_convolution_layer(W):
    u'''plot 1ch input -> N ch output conv layer'''

    out_ch, in_ch, y, x = W.shape
    assert in_ch == 1
    rows = int(np.sqrt(out_ch))
    cols = (out_ch + rows - 1)/rows

    filters = []
    for irow in range(rows):
        for icol in range(cols):
            i = irow*cols + icol
            if i < out_ch:
                filters.append(W[i, 0, :, :])
            else:
                filters.append(np.zeros((y, x)))
    W_square = np.array(filters).reshape((rows, cols, y, x))
    fig, ax = plot_filters(W_square, normalize='kernel', show_ticks=False)

    ax.axis('off')
    return fig, ax

def plot_filters(W, normalize='row', show_ticks=True):
    # manually embed kernels to a single image to reduce heavy matplotlib figure operation.
    out_ch, in_ch, y, x = W.shape
    spacing_px = 1
    img_h = y*out_ch + spacing_px*(out_ch + 1)
    img_w = x*in_ch + spacing_px*(in_ch + 1)
    img = np.ones((img_h, img_w, 3), np.float32) * [10/255., 10/255., 40/255.]
    for iout_ch in range(out_ch):
        if normalize == 'row':
            vmin = W[iout_ch, :, :, :].min()
            vmax = W[iout_ch, :, :, :].max()
        for iin_ch in range(in_ch):
            if normalize == 'kernel':
                vmin = W[iout_ch, iin_ch, :, :].min()
                vmax = W[iout_ch, iin_ch, :, :].max()
            w = W[iout_ch, iin_ch, :, :]
            if normalize is not None:
                if vmin < vmax:
                    w = (w - vmin) / (vmax - vmin)
                else:
                    w = w*0 + 0.5
            img[y*iout_ch + spacing_px*(iout_ch + 1):, x*iin_ch + spacing_px*(iin_ch + 1):, :][:y, :x, :] = w[:, :, np.newaxis]

    xticks, yticks = [], []
    xticklabels, yticklabels = [], []
    for iout_ch in range(out_ch):
        yticks.append(y*iout_ch + y*0.5 + spacing_px*(iout_ch + 1))
        yticklabels.append('%d' % iout_ch)
    for iin_ch in range(in_ch):
        xticks.append(x*iin_ch + x*0.5 + spacing_px*(iin_ch + 1))
        xticklabels.append('%d' % iin_ch)

    # figure size
    max_figsize = 8.0
    if in_ch*x > out_ch*y:
        fig_x = max_figsize
        fig_y = max_figsize*(out_ch*y)/(in_ch*x)
    else:
        fig_y = max_figsize
        fig_x = max_figsize*(in_ch*x)/(out_ch*y)

    fig, ax = plt.subplots(1, 1, figsize=(fig_x, 1 + fig_y))
    ax.imshow(img, interpolation='nearest')
    if show_ticks:
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticklabels, fontsize=8)
        ax.set_yticklabels(yticklabels, fontsize=8)
        ax.set_xlabel('input')
        ax.set_ylabel('output')

    return fig, ax

def plot_convolution_layer(W):
    u'''plot ordinary M ch input -> N ch output conv layer'''
    return plot_filters(W)


def plot_fully_connected_layer(W):
    u'''plot Linear layer'''
    out_ch, in_ch = W.shape
    elem_s = 0.2
    fig_x = min(8, 0.5 + elem_s*in_ch)
    fig_y = min(8, 0.5 + elem_s*out_ch)

    fig, ax = plt.subplots(1, 1, figsize=(fig_x, fig_y))
    cax = ax.imshow(W, cmap='YlGnBu', interpolation='nearest')
    ax.set_xlabel('input')
    ax.set_ylabel('output')
    fig.colorbar(cax)

    return fig, ax

def plot_first_fully_connected_layer(W, conv_ch, verbose):
    u'''plot the first Linear layer after the covolutional layers'''
    out_ch, in_ch = W.shape
    failed = False
    if conv_ch == 0:
        failed = True
    else:
        conv_yx = in_ch / conv_ch
        if in_ch % conv_ch != 0:
            failed = True

        y = x = int(np.sqrt(conv_yx))
        if not (y == x and y*x == conv_yx):
            failed = True

    if failed:
        if verbose: print 'first fc layer error. fallback.'
        return plot_fully_connected_layer(W)

    if verbose: print 'first fc layer: last conv output: CxHxW=%dx%dx%d -> %d' % (conv_ch, y, x, out_ch)
    W = W.reshape((out_ch*conv_ch, 1, y, x))
    return plot_1ch_input_convolution_layer(W)

def plot_model_layer_images(file_path, model, verbose=True):
    if verbose:
        print 'dict items:'
        for k, v in model.__dict__.items():
            print k
            print v

    figs = []
    class Flag(object):
        def __init__(self, terminate):
            self.terminate = terminate
    flag = Flag(True)
    def onkeypress(ev):
        if ev.key in ['escape', ' ']:
            for name, fig in figs:
                plt.close(fig)
            flag.terminate = ev.key == 'escape'

    if verbose: print 'Convolution layers:'
    convs = sorted([(k, v) for k, v in model.__dict__.items() if isinstance(v, F.connection.convolution_2d.Convolution2D)])
    last_conv_channels = 0
    for name, conv in convs:
        if isinstance(conv.parameters, tuple):
            W = conv.parameters[0]
        else:
            W = conv.parameters
        if not isinstance(W, np.ndarray):
            W = W.get()
        out_ch, in_ch, y, x = W.shape

        if verbose: print 'covolution layer: %s: (%dx%d window, %d -> %d ch)' % (name, y, x, in_ch, out_ch)

        if y*x == 1:
            fig, axs = plot_1x1_convolution_layer(W)
        elif in_ch == 1:
            fig, axs = plot_1ch_input_convolution_layer(W)
        else:
            fig, axs = plot_convolution_layer(W)

        fig.suptitle('%s: (%dx%d window, %d -> %d ch)' % (name, y, x, in_ch, out_ch))
        fig.patch.set_alpha(0.0)
        fig.canvas.mpl_connect('key_press_event', onkeypress)
        figs.append((name, fig))
        last_conv_channels = out_ch

    if verbose: print 'Fully connected layers:'
    fcs = sorted([(k, v) for k, v in model.__dict__.items() if isinstance(v, F.connection.linear.Linear)])
    for name, fc in fcs:
        if isinstance(fc.parameters, tuple):
            W = fc.parameters[0]
        else:
            W = fc.parameters
        if not isinstance(W, np.ndarray):
            W = W.get()
        out_ch, in_ch = W.shape

        if verbose: print 'fully connected layer: %s: %d -> %d' % (name, in_ch, out_ch)

        if name == 'l1':
            fig, axs = plot_first_fully_connected_layer(W, last_conv_channels, verbose)
        else:
            fig, axs = plot_fully_connected_layer(W)

        fig.suptitle('%s: (%d -> %d ch)' % (name, in_ch, out_ch))
        fig.patch.set_alpha(0.0)
        fig.canvas.mpl_connect('key_press_event', onkeypress)
        figs.append((name, fig))

    # save figures
    for name, fig in figs:
        fn = os.path.join(
                os.path.dirname(file_path),
                os.path.splitext(os.path.basename(file_path))[0] + '_img_%s_epoch%04d.png' % (name, model.total_epoch))
        fig.savefig(fn)

    # # tile windows
    # for i, (name, fig) in enumerate(figs):
    #     plt.figure(fig.number)
    #     manager = plt.get_current_fig_manager()
    #     manager.window.move(600*(i%3), 300*(i/3))
    #     manager.window.resize(600, 300)

    # count total parameters
    total_size = 0
    for p in model.parameters:
        if not isinstance(p, np.ndarray):
            p = p.get()
        total_size += p.size
    if verbose: print 'Total model tunable parameters: {} ({:.2f} KB)'.format(total_size, total_size*4.0/1024)

    return figs, flag

def plot_model_layer_images_and_close(model_file, model):
    u'''plot model file to create snapshot files, then close all figures created.'''
    figs, flag = plot_model_layer_images(model_file, model, verbose=False)
    for name, fig in figs:
        plt.close(fig)

def plot_model_layer_histograms(file_path, model, verbose=True):
    if verbose:
        print 'dict items:'
        for k, v in model.__dict__.items():
            print k
            print v

    figs = []
    class Flag(object):
        def __init__(self, terminate):
            self.terminate = terminate
    flag = Flag(True)
    def onkeypress(ev):
        if ev.key in ['escape', ' ']:
            for name, fig in figs:
                plt.close(fig)
            flag.terminate = ev.key == 'escape'

    if verbose: print 'Convolution layers:'
    convs = sorted([(k, v) for k, v in model.__dict__.items() if isinstance(v, F.connection.convolution_2d.Convolution2D)])
    for name, conv in convs:
        W, b = (conv.parameters + (None,))[:2]
        if not isinstance(W, np.ndarray):
            W = W.get()
        if b is not None:
            if not isinstance(b, np.ndarray):
                b = b.get()
            n_axs = 2
        else:
            n_axs = 1
        out_ch, in_ch, y, x = W.shape

        if verbose: print 'covolution layer: %s: (%dx%d window, %d -> %d ch)' % (name, y, x, in_ch, out_ch)

        fig, axs = plt.subplots(n_axs, 1)
        if n_axs == 1:
            axs = [axs]
        axs[0].hist(W.ravel(), bins=100)
        axs[0].set_title('W')
        if n_axs == 2:
            axs[1].hist(b.ravel(), bins=100)
            axs[1].set_title('b')
        fig.suptitle('%s: (%dx%d window, %d -> %d ch)' % (name, y, x, in_ch, out_ch))
        fig.patch.set_alpha(0.0)
        fig.canvas.mpl_connect('key_press_event', onkeypress)
        fig.tight_layout()
        figs.append((name, fig))

    if verbose: print 'Fully connected laers:'
    fcs = sorted([(k, v) for k, v in model.__dict__.items() if isinstance(v, F.connection.linear.Linear)])
    for name, fc in fcs:
        W, b = (fc.parameters + (None,))[:2]
        if not isinstance(W, np.ndarray):
            W = W.get()
        if b is not None:
            if not isinstance(b, np.ndarray):
                b = b.get()
            n_axs = 2
        else:
            n_axs = 1

        out_ch, in_ch = W.shape
        if verbose: print 'fully connected layer: %s: %d -> %d' % (name, in_ch, out_ch)

        fig, axs = plt.subplots(n_axs, 1)
        if n_axs == 1:
            axs = [axs]
        axs[0].hist(W.ravel(), bins=100)
        axs[0].set_title('W')
        if n_axs == 2:
            axs[1].hist(b.ravel(), bins=100)
            axs[1].set_title('b')

        fig.suptitle('%s: (%d -> %d ch)' % (name, in_ch, out_ch))
        fig.patch.set_alpha(0.0)
        fig.canvas.mpl_connect('key_press_event', onkeypress)
        fig.tight_layout()
        figs.append((name, fig))

    # save figures
    for name, fig in figs:
        fn = os.path.join(
                os.path.dirname(file_path),
                os.path.splitext(os.path.basename(file_path))[0] + '_hist_%s_epoch%04d.png' % (name, model.total_epoch))
        fig.savefig(fn)

    # tile windows
    for i, (name, fig) in enumerate(figs):
        plt.figure(fig.number)
        manager = plt.get_current_fig_manager()
        manager.window.move(600*(i%3), 300*(i/3))
        manager.window.resize(600, 300)

    # count total parameters
    total_size = 0
    for p in model.parameters:
        if not isinstance(p, np.ndarray):
            p = p.get()
        total_size += p.size
    if verbose: print 'Total model tunable parameters: {} ({:.2f} KB)'.format(total_size, total_size*4.0/1024)

    return figs, flag

def plot_model_layer_histograms_and_close(model_file, model):
    u'''plot model file to create snapshot files, then close all figures created.'''
    figs, flag = plot_model_layer_histograms(model_file, model, verbose=False)
    for name, fig in figs:
        plt.close(fig)

def plot_model_firstnin(file_path, model, verbose=True):
    figs = []
    class Flag(object):
        def __init__(self, terminate):
            self.terminate = terminate
    flag = Flag(True)
    def onkeypress(ev):
        if ev.key in ['escape', ' ']:
            for name, fig in figs:
                plt.close(fig)
            flag.terminate = ev.key == 'escape'

    if verbose: print 'First NIN layers:'
    convs = sorted([(k, v) for k, v in model.__dict__.items() if isinstance(v, F.connection.convolution_2d.Convolution2D)])
    firstnins = []
    for name, conv in convs:
        W, b = (conv.parameters + (None,))[:2]
        if not isinstance(W, np.ndarray):
            W = W.get()
        out_ch, in_ch, y, x = W.shape

        if (y, x) != (1, 1): continue
        if not name.startswith('conv0_'): continue
        firstnins.append((name, conv))

    xs = np.linspace(0.0, 1.0, 100)
    ys = xs.reshape(1, -1)
    for name, conv in firstnins:
        W, b = (conv.parameters + (None,))[:2]
        if not isinstance(W, np.ndarray):
            W = W.get()
        if b is not None:
            if not isinstance(b, np.ndarray):
                b = b.get()
            n_axs = 2
        else:
            n_axs = 1
        out_ch, in_ch, y, x = W.shape

        if verbose: print 'covolution layer: %s: (%dx%d window, %d -> %d ch)' % (name, y, x, in_ch, out_ch)

        #print 'ys', ys.shape
        #print 'W', W.shape
        #print 'b', b.shape
        def LReLU(x): return np.maximum(0, x) + np.minimum(0, x)*0.2
        def ReLU(x):  return np.maximum(0, x)

        activation = {'conv0_1': LReLU, 'conv0_2': LReLU, 'conv0_3': LReLU}[name]
        ys = activation(W.reshape(out_ch, in_ch).dot(ys) + b.reshape(-1, 1))
        print ys.shape

    print xs.shape
    print ys.shape
    fig = plt.figure()
    ax = None
    nfig = len(ys) + 1
    for i, row in enumerate(ys):
        ax = fig.add_subplot((nfig + 1)/2, 2, i + 1, sharex=ax)
        ax.plot(xs, row)
        ax.set_title('dim{}'.format(i))
        #ax.set_ylim(ys.min(), ys.max())
    i += 1
    ax = fig.add_subplot((nfig + 1)/2, 2, i + 1, sharex=ax)
    ax.plot(xs, ys.max(axis=0))
    ax.plot(xs, ys.sum(axis=0))
    ax.set_title('all at once')
    #ax.set_ylim(ys.min(), ys.max())
    fig.suptitle('->'.join(name for name, conv in firstnins))
    fig.patch.set_alpha(0.0)
    fig.canvas.mpl_connect('key_press_event', onkeypress)
    fig.tight_layout()
    figs.append((name, fig))

    # save figures
    for name, fig in figs:
        fn = os.path.join(
                os.path.dirname(file_path),
                os.path.splitext(os.path.basename(file_path))[0] + '_hist_%s_epoch%04d.png' % (name, model.total_epoch))
        fig.savefig(fn)

    return figs, flag

def plot_model_firstnin_and_close(model_file, model):
    u'''plot model file to create snapshot files, then close all figures created.'''
    figs, flag = plot_model_firstnin(model_file, model, verbose=False)
    for name, fig in figs:
        plt.close(fig)

def run_repeated_model_viewer(model_file, viewer_func):
    while True:
        if os.path.isfile(model_file):
            print 'Loadding module'
            model_dir = os.path.dirname(model_file)
            with open(os.path.join(model_dir, 'model_class.txt'), 'rb') as fi:
                model_module_name, model_class_name = fi.read().split('@')
                model_structure_path = os.path.join(model_dir, 'model.py')
                with open(model_structure_path, 'rb') as fii:
                    module = imp.load_module(model_module_name, fii, model_structure_path, ('.py', 'r', imp.PY_SOURCE))

            print 'Loading model.', model_file
            with open(model_file, 'rb') as fi:
                model = pickle.load(fi)
            
            figs, flag = viewer_func(model_file, model)
            plt.show()

            if flag.terminate:
                break
        else:
            break

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Model Pickle Viewer')
    parser.add_argument('--model_file', '-m',
            help='model pickle file')
    parser.add_argument('--view-mode', '-v', default='image', type=str,
            help='view mode. image|histogram')
    args = parser.parse_args()

    if os.path.isdir(args.model_file):
        model_file = os.path.join(args.model_file, 'model.parameters.pickle')

    if args.view_mode == 'image':
        run_repeated_model_viewer(model_file, plot_model_layer_images)
    elif args.view_mode == 'histogram':
        run_repeated_model_viewer(model_file, plot_model_layer_histograms)
    elif args.view_mode == 'firstnin':
        run_repeated_model_viewer(model_file, plot_model_firstnin)

