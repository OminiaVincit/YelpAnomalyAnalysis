#!env python
# record/visualize training progress in a database.
import os
import sys
import csv
import time
import shutil
import datetime
import argparse
import numpy as np
import matplotlib
if sys.platform in ['linux', 'linux2']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cPickle as pickle
import cStringIO as StringIO
import tempfile
import lmdb

# serialize object to binary
def serialize(obj):
    buf = StringIO.StringIO()
    pickle.dump(obj, buf)
    return buf.getvalue()

# deserialize binary to object
def deserialize(buf):
    if buf is None: return None
    return pickle.load(StringIO.StringIO(buf))

# shrink-to-fit LMDB. db is closed.
def lmdb_shrink_to_fit(db, path):
    dir_path = os.path.dirname(path)
    temp_path = tempfile.mkdtemp(suffix='_lmdb.db', dir=dir_path)
    db.copy(temp_path, compact=True)
    db.close()
    if os.path.isdir(path):
        shutil.rmtree(path)
    shutil.move(temp_path, path)
    print 'log DB shrinked using temporary directory {}.'.format(temp_path)

def retry(t, count):
    def _decorator(f):
        def _f(*av, **ak):
            for i in range(count):
                try:
                    return f(*av, **ak)
                except Exception, e:
                    print 'Error:', e, e.message
                    print 'Sleep and retry..'
                    time.sleep(t)
        return _f
    return _decorator

class LogType(object):
    epoch     = 'epoch'
    minibatch = 'minibatch'
    other     = 'other'

class DataType(object):
    train = 'train'
    test  = 'test'

class MeasureType(object):
    training   = 'training'
    validation = 'validation'

def create_norm(grad=None, weight=None, grad_ratio=None): return locals()

def create_cost(loss=None, accuracy=None, mse=None): return locals()

def create_cost_distribution(costs):
    distrib = dict(loss=[], accuracy=[], mse=[])
    for c in costs:
        for k, v in c.items():
            if v is not None:
                distrib[k].append(v)
    for k, v in distrib.items():
        if len(v) == 0:
            del distrib[k]
    return _dict2obj(distrib)

class Obj(object):
    def __repr__(self):
        return '<Obj {}>'.format(', '.join('{}={}'.format(k, v) for k, v in self.__dict__.items()))

def _dict2obj(d):
    o = Obj()
    for k, v in d.items():
        if k != 'o':
            setattr(o, k, v)
    return o

def log_entry(
        total_epoch=None,
        inepoch_minibatch=None, n_minibatch_in_epoch=None, total_sample=None,
        log_type=None,
        data_type=None,
        measure_type=None,
        cost=None,
        norm=None,
        cost_distribution=None,
        created_at=None):
    if created_at is None:
        created_at = datetime.datetime.now()
    return _dict2obj(locals())

def test_log_entry():
    costs = []
    for i in range(5):
        costs.append(create_cost(loss=0.3*i, accuracy=0.8))
    distrib = create_cost_distribution(costs)

    e = log_entry(
            total_epoch=1,
            total_minibatch=10+np.random.randint(0, 100),
            log_type=LogType.epoch,
            data_type=DataType.test,
            measure_type=MeasureType.validation,
            cost=create_cost(loss=np.random.randn(), accuracy=0.5+np.random.randn()),
            cost_distribution=distrib
            )
    print e
    return e

class TrainingLog(object):
    u'''training log DB wrapper'''
    def __init__(self, file_path=None):
        self.s = None
        self.file_path = None
        if file_path is not None:
            self.open(file_path)

    def open(self, file_path, type='c', task_type=None):
        self.file_path = file_path
        self.s = lmdb.open(file_path, map_size=128*1024*1024, readonly=(type == 'r'))
        if type != 'r':
            with self.s.begin(write=True) as txn:
                self.primary_key = deserialize(txn.get('primary_key', serialize(0)))
                txn.put('primary_key', serialize(self.primary_key), dupdata=False)
                task_type = deserialize(txn.get('task_type', serialize(task_type)))
                txn.put('task_type', serialize(task_type))
        else:
            with self.s.begin() as txn:
                self.primary_key = deserialize(txn.get('primary_key'))

    def __del__(self):
        self.close()

    @retry(0.5, 5)
    def close(self):
        if self.s:
            if not self.s.flags()['readonly']:
                lmdb_shrink_to_fit(self.s, self.file_path)
            else:
                self.s.close()
            self.s = None

    @retry(0.5, 5)
    def flush(self):
        if self.s:
            self.s.sync()

    @retry(0.5, 5)
    def add_log(self, item):
        if not self.s: raise RuntimeError('DB not opened.')
        with self.s.begin(write=True) as txn:
            txn.put('{:8}'.format(self.primary_key), serialize(item))
            txn.put('primary_key', serialize(self.primary_key + 1), dupdata=False)
        self.primary_key += 1

    @retry(0.5, 5)
    def add_logs(self, items):
        if not self.s: raise RuntimeError('DB not opened.')
        with self.s.begin(write=True) as txn:
            k = self.primary_key
            for item in items:
                txn.put('{:8}'.format(k), serialize(item))
                txn.put('primary_key', serialize(k + 1), dupdata=False)
                k += 1
        self.primary_key = k

    @retry(0.5, 5)
    def store_value(self, key, val):
        if not self.s: raise RuntimeError('DB not opened.')
        with self.s.begin(write=True) as txn:
            txn.put(key, serialize(val))

    @retry(0.5, 5)
    def load_value(self, key):
        if not self.s: raise RuntimeError('DB not opened.')
        with self.s.begin() as txn:
            return deserialize(txn.get(key, None))

    @retry(0.5, 5)
    def read_all(self):
        items = []
        with self.s.begin() as txn:
            primary_key = deserialize(txn.get('primary_key'))
            for i in range(primary_key):
                try:
                    items.append(deserialize(txn.get('{:8}'.format(i))))
                except KeyError:
                    print 'key {} not found'.format(i)
        return np.array(items)

    @retry(0.5, 5)
    def is_regression(self):
        with self.s.begin() as txn:
            return deserialize(txn.get('task_type')) == 'regression'

def test_logging():
    log = TrainingLog()
    log.open('testlog.db', 'regression')
    log.add_log(test_log_entry())
    time.sleep(0.2)
    log.add_log(test_log_entry())
    log.flush()
    del log
    log = TrainingLog()
    log.open('testlog.db', 'regression')
    time.sleep(0.2)
    log.add_log(test_log_entry())
    del log
    log = TrainingLog()
    log.open('testlog.db')
    print log.read_all()
    print log.s


class LogPlot(object):
    def __init__(self):
        self.setup_figure()

    def setup_figure(self):
        self.fig = plt.figure(figsize=(12, 12))
        self.fig.patch.set_alpha(0)
        self.ax_perf_l = self.fig.add_subplot(2, 1, 1)
        self.ax_perf_r = self.ax_perf_l.twinx()
        self.ax_norm_l = self.fig.add_subplot(2, 1, 2, sharex=self.ax_perf_l)
        self.ax_norm_r = self.ax_norm_l.twinx()
        self.fig.subplots_adjust(top=0.90, bottom=0.05, left=0.08, right=0.92, hspace=0.4)
        self.terminate = True
        self.timer = None
        def onkeypress(ev):
            if ev.key == 'escape':
                plt.close(self.fig)
            if ev.key == ' ':
                self.refresh()
            if ev.key == 'a': # auto refresh
                if self.timer:
                    self.timer.stop()
                else:
                    self.timer = self.fig.canvas.new_timer(interval=10000)
                    self.timer.add_callback(self.refresh)
                    self.timer.start()
        self.fig.canvas.mpl_connect('key_press_event', onkeypress)
        self.do_resize = True
        self.text = None

    def refresh(self):
        self.ax_perf_l.cla()
        self.ax_perf_r.cla()
        self.ax_norm_l.cla()
        self.ax_norm_r.cla()
        self.plot_log(self.file_path)
        plt.draw()

    def _load(self, file_path):
        # load
        print 'Loading log..',
        log = TrainingLog()
        log.open(file_path, type='r')
        items = log.read_all()
        print '.done'
        return log, items

    def plot_log(self, file_path):
        self.file_path = file_path
        log, items = self._load(self.file_path)

        self.update_log(log, items)

        # save plot
        if len(items) > 0:
            fig_path = os.path.join(os.path.dirname(self.file_path), 'training_log.png')
            self.fig.savefig(fig_path)

    def update_log(self, log, items):
        loss_ls, accuracy_ls = '-', '-'
        minibatch_ls = '-'
        minibatch_marker = ''
        epoch_marker = 'o'
        train_color, test_color = 'blue', 'red'
        train_accuracy_color, test_accuracy_color = 'cyan', 'magenta'
        gnorm_color, wnorm_color, ratio_color = 'blue', 'green', 'brown'
        plot_xshrink = 0.7
        plot_yshrink = 0.8
        lines_upper = []
        lines_lower = []

        def plot_loss(log_type):
            # loss/mse
            if log_type == LogType.minibatch:
                marker = minibatch_marker
                alpha = 0.2
                label = 'minibatch'
                ls = minibatch_ls
            else:
                marker = epoch_marker
                alpha = 0.5
                label = 'epoch'
                ls = loss_ls

            err_key = None
            train_loss, test_loss = [], []
            for item in items:
                if item.cost is None: continue
                if item.log_type != log_type: continue
                if err_key is None:
                    if item.cost['loss'] is not None:
                        err_key = 'loss'
                    elif item.cost['mse'] is not None: 
                        err_key = 'mse'
                if item.cost.has_key(err_key) and item.cost[err_key] is not None:
                    x = item.total_epoch
                    if item.inepoch_minibatch is not None:
                        x += item.inepoch_minibatch/float(item.n_minibatch_in_epoch) - 1.0
                    if item.measure_type == MeasureType.validation:
                        if item.data_type == DataType.train:
                            train_loss.append((x, item.cost[err_key]))
                        if item.data_type == DataType.test:
                            test_loss.append((x, item.cost[err_key]))
            train_loss, test_loss = map(np.array, (train_loss, test_loss))
            tr_last, test_last = None, None
            
            if len(train_loss) > 0:
                lines_upper.extend(self.ax_perf_l.plot(train_loss[:, 0], train_loss[:, 1], ls=ls, color=train_color, marker=marker, alpha=alpha, label='train {} loss'.format(label)))
                tr_last = (train_loss[-1, 0], train_loss[-1, 1])
            if len(test_loss) > 0:
                lines_upper.extend(self.ax_perf_l.plot(test_loss[:, 0],  test_loss[:, 1],  ls=ls, color=test_color,  marker=marker, alpha=alpha, label='test {} loss'.format(label)))
                test_last = (test_loss[-1, 0], test_loss[-1, 1])

            return err_key, tr_last, test_last

        err_key, tr_loss_last_minibatch, test_loss_last_minibatch = plot_loss(LogType.minibatch)
        err_key, tr_loss_last_epoch,     test_loss_last_epoch = plot_loss(LogType.epoch)

        self.ax_perf_l.set_xlabel('epoch')
        self.ax_perf_l.set_ylim(0, self.ax_perf_l.get_ylim()[1])
        self.ax_perf_l.set_ylabel('loss' if err_key == 'loss' else 'mean squared error')

        # accuracy
        def plot_accuracy(log_type):
            # accuracy/mse
            if log_type == LogType.minibatch:
                marker = minibatch_marker
                alpha = 0.2
                label = 'minibatch'
            else:
                marker = 'o'
                marker = epoch_marker
                alpha = 0.5
                label = 'epoch'

            train_accuracy, test_accuracy = [], []
            for item in items:
                if item.cost is None: continue
                if item.log_type != log_type: continue
                if item.cost['accuracy'] is not None:
                    x = item.total_epoch
                    if item.inepoch_minibatch is not None:
                        x += item.inepoch_minibatch/float(item.n_minibatch_in_epoch) - 1.0
                    if item.measure_type == MeasureType.validation:
                        if item.data_type == DataType.train:
                            train_accuracy.append((x, item.cost['accuracy']))
                        if item.data_type == DataType.test:
                            test_accuracy.append((x, item.cost['accuracy']))
            train_accuracy, test_accuracy = map(np.array, (train_accuracy, test_accuracy))
            tr_last, test_last = None, None

            if len(train_accuracy) > 0:
                lines_upper.extend(self.ax_perf_r.plot(train_accuracy[:, 0], train_accuracy[:, 1], ls=accuracy_ls, color=train_accuracy_color, marker=marker, alpha=alpha, label='train {} acc.'.format(label)))
                tr_last = (train_accuracy[-1, 0], train_accuracy[-1, 1])
            if len(test_accuracy) > 0:
                lines_upper.extend(self.ax_perf_r.plot(test_accuracy[:, 0],  test_accuracy[:, 1],  ls=accuracy_ls, color=test_accuracy_color,  marker=marker, alpha=alpha, label='test {} acc.'.format(label)))
                test_last = (test_accuracy[-1, 0], test_accuracy[-1, 1])

            return tr_last, test_last

        tr_acc_last_minibatch, test_acc_last_minibatch = plot_accuracy(LogType.minibatch)
        tr_acc_last_epoch, test_acc_last_epoch = plot_accuracy(LogType.epoch)

        self.ax_perf_r.set_ylim(0, self.ax_perf_r.get_ylim()[1])
        self.ax_perf_r.set_ylabel('accuracy')

        # -- lower axes ----------------------------------------------------------------------------------

        '''
        # gradient norm
        def plot_gnorm():
            # gnorm
            alpha = 0.5
            marker = ''
            ls = '-'

            train_gnorm, test_gnorm = [], []
            for item in items:
                if item.norm is None: continue
                if item.log_type != LogType.minibatch: continue
                if item.norm.has_key('grad') and item.norm['grad'] is not None:
                    x = item.total_epoch
                    if item.inepoch_minibatch is not None:
                        x += item.inepoch_minibatch/float(item.n_minibatch_in_epoch) - 1.0
                    if item.measure_type == MeasureType.training:
                        if item.data_type == DataType.train:
                            train_gnorm.append((x, item.norm['grad']))
                        if item.data_type == DataType.test:
                            test_gnorm.append((x, item.norm['grad']))
            train_gnorm, test_gnorm = map(np.array, (train_gnorm, test_gnorm))

            if len(train_gnorm) > 0:
                lines_lower.extend(self.ax_norm_l.plot(train_gnorm[:, 0], train_gnorm[:, 1], ls=ls, color=gnorm_color, marker=marker, alpha=alpha, label='train grad. norm'))

        plot_gnorm()

        self.ax_norm_l.set_xlabel('epoch')
        self.ax_norm_l.set_ylim(0, self.ax_norm_l.get_ylim()[1])
        self.ax_norm_l.set_ylabel('gradient norm')
        '''

        # weight norm
        def plot_wnorm():
            # wnorm
            alpha = 0.5
            marker = ''
            ls = '-'

            train_wnorm, test_wnorm = [], []
            for item in items:
                if item.norm is None: continue
                if item.log_type != LogType.minibatch: continue
                if item.norm.has_key('weight') and item.norm['weight'] is not None:
                    x = item.total_epoch
                    if item.inepoch_minibatch is not None:
                        x += item.inepoch_minibatch/float(item.n_minibatch_in_epoch) - 1.0
                    if item.measure_type == MeasureType.training:
                        if item.data_type == DataType.train:
                            train_wnorm.append((x, item.norm['weight']))
                        if item.data_type == DataType.test:
                            test_wnorm.append((x, item.norm['weight']))
            train_wnorm, test_wnorm = map(np.array, (train_wnorm, test_wnorm))

            if len(train_wnorm) > 0:
                lines_lower.extend(self.ax_norm_r.plot(train_wnorm[:, 0], train_wnorm[:, 1], ls=ls, color=wnorm_color, marker=marker, alpha=alpha, label='weight'))

        plot_wnorm()
        self.ax_norm_r.set_ylim(0, self.ax_norm_r.get_ylim()[1])
        self.ax_norm_r.set_ylabel('weight norm')

        # grad/weight ratio
        def plot_ratio():
            alpha = 0.5
            marker = ''
            ls = '-'

            ratio = []
            for item in items:
                if item.norm is None: continue
                if item.log_type != LogType.minibatch: continue
                if item.norm.has_key('grad_ratio') and item.norm['grad_ratio'] is not None:
                    x = item.total_epoch
                    if item.inepoch_minibatch is not None:
                        x += item.inepoch_minibatch/float(item.n_minibatch_in_epoch) - 1.0
                    if item.measure_type == MeasureType.training:
                        if item.data_type == DataType.train:
                            ratio.append((x, item.norm['grad_ratio']))
            ratio = np.array(ratio)

            if len(ratio) > 0:
                lines_lower.extend(self.ax_norm_l.plot(ratio[:, 0], ratio[:, 1], ls=ls, color=ratio_color, marker=marker, alpha=alpha, label='grad/weight'))

        plot_ratio()
        self.ax_norm_l.set_xlabel('epoch')
        self.ax_norm_l.set_ylim(0, self.ax_norm_l.get_ylim()[1])
        self.ax_norm_l.set_ylabel('gradient norm/weight norm')

        # -- common ----------------------------------------------------------------------------------
        # legend
        box = self.ax_perf_l.get_position()
        if self.do_resize:
            self.text_xy = (box.x0 + box.width*plot_xshrink + 0.05, box.y0 + box.height - 0.05)
            self.ax_perf_l.set_position([box.x0, box.y0 + box.height*(1 - plot_yshrink), box.width*plot_xshrink, box.height*plot_yshrink])
            box = self.ax_perf_r.get_position()
            self.ax_perf_r.set_position([box.x0, box.y0 + box.height*(1 - plot_yshrink), box.width*plot_xshrink, box.height*plot_yshrink])

            box = self.ax_norm_l.get_position()
            self.ax_norm_l.set_position([box.x0, box.y0 + box.height*(1 - plot_yshrink), box.width*plot_xshrink, box.height*plot_yshrink])
            box = self.ax_norm_r.get_position()
            self.ax_norm_r.set_position([box.x0, box.y0 + box.height*(1 - plot_yshrink), box.width*plot_xshrink, box.height*plot_yshrink])
            self.do_resize = False
        self.ax_perf_l.legend(lines_upper, [l.get_label() for l in lines_upper], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        self.ax_norm_l.legend(lines_lower, [l.get_label() for l in lines_lower], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

        # cursor
        self.ax_perf_r.format_coord = self.make_format(self.ax_perf_r, self.ax_perf_l)
        self.ax_norm_r.format_coord = self.make_format(self.ax_norm_r, self.ax_norm_l)

        if True:
            # text area

            text_lines = []
            if tr_acc_last_epoch and  tr_loss_last_epoch:
                text_lines.append('epoch={:.0f}: train_acc={:.4f}, train_loss={:.4f}'.format(tr_acc_last_epoch[0], tr_acc_last_epoch[1], tr_loss_last_epoch[1]))

            if test_acc_last_epoch and  test_loss_last_epoch:
                text_lines.append('epoch={:.0f}: test_acc={:.4f}, test_loss={:.4f}'.format(test_acc_last_epoch[0], test_acc_last_epoch[1], test_loss_last_epoch[1]))
                
            task = log.load_value('task')
            if task and getattr(task, 'settings', None):
                text_lines.append('task:{}'.format(task.__class__.__name__))
                # if task.settings.has_key('tolerance_px'):
                #     text_lines.append('regr.tolerance={}px'.format(task.settings['tolerance_px']))
                if task.settings.has_key('weight_decay'):
                    text_lines.append('weight_decay={}'.format(task.settings['weight_decay']))

            # args option
            args = log.load_value('args')
            if args and getattr(args, 'optimizer', None):
                opt = args.optimizer
                if opt == 'Adam' and getattr(args, 'alpha', None):
                        text_lines.append('Adam, alpha={}'.format(args.alpha))
                elif opt == 'MomentumSGD':
                    if getattr(args, 'lr', None):
                        text_lines.append('MomentumSGD, learning_rate={}'.format(args.lr))

                    if getattr(args, 'momentum', None) and getattr(args, 'lr_decay', None):
                        text_lines.append('momentum={}, lr_decay_rate={}'.format(args.momentum, args.lr_decay))

            if args and getattr(args, 'downsample_train', None) and getattr(args, 'downsample_test', None):
                text_lines.append('downsample_rate test={}, train={}'.format(args.downsample_test, args.downsample_train))

            test_samples = log.load_value('test_samples')
            train_samples = log.load_value('train_samples')

            if test_samples and train_samples:
                text_lines.append('number of samples: test={}, train={}'.format(test_samples, train_samples))

            if args and getattr(args, 'train_level_list', None):
                text_lines.append('train_level_list={}'.format(args.train_level_list))

            if args and getattr(args, 'train_pos_samples', None):
                text_lines.append('train_pos_samples={}'.format(args.train_pos_samples))

            if args and getattr(args, 'train_neg_samples', None):
                text_lines.append('train_neg_samples={}'.format(args.train_neg_samples))

            if args and getattr(args, 'test_level_list', None):
                text_lines.append('test_level_list={}'.format(args.test_level_list))

            if args and getattr(args, 'test_pos_samples', None):
                text_lines.append('test_pos_samples={}'.format(args.test_pos_samples))

            if args and getattr(args, 'test_neg_samples', None):
                text_lines.append('test_neg_samples={}'.format(args.test_neg_samples))

            if args and getattr(args, 'bone_interval', None) and getattr(args, 'num_channels', None):
                text_lines.append('num_channels = {}, bone_interval = {}'.format(args.num_channels, args.bone_interval))

            model_desc = log.load_value('model_desc')
            if model_desc:
                text_lines.append('model_desc={}'.format(model_desc))

            # list of test and train patients
            test_list = log.load_value('test_list')
            train_list = log.load_value('train_list')
            if test_list and train_list:
                text_lines.append('Test patient list')
                for tmp in test_list:
                    text_lines.append(tmp)
                
                text_lines.append('Train patient list')
                for tmp in train_list:
                    text_lines.append(tmp)

            try: self.text.remove()
            except AttributeError: pass

            self.text = self.fig.text(self.text_xy[0], self.text_xy[1], '\n'.join(text_lines),
                    horizontalalignment='left',
                    verticalalignment='top',
                    multialignment='left',
                    size='x-small',
                    )

        # figure
        self.fig.suptitle('{path}'.format(
            path=os.path.relpath(self.file_path, os.path.join(os.path.dirname(self.file_path), '..')),
            ))

    def make_format(self, current, other):
        u'''from http://stackoverflow.com/questions/21583965/matplotlib-cursor-value-with-two-axes'''
        # current and other are axes
        def format_coord(x, y):
            # x, y are data coordinates
            # convert to display coords
            display_coord = current.transData.transform((x,y))
            inv = other.transData.inverted()
            # convert back to data coords with respect to ax
            ax_coord = inv.transform(display_coord)
            coords = [ax_coord, (x, y)]
            return ('Left({0}): {2:<40}    Right({1}): {3:<}'
                    .format(other.get_ylabel(), current.get_ylabel(), *['({:.3f}, {:.3f})'.format(x, y) for x,y in coords]))
        return format_coord


def main_log_viewer():
    # arguments
    parser = argparse.ArgumentParser(description='Log Viewer')
    parser.add_argument('--logfile', type=str,
                        help='log database file')
    args = parser.parse_args()
    file_path = args.logfile
    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, 'training_log.db')

    # figure
    plotter = LogPlot()

    while True:
        # update figure
        plotter.plot_log(file_path)
        plt.show()

        if plotter.terminate:
            break

if __name__=='__main__':
    #test_log_entry()
    #test_logging()
    main_log_viewer()


