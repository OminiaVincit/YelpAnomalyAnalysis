# utilities to construct a chainer model.
import copy
import math
import numpy as np

from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from chainer import function
from chainer.utils import type_check

class ModelBase(FunctionSet):
    def __init__(self, *va, **kwa):
        super(ModelBase, self).__init__(*va, **kwa)
        self.total_epoch = 0
        self.total_minibatch = 0
        self.total_samples = 0
        self.name = 'noname'
        self.max_batchsize = 128 # too large batch will cause GPU halt.

    def feed_samples(self, x_train, y_train):
        # all training samples are fed before training.
        # a model can adapt something using the information.
        pass

    def apply(self, x_data, train, enable_dropout=False):
        raise NotImplementedError()

    def increment_epoch(self):
        self.total_epoch += 1

    def increment_minibatch(self, batchsize):
        self.total_minibatch += 1
        self.total_samples += batchsize

    def start_finetuning(self):
        u'''if you use F.BatchNormalization, finetuning is required after training.
        call bn.start_finetuning() for all batch normalization layers and return True in the overrided function.
        '''
        return False

class ClassificationModelBase(ModelBase):
    def __init__(self, *va, **kwa):
        super(ClassificationModelBase, self).__init__(*va, **kwa)

    def apply(self, x_data, train, enable_dropout=False):
        raise NotImplementedError()

    def forward(self, x_data, y_data, train, enable_dropout=False):
        t = Variable(y_data, volatile=not train)
        y = self.apply(x_data, train, enable_dropout=enable_dropout)
        #print t.data.shape, y.data.shape
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t), y

    def deploy(self, x_data):
        y = self.apply(x_data, train=False, enable_dropout=False)
        return F.softmax(y)

    def confidence(self, x_data, n_try):
        xp = cuda.get_array_module(x_data)
        ps = []
        for i in range(n_try):
            y = self.apply(x_data, False, enable_dropout=True)
            p = F.softmax(y)
            ps.append(p.data[:, 1])
        ps = xp.vstack(ps).T
        return xp.var(ps, axis=1)

def copy_model_to_cpu(model):
    u'''create a copy of a model which have all parameters in CPU side.'''
    cpu_model = copy.deepcopy(model)
    cpu_model.to_cpu()
    return cpu_model

def compute_weight_norm(model, exclude_bias=True):
    u'''compute the L2 norm of the weights of the given model. (not including bias term, indicated by its name 'b')'''
    squared_norm = 0.0
    for _, func in model._get_sorted_funcs():
        for name in func.parameter_names:
            if not exclude_bias or name != 'b':
                p = getattr(func, name)
                with cuda.get_device(p):
                    p = p.ravel()
                    squared_norm += float(p.dot(p))
    return math.sqrt(squared_norm)

# initialize convolution kernel with PCA kernels + noise.
def initial_conv_weight_by_PCA(x_data, n_out_channels, n_in_channels, kw, kh, n_pca_channels=-1, alpha=0.2, n_samples=10000, shuffle=True):
    assert x_data.ndim == 3 or x_data.ndim == 4 and x_data.shape[1] == 1
    if x_data.ndim == 4:
        x_data = x_data.reshape(x_data.shape[0], x_data.shape[2], x_data.shape[3])
    n, h, w = x_data.shape
    if n_pca_channels < 0:
        # half.
        n_pca_channels = min(kw*kh/2, n_out_channels*n_in_channels/2)
    assert 0 <= kh < h and 0 <= kw < w
    assert n_pca_channels <= kw*kh
    assert n_pca_channels <= n_out_channels*n_in_channels

    X = np.ndarray((n_samples, kh, kw))
    for i in range(n_samples):
        z, y, x = np.random.randint(0, n), np.random.randint(0, h - kh), np.random.randint(0, w - kw)
        X[i] = x_data[z, y:y+kh, x:x+kh]

    print 'PCA..',
    import sklearn.decomposition
    pca = sklearn.decomposition.PCA(n_components=n_pca_channels)
    pca.fit(X.reshape(len(X), -1))
    print 'done.'

    W = np.ndarray((n_out_channels*n_in_channels, kh*kw), dtype=np.float32)
    for i in range(n_pca_channels):
        v = pca.components_[i]
        W[i] = (v - v.min())/v.ptp()*(1.0 - alpha) + np.random.randn(v.size)*alpha
    for i in range(n_pca_channels, n_out_channels*n_in_channels):
        W[i] = np.random.randn(kh*kw)
    W = W.reshape((n_out_channels, n_in_channels, kh, kw))
    W *= 1.0/np.sqrt(n_in_channels*kh*kw)
    if shuffle:
        W = W[np.random.permutation(len(W))]
    return W

def initialize_conv_weight_by_PCA(conv, x_data, n_pca_channels=-1, alpha=0.2, n_samples=10000, shuffle=True):
    assert isinstance(conv, F.Convolution2D)
    cout, cin, kh, kw = conv.W.shape
    xp = cuda.get_array_module(conv.W)
    conv.W = xp.asarray(initial_conv_weight_by_PCA(x_data, cout, cin, kh, kw, n_pca_channels, alpha, n_samples, shuffle)).astype(conv.dtype)


def test_initial_conv_weight_by_PCA():
    import matplotlib.pyplot as plt
    import misc_util
    _, x_train, x_test, y_train, y_test = misc_util.retrying_pickle_load(r'..\samples\aao_regr_samples_32x32_3.00_t1_T080_V010_#36000.pickle')
    print x_train.shape
    W = initial_conv_weight_by_PCA(x_train, n_out_channels=5, n_in_channels=3, kw=5, kh=5, n_pca_channels=14, n_samples=10000)

    W = W.reshape(W.shape[0]*W.shape[1], W.shape[2], W.shape[3])
    print W.min(), W.max()
    fig, axs = plt.subplots(5, 3)
    for i, ax in enumerate(axs.ravel()):
        ax.imshow(W[i], cmap='gray', interpolation='nearest')
    plt.show()


if __name__=='__main__':
    test_initial_conv_weight_by_PCA()
