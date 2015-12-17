#!env python
import os
import cPickle as pickle
import time

import numpy as np
from chainer import cuda

def retry(t, count, do_raise=False):
    def _decorator(f):
        def _f(*av, **ak):
            for i in range(count):
                try:
                    return f(*av, **ak)
                except Exception, e:
                    print 'Error:', e.message
                    print 'wait..'
                    time.sleep(t)
            if do_raise:
                raise RuntimeError('{} retrying of {} finally failed.'.format(count, f.__name__))
            return False
        return _f
    return _decorator

@retry(2.0, 5)
def retrying_pickle_load(file_path):
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'rb') as fi:
        return pickle.load(fi)

@retry(2.0, 5)
def retrying_pickle_dump(obj, file_path):
    with open(file_path, 'wb') as fo:
        pickle.dump(obj, fo, protocol=pickle.HIGHEST_PROTOCOL)
    return True


def setup_gpu(gpu_no):
    if gpu_no >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu_no).use()
        print '***GPU ENABLED***'
    else:
        print '***GPU DISABLED***'
    xp = cuda.cupy if gpu_no >= 0 else np
    return xp
