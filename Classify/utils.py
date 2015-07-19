'''Data utils'''
import logging 
import time, os
import numpy as np

class Dataset( object ):
  u'''Hold dataset object'''

  def __init__( self ):
    u'''Init Dataset'''
    self.features = None
    self.target = None

class Evaluation( object ):
  u'''Hold evaluation object'''

  def __init__( self ):
    u'''Init evaluation class'''
    self.X_train = None
    self.y_train = None
    self.X_val = None
    self.y_val = None

def make_data(filename, usecols=None):
  u'Make data set from features space'

  ds = Dataset()
  data = np.loadtxt(filename, dtype='float', delimiter=' ', usecols=usecols)
  (nr, nc) = data.shape
  ds.target = np.zeros(nr).astype(np.uint8)
  for i in range(nr):
    # Check label
    # quality [0.8, 1] -> label 1, [0.6, 0.8) -> label 2
    # [0.4, 0.6] -> label 3, [0.2, 0.4) -> label 1
    # [0, 0.2) -> label 0
    ds.target[i] = int( data[i, nc-1] / 0.2 )
    if ds.target[i] == 5:
      ds.target[i] = 4
  ds.features = data[:, 0:(nc-1)]
  return ds

def create_result_dir(args):
  u'''Crate result directory for experiments'''
  if args.result_dir:
    result_dir = args.result_dir
  else:
    result_dir = 'results'
  result_dir = os.path.join(result_dir, args.site)
  result_dir += '_' + os.path.basename(args.features)
  result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
  result_dir += str(time.time()).replace('.', '')
  if not os.path.exists(result_dir):
      os.makedirs(result_dir)
  log_fn = '%s/log.txt' % result_dir
  logging.basicConfig(
      format='%(asctime)s [%(levelname)s] %(message)s',
      filename=log_fn, level=logging.DEBUG)
  logging.info(args)
  return log_fn, result_dir

def get_all_class_names():
  """
  Get all class name
  """
  all_cl = ['Terrible', 'Bad', 'Normal',  'Good', 'Excellent']
  return all_cl

def get_class_names_for_class_indices(indices):
  """
  Convert from numeric value to label name
  """
  all_cl = get_all_class_names()
  class_names = []
  for i in range(len(indices)):
    assert(indices[i] >= 0 and indices[i] < len(all_cl))
    class_names.append(all_cl[indices[i]])
  return class_names

