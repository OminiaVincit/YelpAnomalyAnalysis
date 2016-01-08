import os
import numpy as np
import time

from sklearn import svm
from scipy.stats import spearmanr

DATA_DIR = r'/home/zoro/work/Dataset/Features'
text_ncol = 13

def load_data(site, index, num_features, feature_tag, data_dir = DATA_DIR):
  """
  Loading train and test data
    feature_tag = 0 --> using only textual feature
    feature_tag = 1 --> using only topics feature
    feature_tag = 2 --> using both textual and topics feature
  """

  train_file_name = '%s_%d_train_data_split_%d.npy' % (site, num_features, index)
  test_file_name = '%s_%d_test_data_split_%d.npy' % (site, num_features, index)
  train_data = np.load(os.path.join(data_dir, train_file_name))
  test_data = np.load(os.path.join(data_dir, test_file_name))

  (N_train, F_train) = train_data.shape
  (N_test, F_test) = test_data.shape

  if feature_tag == 0:
    feature_col = range(0, text_ncol)
  elif feature_tag == 1:
    feature_col = range(text_ncol, (text_ncol + num_features))
  else:
    feature_col = range(0, (text_ncol + num_features))

  ncol = len(feature_col)
  N = N_train
  x_train = train_data[0:N, feature_col].reshape(N, ncol)
  y_train = train_data[0:N, F_train-2].reshape(N, )

  x_test = test_data[:, feature_col].reshape(N_test, ncol)
  y_test = test_data[:, F_test-2].reshape(N_test, )

  return x_train, y_train, x_test, y_test

def svr(x_train, y_train, x_test, y_test):
  """
  Support vector regression
  """
  start = time.time()
  reg = svm.SVR(kernel='rbf', C=1).fit(x_train, y_train)
  mse, rmse, mae, rho, pval = score(reg, x_test, y_test)
  print mse, rmse, mae, rho, pval, time.time() - start, 'seconds'

def score(reg, x_test, y_test):
  """
  Some of metric score for regression evaluation
  """
  #orig_err = reg.score(x_test, y_test)
  y_predict = reg.predict(x_test)
  diff = y_test - y_predict
  mse = np.mean(diff*diff)
  #rmse = np.sqrt(mse)
  #mae = np.mean(abs(diff))

  # Compute spearmanr correlation score
  #y_predict = np.sort(y_predict)
  #y_test = np.sort(y_test)
  #rho, pval = spearmanr(y_predict, y_test)
  return mse

def grid_search_cv(x_train, y_train, x_test, y_test):
  start = time.time()
  from sklearn.grid_search import GridSearchCV
  tuned_parameters = [
    {'kernel': ['rbf'], 'gamma': [10**i for i in range(-4,3)], 'C': [10**i for i in range(-3, 4)], 'epsilon':[float(i)/10.0 for i in range(0, 11)] }]
    #{'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'epsilon':[float(i)/10.0 for i in range(0, 11)] } ]
  gscv = GridSearchCV(svm.SVR(), tuned_parameters, cv=5, scoring='mean_squared_error', n_jobs=8)
  gscv.fit(x_train, y_train)

  # Worse and best score
  params_min,_,_ = gscv.grid_scores_[np.argmin([x[1] for x in gscv.grid_scores_])]
  params_max,_,_ = gscv.grid_scores_[np.argmax([x[1] for x in gscv.grid_scores_])]
  reg_min = svm.SVR(kernel=params_min['kernel'], C=params_min['C'], gamma=params_min['gamma'])
  reg_max = gscv.best_estimator_
  #params_max = reg_max.get_params()
  
  # Refit using all training data
  reg_min.fit(x_train, y_train)
  reg_max.fit(x_train, y_train)
  print 'reg_min ', score(reg_min, x_test, y_test), params_min
  print 'reg_max ', score(reg_max, x_test, y_test), params_max
  #params_max['kernel'], params_max['gamma'], params_max['C'], params_max['epsilon']
  print 'grid_search_cv in', time.time() - start

if __name__ == '__main__':
  x_train, y_train, x_test, y_test = load_data('yelp', 1, 64, 0)
  print x_train.shape, x_test.shape, y_train.shape, y_test.shape
  grid_search_cv(x_train, y_train, x_test, y_test)