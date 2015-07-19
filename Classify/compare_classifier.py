'''Compare classifier for different features design'''
import numpy as np
import os
import logging 
import argparse

from utils import Dataset, make_data, create_result_dir
from sklearn.preprocessing import StandardScaler

# For split training and test data
from sklearn.cross_validation import train_test_split


# For evaluation
from sklearn.metrics import roc_auc_score

#import matplotlib.pyplot as plt

BASE_DIR = r'../../Dataset/Features'
RESULT_DIR = r'../Results'

if __name__ == '__main__':
  u'Compare classifier with different features design'
  from sklearn.naive_bayes import GaussianNB, MultinomialNB
  from sklearn.lda import LDA
  from sklearn.qda import QDA
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
  from sklearn.neighbors.nearest_centroid import NearestCentroid
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.linear_model import SGDClassifier, LogisticRegression

  from sklearn import svm, metrics
  from sklearn.svm import LinearSVC, SVC
  from sklearn.multiclass import OneVsRestClassifier
  from sklearn.multiclass import OneVsOneClassifier

  parser = argparse.ArgumentParser()
  parser.add_argument('--site', type=str, default='all',
                      choices=['yelp', 'tripadvisor', 'movies', 'all'])
  parser.add_argument('--data_dir', type=str, default=BASE_DIR)
  parser.add_argument('--result_dir', type=str, default=RESULT_DIR)
  parser.add_argument('--features', type=str, default='all',
                      choices=['text_only', 'topics_only', 'text_topics', 'all'])
  parser.add_argument('--num_test', type=int, default=10)
  parser.add_argument('--classifier', type=str, default='all',
                      choices= [
                        'all',
                        'GaussianNB',
                        'MultinomialNB',
                        'LDA',
                        'QDA',
                        'DecisionTreeClassifier',
                        'RandomForestClassifier',
                        'ExtraTreesClassifier',
                        'AdaBoostClassifier',
                        'NearestCentroid',
                        'KNeighborsClassifier',
                        'SGDClassifier',
                        'LogisticRegression',
                        #'SVC',
                        #'SVC_linear',
                        'OneVsRestClassifier_LinearSVC',
                        'OneVsOneClassifier_LinearSVC'
                      ])

  
  args = parser.parse_args()

  print (args)

  sitelist = []
  designs = []
  classifiers = []
  names = []
  # List of classifiers
  CLASSIFIERS = [
    GaussianNB(),
    MultinomialNB(),
    LDA(),
    QDA(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=10),
    ExtraTreesClassifier(n_estimators=10),
    AdaBoostClassifier(n_estimators= 10, learning_rate = 1.0),
    NearestCentroid(),
    KNeighborsClassifier(),
    SGDClassifier(loss='hinge'),
    LogisticRegression(),
    #SVC(gamma = 2, C=1), # VERY SLOW
    #SVC(kernel='linear', C=0.025), # VERY SLOW
    OneVsRestClassifier( LinearSVC() ),
    OneVsOneClassifier( LinearSVC() ),
  ]

  NAMES = [
    'GaussianNB',
    'MultinomialNB',
    'LDA',
    'QDA',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'ExtraTreesClassifier',
    'AdaBoostClassifier',
    'NearestCentroid',
    'KNeighborsClassifier',
    'SGDClassifier',
    'LogisticRegression',
    #'SVC',
    #'SVC_linear',
    'OneVsRestClassifier_LinearSVC',
    'OneVsOneClassifier_LinearSVC'
  ]

  # Create result dir
  log_fn, result_dir = create_result_dir(args)

  # Set site and features, classifiers
  if args.site == 'all':
    sitelist = ['yelp', 'tripadvisor', 'movies']
  else:
    sitelist.append(args.site)   

  if args.features == 'all':
    designs = ['text_only', 'topics_only', 'text_topics']
  else:
    designs.append(args.features)

  if args.classifier == 'all':
    classifiers = CLASSIFIERS
    names = NAMES
  else:
    clf = CLASSIFIERS[NAMES.index()]
    classifiers.append(clf)
    names.append(args.classifiers)

  num_test = 10
  if args.num_test:
    num_test = args.num_test

  for site in sitelist:
    filename = os.path.join(args.data_dir, site + '_all_features.txt')
    for design in designs:
      # Choosing active column
      usecols = None
      if design == 'text_only':
        flist = range(15)
        flist.append(65)
        usecols = tuple(flist)
      elif design == 'topics_only':
        flist = range(15, 66)
        usecols = tuple(flist)

      # Make training and testing data
      ds = make_data(filename, usecols)
      TEST_SIZE = 0.2

      for name, clf in zip(names, classifiers):
        acc_list = []
        for i in range(num_test):
          X_train, X_val, y_train, y_val = \
            train_test_split(ds.features, ds.target, \
                             test_size=TEST_SIZE)

          clf.fit(X_train, y_train)
          accuracy = clf.score(X_val, y_val)
          acc_list.append(accuracy)
          msg = 'Iter:\t site={}, design={}, classifier={}, iter={}, accuracy={}'\
            .format(site, design, name, i, accuracy)
          logging.info(msg)
          print ('%s' % msg)

        # Out to log - stat
        mean_s = np.mean(acc_list)
        med_s = np.median(acc_list)

        msg = 'Stat:\t site={}, design={}, classifier={}, num_test={}, mean={}, median={}'\
          .format(site, design, name, num_test, mean_s, med_s)
        logging.info(msg)
        print ('%s' % msg)  
