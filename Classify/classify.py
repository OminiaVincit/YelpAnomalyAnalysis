'''Clasify review into 5 class'''
import numpy as np
import os

from utils import Dataset
from sklearn.preprocessing import StandardScaler

# For split training and test data
from sklearn.cross_validation import train_test_split

# For SVM classifier
from sklearn import svm, metrics
from sklearn.svm import LinearSVC

# For evaluation
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

BASE_DIR = r'../../Dataset/Features'

def make_data(filename, usecols=None):
  u'''Make data set from features space'''

  ds = Dataset()
  data = np.loadtxt(filename, dtype='float', delimiter=' ', usecols=usecols)
  (nr, nc) = data.shape
  ds.target = np.zeros(nr)
  for i in range(nr):
    # Check label
    # quality [0.8, 1] -> label 1, [0.6, 0.8) -> label 2
    # [0.4, 0.6] -> label 3, [0.2, 0.4) -> label 1
    # [0, 0.2) -> label 0
    ds.target[i] = int( data[i, nc-1] / 0.2 )
    if ds.target[i] == 5:
      ds.target[i] = 4
  ds.features = data[:, 0:(nc-1)]
  print 'label ', np.sum( ds.target == 0 ), np.sum( ds.target == 1 ), \
    np.sum( ds.target == 2 ), np.sum( ds.target == 3 ), np.sum( ds.target == 4 )
  return ds


###########################################
#### Inherently multiclass iwth Naive Bayes
###########################################

# Gaussian Naive Bayes
def gaussianNB_classify( ds ):
  u'''Gaussian Naive Bayes for classification'''
  # For Gaussian Naive Bayes
  from sklearn.naive_bayes import GaussianNB
  gnb = GaussianNB()
  
  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )
    
    # scaler = StandardScaler().fit( X_train )
    # X_train = scaler.transform( X_train )
    # X_val = scaler.transform( X_val )

    y_pred = gnb.fit( X_train, y_train ).predict( X_val )
    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy 

# Multinomial Naive Bayes
# Require integer count features but sometime fraction (like tf-idf) will work
def multinomialNB_classify( ds ):
  u'''Multinomial naive bayes'''
  from sklearn.naive_bayes import MultinomialNB
  clf = MultinomialNB()

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )
    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy 

# Linear Discriminant Analysis (LDA)
def LDA_classify( ds ):
  u'''LDA classify'''
  from sklearn.lda import LDA
  clf = LDA()

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )
    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy 

# QDA
def QDA_classify( ds ):
  u'''QDA classify'''
  from sklearn.qda import QDA
  clf = QDA()

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )
    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy 

# Decision Tree Classifier
def decisionTree_classify( ds ):
  u'''Decision Tree Classifier'''
  from sklearn import tree
  clf = tree.DecisionTreeClassifier()

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )
    score = clf.score(X_val, y_val)
    # print ('Classification report for classifier %s:\n%s\n' \
    #      % ( clf, metrics.classification_report(y_val, y_pred) ) )

    # print ('Confusion matrix:\n%s' \
    #      % metrics.confusion_matrix(y_val, y_pred) )

    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy, 'score = ', score
    # print clf.get_params()

# Random forest Classifier
def randomForest_classify( ds ):
  u'''Random Forest Tree Classifier'''
  from sklearn.ensemble import RandomForestClassifier
  clf = RandomForestClassifier( n_estimators= 10 )

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )

    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy 

# Extra Tree Classifier
def extraTree_classify( ds ):
  u'''Extra Tree Classifier'''
  from sklearn.ensemble import ExtraTreesClassifier
  clf = ExtraTreesClassifier( n_estimators=250 )

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )
    
    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy

    # importances = clf.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in clf.estimators_],
    #          axis=0)
    # indices = np.argsort(importances)[::-1]

    # # Print the feature ranking
    # print("Feature ranking:")
    # num_features = 65

    # for f in range(num_features):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(num_features), importances[indices],
    #        color="r", yerr=std[indices], align="center")
    # plt.xticks(range(num_features), indices)
    # plt.xlim([-1, num_features])
    # plt.show()


# AdaBoost Classifier
def adaBoost_classify( ds ):
  u'''AdaBoost Classifier'''
  from sklearn.ensemble import AdaBoostClassifier
  clf = AdaBoostClassifier( n_estimators= 10, learning_rate = 1.0 )

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )

    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy 

# Nearest centroid classifier
def nearestCentroid_classify( ds ):
  u'''Nearest neighbors classifier'''
  from sklearn.neighbors.nearest_centroid import NearestCentroid
  clf = NearestCentroid()

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )

    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy

# k-Nearest neighbor classifier
def knearestNeighbor_classify( ds ):
  u'''Nearest neighbors classifier'''
  from sklearn.neighbors import KNeighborsClassifier
  clf = KNeighborsClassifier()

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )

    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy

# Linear model
def linear_svm_sgd_classify( ds ):
  u'''Linear model classifier'''
  from sklearn import linear_model
  clf = linear_model.SGDClassifier(loss='hinge')

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )

    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy  

# logistic model
def logistic_classify( ds ):
  u'''Linear model classifier'''
  from sklearn import linear_model
  clf = linear_model.LogisticRegression()

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )

    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy  

# SVC
def svc_classify( ds ):
  u'''SVM.SVC Classify for dataset'''
  # Create a classifier: a support vector classifier
  # Using SVM (one-vs-one)
  # Setting for hyperparameters (gamma) of RBF Kernel (Gaussian kernel)
  # The gamma is larger, the more difficult of boundary condition
  #classifier = svm.SVC(gamma = 2, C=1)
  classifier = svm.SVC(kernel='linear', C=0.025)

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )
    
    # scaler = StandardScaler().fit( X_train )
    # X_train = scaler.transform( X_train )
    # X_val = scaler.transform( X_val )

    classifier.fit( X_train, y_train )
    y_pred = classifier.predict( X_val )

    # print ('Classification report for classifier %s:\n%s\n' \
    #      % ( classifier, metrics.classification_report(y_val, y_pred) ) )

    # print ('Confusion matrix:\n%s' \
    #      % metrics.confusion_matrix(y_val, y_pred) )

    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy 

# One-Vs-Rest
def oneVsRest_classify( ds ):
  u'''One vs rest classifier'''
  from sklearn.multiclass import OneVsRestClassifier
  from sklearn.svm import LinearSVC
  clf = OneVsRestClassifier( LinearSVC() )

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )

    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy

# One-Vs-One
def oneVsOne_classify( ds ):
  u'''One vs rest classifier'''
  from sklearn.multiclass import OneVsOneClassifier
  from sklearn.svm import LinearSVC
  clf = OneVsOneClassifier( LinearSVC() )

  TEST_SIZE = 0.2
  NUM_ITERS = 10
  for i in range(NUM_ITERS):
    X_train, X_val, y_train, y_val = \
      train_test_split( ds.features, ds.target, test_size = TEST_SIZE )

    y_pred = clf.fit( X_train, y_train ).predict( X_val )

    accuracy = float( (y_val == y_pred).sum() ) / float( y_val.shape[0] )
    print 'Iter i =', i, 'accuracy = ', accuracy

def _test():
  filename = os.path.join(BASE_DIR, 'yelp_all_features.txt')
  #flist = range(13,63)
  flist = range(13)
  #flist = range(63)
  flist.append(65)
  usecols = tuple(flist)
  #usecols = None
  ds = make_data(filename, usecols)
  TEST_SIZE = 0.2
  RANDOM_STATE = 0
  X_train, X_val, y_train, y_val = \
    train_test_split(ds.features, ds.target, test_size = TEST_SIZE, random_state = RANDOM_STATE)
  
  print X_train.shape, X_val.shape, y_train.shape, y_val.shape
  
  #gaussianNB_classify( ds )  # Bad (30%)
  #multinomialNB_classify( ds ) # Bad (40%)
  # LDA_classify( ds ) # Normal and fast (70%)
  # QDA_classify( ds ) # Fair bad (50%)
  
  #adaBoost_classify( ds ) # Fair bad (63%)
  #decisionTree_classify( ds ) # VERY GOOD and FAST (99%)
  #randomForest_classify( ds ) # fair good and fast (80%)
  # extraTree_classify( ds ) # fair good (80%) but slow

  # nearestCentroid_classify( ds ) # bad (8%)but only text is 30%
  # knearestNeighbor_classify( ds ) # bad about 60%
  
  # svc_classify(ds) # Normal but very slow (77%)
  # linear_svm_sgd_classify( ds ) # normal 70% and normal fast
  logistic_classify( ds ) # Fair good and normal fast (92%)  
  
  # oneVsRest_classify( ds ) # bad and slow about 54%, linearSVC
  # oneVsOne_classify( ds ) # GOOD but slow about 98%, linearSVC, computational
  
  return True

if __name__ == '__main__':
  _test()