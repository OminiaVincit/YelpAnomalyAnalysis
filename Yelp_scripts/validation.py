u'''Validation class'''

import numpy as np
from pymongo import MongoClient
from settings import Settings
from optimize import RegularOptimizer
import itertools

class DataSource(object):
    u'''Loading data from source'''

    def __init__(self):
        u'''Initialize data'''
        self.data = None
        self.target = None

    def load_data_from_mongo(self, connection_dir=Settings.MONGO_CONNECTION_STRING, \
        database_name=Settings.YELP_DATABASE, collection_name=Settings.INPUT_COLLECTION):
        u'''Load data from mongo database'''
        collection = MongoClient(connection_dir)[database_name][collection_name]
        cursor = collection.find().batch_size(50)
        size = cursor.count()
        num_features = 18
        X = np.zeros(shape=(size, num_features))
        Q = np.zeros(shape=(size, 1))
        m = 0
        EPSILON = 1e-6
        for features in cursor:
            X[m][0] = float(features['num_sent'])
            X[m][1] = float(features['sent_len'])
            X[m][2] = float(features['num_token'])
            X[m][3] = float(features['uniq_word_ratio'])
            X[m][4] = float(features['pos_nn'])
            X[m][5] = float(features['pos_adj'])
            X[m][6] = float(features['pos_comp'])
            X[m][7] = float(features['pos_v'])
            X[m][8] = float(features['pos_rb'])
            X[m][9] = float(features['pos_fw'])
            X[m][10] = float(features['pos_cd'])
            
            X[m][11] = float(features['rate_deviation'])
            
            X[m][12] = float(features['business.global_topics_50_normally'])
            X[m][13] = float(features['business.global_topics_50_anomaly_minor_major'])
            X[m][14] = float(features['business.global_topics_50_anomaly_major_minor'])
            
            X[m][15] = float(features['user.global_topics_50_normally'])
            X[m][16] = float(features['user.global_topics_50_anomaly_minor_major'])
            X[m][17] = float(features['user.global_topics_50_anomaly_major_minor'])
            
            #X[m][18] = float(features['user.local_topics_5_normally'])
            #X[m][19] = float(features['user.local_topics_5_anomaly_minor_major'])
            #X[m][20] = float(features['user.local_topics_5_anomaly_major_minor'])

            Q[m] = float(features['quality'])
            
            #print X[m][0], X[m][1], X[m][2], X[m][3], X[m][4], X[m][5], X[m][6], \
            #    X[m][7], X[m][8], X[m][9], X[m][10], X[m][11], X[m][12], X[m][13], \
            #    X[m][14], X[m][15], X[m][16], X[m][17],\
            #    Q[m][0], features['votes']
            m += 1

        self.data = X
        self.target = Q

    def load_data_from_txt_file(self, filename):
        u'''Load data from file'''
        X = np.genfromtxt(filename, delimiter=' ', dtype=np.float32)
        self.data = X[:, 0:(X.shape[1]-1)]
        self.target = X[:, X.shape[1]-1]

    def features_augmentation(self, num_feature, d):
        u'''Try adding polynomial features'''
        X = self.data[:, 0:num_feature]
        first_order = X
        n_r = X.shape[0]

        for s in range(2, d+1):
            n_ls = []
            for i in range(n_r):
                for elements in itertools.combinations_with_replacement(first_order[i, :], s):
                    n_ls.append(np.product(elements))
            n_c = len(n_ls) / n_r
            n_mtr = np.reshape(n_ls, (n_r, n_c))
            X = np.hstack((X, n_mtr))
        self.data = X


class Validation(object):
    u'''Validation class, divide train, cross and test'''

    def __init__(self):
        u'''Initialize validation class'''
        self.train = None
        self.train_target = None
        self.train_error = None

        self.mean_r = []
        self.std_r = []

        self.cross = None
        self.cross_target = None
        self.cross_error = None

        self.test = None
        self.test_target = None
        self.test_error = None

        self.ratio = None

    def load(self, features, target, ratio=[0.6, 0.2, 0.2]):
        u'''Divide data into training, cross, test'''
        # Ratio is list, Ex. [0.6, 0.2, 0.2]
        # 60% training, 20% cross, 20% test
        try:
            assert (len(ratio) == 3)
            assert np.sum(ratio) == 1.0
        except:
            ratio = [0.6, 0.2, 0.2]

        n_c = features.shape[1]
        n_r = features.shape[0]
        assert target.shape == (n_r, 1)

        num_train = int(n_r * ratio[0])
        num_cross = int(n_r * ratio[1])
        num_test = n_r - num_train - num_cross

        self.train = features[0:num_train, :]
        self.train_target = target[0:num_train]

        self.cross = features[num_train:(num_train + num_cross), :]
        self.cross_target = target[num_train:(num_train + num_cross)]

        self.test = features[(num_train + num_cross):(n_r + 1), :]
        self.test_target = target[(num_train + num_cross):(n_r + 1)]

        # Normalize training features
        self._normalize_features()

    def _normalize_features(self):
        u'''Normalize features'''
        mean_r = []
        std_r = []
        X = self.train
        train_norm = X
        cross_norm = self.cross
        test_norm = self.test
        
        n_c = X.shape[1]
        for i in range(n_c):
            m = np.mean(X[:, i])
            s = np.std(X[:, i])
            mean_r.append(m)
            std_r.append(s)

            train_norm[:, i] = (train_norm[:, i] - m) / float(s)

            # Normalize test and cross        
            # By mean and std
            cross_norm[:, i] = (cross_norm[:, i] - m) / float(s)
            test_norm[:, i] = (test_norm[:, i] - m) / float(s)
            
        self.train = train_norm
        self.cross = cross_norm
        self.test = test_norm

        self.mean_r = mean_r
        self.std_r = std_r

    
    def base_run(self):
        u'''Run base line test'''
        predict_val = np.mean(self.train_target)
        diff = self.train_target - predict_val
        self.train_error = diff.T.dot(diff) / float(len(diff))

        diff = self.cross_target - predict_val
        self.cross_error = diff.T.dot(diff) / float(len(diff))

        diff = self.test_target - predict_val
        self.test_error = diff.T.dot(diff) / float(len(diff))
        print self.train_error[0][0], self.cross_error[0][0], self.test_error[0][0]

    def run(self, alphas):
        u'''Run validation by optimizer'''
        for alpha in alphas:
            optimizer = RegularOptimizer(self.train, self.train_target, alpha = alpha)
            optimizer.solve()
            self._compute_RMSE(optimizer.weights)
            print alpha, self.train_error[0][0], self.cross_error[0][0], self.test_error[0][0]
            #diffs = self._residual(optimizer.weights)
            #for diff in diffs:
            #    print diff[0]

    def _compute_RMSE(self, weights):
        u'''Calculate cost for test and cross'''
        labels = ['train', 'cross', 'test']
        for label in labels:
            if label == 'train':
                X = self.train
                Q = self.train_target
            elif label == 'cross':
                X = self.cross
                Q = self.cross_target
            else:
                X = self.test
                Q = self.test_target

            n_c = X.shape[1]
            n_r = X.shape[0]
            X_it = np.ones(shape=(n_r, n_c + 1))
            X_it[:, 1:(n_c+1)] = X

            m = X_it.shape[0]
            Diff = X_it.dot(weights) - Q
            error = (Diff.T.dot(Diff)) / float(2*m)
            if label == 'train':
                self.train_error = error
            elif label == 'cross':
                self.cross_error = error
            else:
                self.test_error = error

    def _residual(self, weights):
        X = self.train
        Q = self.train_target
        n_c = X.shape[1]
        n_r = X.shape[0]
        X_it = np.ones(shape=(n_r, n_c + 1))
        X_it[:, 1:(n_c+1)] = X

        m = X_it.shape[0]
        Diff = X_it.dot(weights) - Q
        return Diff


def _test():
    u'''Test for class function'''
    source = DataSource()
    source.load_data_from_mongo()
    num_features = source.data.shape[1]

    # source.features_augmentation(num_features, d=2)
    X = source.data
    target = source.target
    # for quality in target:
    #     print quality[0]

    # val = Validation()
    # val.load(X, target, ratio=[0.6, 0.2, 0.2])
    # val.run([0])

    # rates = [0.2, 0.4, 0.6, 0.8, 1.0]
    # for rate in rates:
    #     ratio = [rate, (1.0-rate)/2.0, (1.0-rate)/2.0]
    #     for d in range(1, 4):
    #         source.features_augmentation(num_features, d)
    #         X = source.data
    #         target = source.target
    #         val = Validation()
    #         val.load(X, target, ratio=ratio)
    #         #val.base_run()
    #         val.run([0, 0.01, 0.1, 1, 10])

if __name__ == '__main__':
    _test()
