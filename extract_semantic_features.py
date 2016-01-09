#!env python
# -*- coding:utf-8 -*-
'''Calculate for tf-idf features score
'''

import os
import time
import sys
import nltk
import json
import numpy as np
import multiprocessing

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import stopwords
from nltk.tokenize import regexp
import re

from settings import Settings
from data_utils import Reviews, GenCollection

import json

def freq(word, tokens):
    return tokens.count(word)

def word_count(tokens):
    return len(tokens)

def tf(word, tokens):
    return (freq(word, tokens) / float(word_count(tokens)))

def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count


def idf(word, list_of_docs):
    return np.log(len(list_of_docs) /
            float(num_docs_containing(word, list_of_docs)))

def make_tf_idf_reviews(collection_name):
    # Debug time
    done = 0
    start = time.time()

    # For site name
    idx = collection_name.find('_')
    name = collection_name[0:idx]

    stopwds = stopwords.words('english')
    tokenizer = regexp.RegexpTokenizer("[\w’]+", flags=re.UNICODE)
    lem = WordNetLemmatizer()
    
    # Compute the frequency for each term
    vocabulary = {} # number of docs where word w appeared
    docs = {}
    rvs = Reviews(collection_name=collection_name)
    rvs.cursor = rvs.collection.find()
    N = rvs.cursor.count()

    count = 0
    for review in rvs.cursor:
        if not review.get('text'):
            continue
        votes = int(review['votes'])
        helpful = int(review['helpful'])
        rvid = review['review_id']
        docs[rvid] = {'freq':{}, 'tf':{}}
        
        tokens = tokenizer.tokenize(review['text'])
        tokens = [lem.lemmatize(token.lower()) for token in tokens if len(token) > 2]
        tokens = [token for token in tokens if token not in stopwds]

        for token in set(tokens):
            # The frequency computed for each review
            docs[rvid]['freq'][token] = freq(token, tokens)
            # The true-frequency (normalized)
            docs[rvid]['tf'][token] = tf(token, tokens)
            if token not in vocabulary:
                vocabulary[token] = 1
            else:
                vocabulary[token] += 1
        count += 1
        if count % 1000 == 0:
            end = time.time()
            print str(name) + ' get vocal: Done ' + str(count) + \
                ' out of ' + str(N) + ' reviews in ' + \
                ('%.2f' % (end - start)) + ' sec ~ ' + \
                ('%.2f' % (count / (end - start))) + '/sec'
            sys.stdout.flush()

    # Number of processed documents
    vocabulary['NUMDOCS'] = count
    with open(name + '_tfidf_vocabularies.json', 'w') as _vfile:
        _vfile.write(json.dumps(vocabulary, indent=1).replace('\n', ''))

    count = 0
    rvs.cursor = rvs.collection.find()

    # output file
    filename = name + '_tfidf_tokens.json'
    
    with open(filename, 'w') as _file:
        for review in rvs.cursor:
            if not review.get('text'):
                continue
            rvid = review['review_id']
            if rvid not in docs:
                continue

            features = {}
            features['review_id'] = review['review_id']
            features['item_id'] = review['item_id']
            features['user_id'] = review['user_id']
            features['votes'] = review['votes']
            features['helpful'] = review['helpful']
            features['freq']  = docs[rvid]['freq']
            features['tf']  = docs[rvid]['tf']
            features['idf'] = {}

            for token in docs[rvid]['tf']:
                # The inverse-document-frequency
                features['idf'][token] = np.log(vocabulary['NUMDOCS'] / float(vocabulary[token]) )
            _file.write(json.dumps(features, indent=1).replace('\n', ''))
            _file.write('\n')

            count += 1
            if count % 1000 == 0:
                end = time.time()
                print str(name) + ' write file: Done ' + str(count) + \
                    ' out of ' + str(N) + ' reviews in ' + \
                    ('%.2f' % (end - start)) + ' sec ~ ' + \
                    ('%.2f' % (count / (end - start))) + '/sec'
                sys.stdout.flush()

def load_tfidf_vocal(site, dim):
    """
    Load vocabulary file (with dimension dim) for tfidf model
    """
    vocal_file = site + '_tfidf_vocabularies.json'
    vocal = []
    with open(os.path.join(Settings.DATA_DIR, vocal_file), 'r') as in_f:
        for line in in_f:
            data = json.loads(line)
            num_docs = data['NUMDOCS']
            threshold = num_docs / 128
            print num_docs, len(data)
            # count = 0
            # for token in data:
            #     freq = data[token]
            #     if freq >= threshold:
            #         count += 1
            #         print count, token, freq

            dlist = sorted(data.items(), key=lambda x: -x[1])
            for i in range(1, dim + 1):
                vocal.append(dlist[i])
    return vocal

def extract_tfidf_features(collection_name, dim):
    """
    Extract tfidf features for reviews with >= 10 votes
    """
    # Debug time
    start = time.time()

    # For site name
    idx = collection_name.find('_')
    site = collection_name[0:idx]

    # Load vocabulary file
    vocal = load_tfidf_vocal(site, dim)

    rvs = Reviews(collection_name=collection_name)
    rvs.cursor = rvs.collection.find()
    N = rvs.cursor.count()

    data = []
    done = 0
    for review in rvs.cursor:
        rvid = review['review_id']
        votes = review['votes']
        if votes < 10:
            continue
        helpful = review['helpful']
        features = np.zeros((dim+3), )
        features[dim] = helpful
        features[dim+1] = votes
        features[dim+2] = helpful / float(votes)

        # Extract tf-idf value
        for i in range(dim):
            token = vocal[i][0]
            if token in review['idf']:
                features[i] = review['idf'][token] * review['tf'][token]
        data.append(features)

        done += 1
        if done % 100 == 0:
            end = time.time()
            print str(site) + ' tfidf features: Done ' + str(done) + \
                ' out of ' + str(N) + ' reviews in ' + \
                ('%.2f' % (end - start)) + ' sec ~ ' + \
                ('%.2f' % (done / (end - start))) + '/sec'
            sys.stdout.flush()

    print 'Number of processed reviews ', done
    data = np.vstack(data)
    print 'Data shape', data.shape
    np.save('%s_tfidf_features' % site, data)

def load_GALC_vocal(vocal_file='GALC_0.csv'):
    """
    Load vocabulary file (with dimension dim) for GALC model
    """
    vocal = {}
    with open(os.path.join(Settings.DATA_DIR, vocal_file), 'r') as in_f:
        index = 0
        for line in in_f:
            data = re.split(',|\r|\n|\r\n', line)
            N = len(data)
            # Skip first token (category name)
            for i in range(1, N):
                if len(data[i]) > 0:
                    vocal[data[i]] = index
            index += 1
    return vocal


def galc_eval(vocal, token, outbound):
    """
    Return index of emotion in GALC data for token
    """
    if token in vocal:
        return vocal[token]
    else:
        strlen = len(token)
        for i in range(strlen):
            tmp = token[:(strlen-i)] + '*'
            if tmp in vocal:
                return vocal[tmp]
    return outbound


def extract_GALC_features(collection_name, dim):
    """
    Extract tfidf features for reviews with >= 10 votes
    """
    # Debug time
    start = time.time()

    # For site name
    idx = collection_name.find('_')
    site = collection_name[0:idx]

    # Load vocabulary file
    vocal = load_GALC_vocal()

    rvs = Reviews(collection_name=collection_name)
    rvs.cursor = rvs.collection.find()
    N = rvs.cursor.count()

    data = []
    done = 0
    for review in rvs.cursor:
        rvid = review['review_id']
        votes = review['votes']
        if votes < 10:
            continue
        helpful = review['helpful']
        features = np.zeros((dim+3), )
        features[dim] = helpful
        features[dim+1] = votes
        features[dim+2] = helpful / float(votes)

        # Extract GALC value
        for token in review['idf']:
            categ = galc_eval(vocal, token, dim-1)
            features[categ] += 1 * review['freq'][token]

        data.append(features)
        #print done, features

        done += 1
        if done % 100 == 0:
            end = time.time()
            print str(site) + ' GALC features: Done ' + str(done) + \
                ' out of ' + str(N) + ' reviews in ' + \
                ('%.2f' % (end - start)) + ' sec ~ ' + \
                ('%.2f' % (done / (end - start))) + '/sec'
            sys.stdout.flush()

    print 'Number of processed reviews ', done
    data = np.vstack(data)
    print 'Data shape', data.shape
    np.save('%s_GALC_features' % site, data)

def load_LIWC_vocal(vocal_file='LIWC2007.dic'):
    """
    Load vocabulary file (with dimension dim) for LIWC model
    """
    vocal = {}
    fet = {}
    with open(os.path.join(Settings.DATA_DIR, vocal_file), 'r') as in_f:
        index = 0
        for line in in_f:
            data = re.split(' |\t|\r|\n|\r\n', line)
            N = len(data)
            # map between feature value and feature index
            if N > 0 and data[0].isdigit():
                fet[data[0]] = index
                index += 1
            elif N > 1:
                features = np.zeros((index, ), dtype=np.int32)
                for i in range(1, N):
                    if len(data[i]) > 0 and data[i].isdigit():
                        mdx = fet[data[i]]
                        features[mdx] = 1

                vocal[data[0]] = features
    print len(vocal)        
    return vocal

def liwc_eval(vocal, token, dim):
    """
    Return index of emotion in GALC data for token
    """
    if token in vocal:
        return vocal[token]
    else:
        strlen = len(token)
        for i in range(strlen):
            tmp = token[:(strlen-i)] + '*'
            if tmp in vocal:
                return vocal[tmp]
    return np.zeros((dim, ), dtype=np.int32)

def extract_LIWC_features(collection_name, dim):
    """
    Extract tfidf features for reviews with >= 10 votes
    """
    # Debug time
    start = time.time()

    # For site name
    idx = collection_name.find('_')
    site = collection_name[0:idx]

    # Load vocabulary file
    vocal = load_LIWC_vocal()

    rvs = Reviews(collection_name=collection_name)
    rvs.cursor = rvs.collection.find()
    N = rvs.cursor.count()

    data = []
    done = 0
    for review in rvs.cursor:
        rvid = review['review_id']
        votes = review['votes']
        if votes < 10:
            continue
        helpful = review['helpful']
        features = np.zeros((dim+3), )
        features[dim] = helpful
        features[dim+1] = votes
        features[dim+2] = helpful / float(votes)

        # Extract GALC value
        for token in review['idf']:
            features[:dim] += liwc_eval(vocal, token, dim) * review['freq'][token]

        data.append(features)
        # print done, features

        done += 1
        if done % 100 == 0:
            end = time.time()
            print str(site) + ' LIWC features: Done ' + str(done) + \
                ' out of ' + str(N) + ' reviews in ' + \
                ('%.2f' % (end - start)) + ' sec ~ ' + \
                ('%.2f' % (done / (end - start))) + '/sec'
            sys.stdout.flush()

    print 'Number of processed reviews ', done
    data = np.vstack(data)
    print 'Data shape', data.shape
    np.save('%s_LIWC_features' % site, data)

def load_INQUIRER_vocal(dim, vocal_file='inquirerbasic_remove_col.csv'):
    """
    Load vocabulary file (with dimension dim) for LIWC model
    """
    vocal = {}
    with open(os.path.join(Settings.DATA_DIR, vocal_file), 'r') as in_f:
        index = 0
        for line in in_f:
            data = re.split(',|\r|\n|\r\n', line)
            N = len(data)
            if index == 0:
                index += 1
                continue

            # map between feature value and feature index
            assert(dim <= N-1)
            if N > 0:
                features = np.zeros((dim, ), dtype=np.int32)
                for i in range(dim):
                    if len(data[i+1]) > 0:
                        features[i] = 1
                token = data[0].lower()
                vocal[token] = features
    print len(vocal)        
    return vocal

def inquirer_eval(vocal, token, dim):
    """
    Return index of emotion in INQUIRER data for token
    """
    if token in vocal:
        return vocal[token]
    else:
        tmp = token + '#1'
        if tmp in vocal:
            return vocal[tmp]
    return np.zeros((dim, ), dtype=np.int32)

def extract_INQUIRER_features(collection_name, dim):
    """
    Extract tfidf features for reviews with >= 10 votes
    """
    # Debug time
    start = time.time()

    # For site name
    idx = collection_name.find('_')
    site = collection_name[0:idx]

    # Load vocabulary file
    vocal = load_INQUIRER_vocal(dim)

    rvs = Reviews(collection_name=collection_name)
    rvs.cursor = rvs.collection.find()
    N = rvs.cursor.count()

    data = []
    done = 0
    for review in rvs.cursor:
        rvid = review['review_id']
        votes = review['votes']
        if votes < 10:
            continue
        helpful = review['helpful']
        features = np.zeros((dim+3), )
        features[dim] = helpful
        features[dim+1] = votes
        features[dim+2] = helpful / float(votes)

        # Extract INQUIRER value
        for token in review['idf']:
            #token = token.lower()
            features[:dim] += inquirer_eval(vocal, token, dim) * review['freq'][token]
        
        data.append(features)
        # print done, features

        done += 1
        if done % 100 == 0:
            end = time.time()
            print str(site) + ' INQUIRER features: Done ' + str(done) + \
                ' out of ' + str(N) + ' reviews in ' + \
                ('%.2f' % (end - start)) + ' sec ~ ' + \
                ('%.2f' % (done / (end - start))) + '/sec'
            sys.stdout.flush()

    print 'Number of processed reviews ', done
    data = np.vstack(data)
    print 'Data shape', data.shape
    np.save('%s_INQUIRER_features' % site, data)
    
if __name__ == '__main__':
    #test()
    #extract_text_features(Settings.MOVIES_REVIEWS_COLLECTION)
    #make_tf_idf_reviews(Settings.YELP_REVIEWS_COLLECTION)
    #make_tf_idf_reviews(Settings.TRIPADVISOR_REVIEWS_COLLECTION)
    #load_tfidf_vocal('yelp')
    #extract_tfidf_features(Settings.YELP_TFIDF_COLLECTION, dim=1024)
    #extract_tfidf_features(Settings.TRIPADVISOR_TFIDF_COLLECTION, dim=1024)
    extract_GALC_features(Settings.TRIPADVISOR_TFIDF_COLLECTION, dim=Settings.GALC_DIM)
    extract_GALC_features(Settings.YELP_TFIDF_COLLECTION, dim=Settings.GALC_DIM)
    #load_LIWC_vocal()
    extract_LIWC_features(Settings.YELP_TFIDF_COLLECTION, dim=Settings.LIWC_DIM)
    extract_LIWC_features(Settings.TRIPADVISOR_TFIDF_COLLECTION, dim=Settings.LIWC_DIM)
    #load_INQUIRER_vocal(dim=Settings.INQUIRER_DIM)
    extract_INQUIRER_features(Settings.TRIPADVISOR_TFIDF_COLLECTION, dim=Settings.INQUIRER_DIM)
    extract_INQUIRER_features(Settings.YELP_TFIDF_COLLECTION, dim=Settings.INQUIRER_DIM)