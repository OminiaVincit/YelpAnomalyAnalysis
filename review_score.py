#!env python
# -*- coding:utf-8 -*-
'''Calculate score for first kth topics
'''

import os
import time
import sys
import multiprocessing

from settings import Settings
from data_utils import Reviews
from data_utils import Businesses

from topic_predict import Predict

NUMTOPICS = 50
EPSILON = 1e-7

def _test():
    u'''Add topics score for skipped reviews'''
    reviews = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    reviews.load_all_data()
    count = 0
    for review in reviews.cursor:
        if review.get('topics') == None or type(review['topics']) == type(list()):
            predict = Predict()
            topics = predict.run(review['text'])
            top_dict = {x:0 for x in range(NUMTOPICS)}
            for tp in topics:
                top_dict[tp[0]] = tp[1]
            reviews.collection.update({'_id' : review['_id']}, \
                {'$set' : {'topics' : {\
                    '0': top_dict[0], '1': top_dict[1], '2': top_dict[2], \
                    '3': top_dict[3], '4': top_dict[4], '5': top_dict[5], \
                    '6': top_dict[6], '7': top_dict[7], '8': top_dict[8], \
                    '9': top_dict[9], '10': top_dict[10], '11': top_dict[11], \
                    '12': top_dict[12], '13': top_dict[13], '14': top_dict[14], \
                    '15': top_dict[15], '16': top_dict[16], '17': top_dict[17], \
                    '18': top_dict[18], '19': top_dict[19], '20': top_dict[20], \
                    '21': top_dict[21], '22': top_dict[22], '23': top_dict[23], \
                    '24': top_dict[24], '25': top_dict[25], '26': top_dict[26], \
                    '27': top_dict[27], '28': top_dict[28], '29': top_dict[29], \
                    '30': top_dict[30], '31': top_dict[31], '32': top_dict[32], \
                    '33': top_dict[33], '34': top_dict[34], '35': top_dict[35], \
                    '36': top_dict[36], '37': top_dict[37], '38': top_dict[38], \
                    '39': top_dict[39], '40': top_dict[40], '41': top_dict[41], \
                    '42': top_dict[42], '43': top_dict[43], '44': top_dict[44], \
                    '45': top_dict[45], '46': top_dict[46], '47': top_dict[47], \
                    '48': top_dict[48], '49': top_dict[49]\
                }}}, \
                False, True)
            count += 1
            print count

def business_topic(ktops):
    u'''Add histogram of topics for each business'''
    businesses_collection = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION).collection
    businesses_cursor = businesses_collection.find().batch_size(50)
    reviews_collection = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION).collection
    count = 0
    for business in businesses_cursor:
        if business.get('topics') == None:
            reviews_cursor = reviews_collection.find({'business_id' : business['business_id']}).batch_size(50)
            top_dict = {x:0 for x in range(-1, NUMTOPICS)}
            for review in reviews_cursor:
                rv_topics = rank_topics(review, ktops)
                for tp in rv_topics:
                    top_dict[tp] += rv_topics[tp]
            rv_count = business['review_count']
            if rv_count > 0:
                top_sum = sum(top_dict.itervalues())
                diff = rv_count - top_sum
                if diff > EPSILON:
                    top_dict[-1] = diff
                    top_sum = rv_count
                for key in top_dict:
                    top_dict[key] = top_dict[key] / top_sum
            print top_dict
            businesses_collection.update({'_id' : business['_id']}, \
                {'$set' : {'topics' : {'-1': top_dict[-1],\
                    '0': top_dict[0], '1': top_dict[1], '2': top_dict[2], \
                    '3': top_dict[3], '4': top_dict[4], '5': top_dict[5], \
                    '6': top_dict[6], '7': top_dict[7], '8': top_dict[8], \
                    '9': top_dict[9], '10': top_dict[10], '11': top_dict[11], \
                    '12': top_dict[12], '13': top_dict[13], '14': top_dict[14], \
                    '15': top_dict[15], '16': top_dict[16], '17': top_dict[17], \
                    '18': top_dict[18], '19': top_dict[19], '20': top_dict[20], \
                    '21': top_dict[21], '22': top_dict[22], '23': top_dict[23], \
                    '24': top_dict[24], '25': top_dict[25], '26': top_dict[26], \
                    '27': top_dict[27], '28': top_dict[28], '29': top_dict[29], \
                    '30': top_dict[30], '31': top_dict[31], '32': top_dict[32], \
                    '33': top_dict[33], '34': top_dict[34], '35': top_dict[35], \
                    '36': top_dict[36], '37': top_dict[37], '38': top_dict[38], \
                    '39': top_dict[39], '40': top_dict[40], '41': top_dict[41], \
                    '42': top_dict[42], '43': top_dict[43], '44': top_dict[44], \
                    '45': top_dict[45], '46': top_dict[46], '47': top_dict[47], \
                    '48': top_dict[48], '49': top_dict[49]\
                }}}, \
                False, True)
            count += 1
            print business['id'], count

def business_topic_worker(identifier, skip, count, ktops):
    u'''Add histogram of topics for each business by worker'''
    done = 0
    start = time.time()
    businesses_collection = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION).collection
    reviews_collection = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION).collection
    batch_size = 50
    for batch in range(0, count, batch_size):
        lm_size = min(batch_size, count - batch)
        businesses_cursor = businesses_collection.find().skip(skip + batch).limit(lm_size)
        for business in businesses_cursor:
            if business.get('topics') == None:
                reviews_cursor = reviews_collection.find({'business_id' : business['business_id']}).batch_size(batch_size)
                top_dict = {x:0 for x in range(-1, NUMTOPICS)}
                for review in reviews_cursor:
                    rv_topics = rank_topics(review, ktops)
                    for tp in rv_topics:
                        top_dict[tp] += rv_topics[tp]
                rv_count = business['review_count']
                if rv_count > 0:
                    top_sum = sum(top_dict.itervalues())
                    diff = rv_count - top_sum
                    if diff > EPSILON:
                        top_dict[-1] = diff
                        top_sum = rv_count
                    for key in top_dict:
                        top_dict[key] = top_dict[key] / top_sum

                businesses_collection.update({'_id' : business['_id']}, \
                    {'$set' : {'topics' : {'-1': top_dict[-1],\
                        '0': top_dict[0], '1': top_dict[1], '2': top_dict[2], \
                        '3': top_dict[3], '4': top_dict[4], '5': top_dict[5], \
                        '6': top_dict[6], '7': top_dict[7], '8': top_dict[8], \
                        '9': top_dict[9], '10': top_dict[10], '11': top_dict[11], \
                        '12': top_dict[12], '13': top_dict[13], '14': top_dict[14], \
                        '15': top_dict[15], '16': top_dict[16], '17': top_dict[17], \
                        '18': top_dict[18], '19': top_dict[19], '20': top_dict[20], \
                        '21': top_dict[21], '22': top_dict[22], '23': top_dict[23], \
                        '24': top_dict[24], '25': top_dict[25], '26': top_dict[26], \
                        '27': top_dict[27], '28': top_dict[28], '29': top_dict[29], \
                        '30': top_dict[30], '31': top_dict[31], '32': top_dict[32], \
                        '33': top_dict[33], '34': top_dict[34], '35': top_dict[35], \
                        '36': top_dict[36], '37': top_dict[37], '38': top_dict[38], \
                        '39': top_dict[39], '40': top_dict[40], '41': top_dict[41], \
                        '42': top_dict[42], '43': top_dict[43], '44': top_dict[44], \
                        '45': top_dict[45], '46': top_dict[46], '47': top_dict[47], \
                        '48': top_dict[48], '49': top_dict[49]\
                    }}}, \
                    False, True)
            done += 1
            if done % 100 == 0:
                finish = time.time()
                print ' Business topic worker' + str(identifier) + ': Done ' + str(done) + \
                    ' out of ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
                    ' sec ~ ' + ('%.2f' % (done / (finish - start))) + '/sec'
                sys.stdout.flush()

def business_topic_parallel(ktops):
    u'''Add histogram of topics for each business by multi-threads'''
    bss = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION)
    bss.load_all_data()
    workers = 4
    batch = bss.count / workers
    jobs = []
    for i in range(workers):
        _ps = multiprocessing.Process(target=business_topic_worker, \
            args=((i+1), i*batch, batch, ktops))
        jobs.append(_ps)
        _ps.start()

    # Add remainder parts to process
    _rp = multiprocessing.Process(target=business_topic_worker, args=((workers+1), \
        workers*batch, bss.count - workers*batch, ktops))
    jobs.append(_rp)
    _rp.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)

def rank_topics(review, ktops):
    u'''Rearrange the topic distribution of review for top k topics'''
    import numpy as np

    top_rs = {}
    if review.get('topics'):
        top_dict = review['topics']
        top_arr = np.zeros(NUMTOPICS)
        for i in range(NUMTOPICS):
            top_arr[i] = top_dict[str(i)]
        top_arr *= np.var(top_arr)
        top_arr[top_arr < EPSILON] = 0.0
        idx = (-top_arr).argsort()[:ktops]
        sum_norm = np.sum(top_arr[idx])
        if sum_norm > EPSILON:
            top_arr[idx] /= sum_norm
            for ix in idx:
                top_rs[ix] = top_arr[ix]
    return top_rs

if __name__ == '__main__':
    # business_topic_parallel(3)
    # Call normal function once more for update correctly
    business_topic(3)
    #_test()