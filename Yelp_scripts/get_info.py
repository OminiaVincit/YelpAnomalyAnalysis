#!env python
# -*- coding:utf-8 -*-
'''Get info module for YELP challenge data set
Extract restaurant related data
'''

import json
import numpy as np
import time

from settings import Settings
from data_utils import Users
from data_utils import Businesses
from data_utils import Reviews


def _test():
    u'''Test for performance'''
    bss = Businesses()
    bss.load_all_data()
    items = []
    for business in bss.cursor:
        for catego in business['categories']:
            items.append(catego)
    chains = ' '.join(items)
    chains = chains.lower()
    chains = chains.replace('(', '')
    chains = chains.replace(')', '')
    chains = chains.replace('&', '')
    chains = chains.replace('/', ' ')
    chains = chains.replace('\\', ' ')
    print chains

def res_businesses_extract():
    u'''Extract restaurant businesses'''
    bss = Businesses()
    bss.load_all_data()
    bs_index = 0
    tmp = '\"_id\": null,'
    for business in bss.cursor:
        for catego in business['categories']:
            if catego.lower() == 'restaurants':
                bs_index += 1
                business['id'] = bs_index
                business['_id'] = None
                print json.dumps(business, indent=1).replace('\n', '').replace(tmp, '')

def res_reviews_extract():
    u'''Get list of restaurant businesses'''
    dict_bs = {}
    rbss = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION)
    rbss.load_all_data()
    for res_business in rbss.cursor:
        dict_bs[res_business['business_id']] = res_business['id']

    rvs = Reviews()
    rvs.load_all_data()
    rv_index = 0
    tmp = '\"_id\": null,'
    for review in rvs.cursor:
        if review['business_id'] in dict_bs:
            rv_index += 1
            review['_id'] = None
            review['id'] = rv_index
            review['p_business_id'] = dict_bs[review['business_id']]
            print json.dumps(review, indent=1).replace('\n', '').replace(tmp, '')

def res_users_extract():
    u'''Get list of users reviewed for restaurants'''
    dict_count = {}
    dict_star1 = {}
    dict_star2 = {}
    dict_star3 = {}
    dict_star4 = {}
    dict_star5 = {}

    rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    rvs.load_all_data()
    for review in rvs.cursor:
        user_id = review['user_id']
        if user_id not in dict_count:
            dict_star1[user_id] = 0
            dict_star2[user_id] = 0
            dict_star3[user_id] = 0
            dict_star4[user_id] = 0
            dict_star5[user_id] = 0
            dict_count[user_id] = 0

        stars = review['stars']
        dict_count[user_id] += 1
        if stars == 1:
            dict_star1[user_id] += 1
        if stars == 2:
            dict_star2[user_id] += 1
        if stars == 3:
            dict_star3[user_id] += 1
        if stars == 4:
            dict_star4[user_id] += 1
        if stars == 5:
            dict_star5[user_id] += 1

    users = Users()
    users.load_all_data()
    user_index = 0
    tmp = '\"_id\": null,'
    for user in users.cursor:
        user_id = user['user_id']
        star1 = 0
        star2 = 0
        star3 = 0
        star4 = 0
        star5 = 0
        count = 0

        if user_id in dict_count:
            star1 = dict_star1[user_id]
            star2 = dict_star2[user_id]
            star3 = dict_star3[user_id]
            star4 = dict_star4[user_id]
            star5 = dict_star5[user_id]
            count = dict_count[user_id]

        if count > 0:
            user_index += 1
            aveg = (star1*1.0 + star2*2.0 + star3*3.0 + star4*4.0 + star5*5.0)/count

            user['_id'] = None
            user['id'] = user_index
            user['review_count'] = count
            user['average_stars'] = aveg
            user['stars_distribution']['one'] = star1
            user['stars_distribution']['two'] = star2
            user['stars_distribution']['three'] = star3
            user['stars_distribution']['four'] = star4
            user['stars_distribution']['five'] = star5
            print json.dumps(user, indent=1).replace('\n', '').replace(tmp, '')

def res_businesses_extract2():
    u'''Get list of restaurants'''
    dict_count = {}
    dict_star1 = {}
    dict_star2 = {}
    dict_star3 = {}
    dict_star4 = {}
    dict_star5 = {}

    rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    rvs.load_all_data()
    for review in rvs.cursor:
        stars = review['stars']
        business_id = review['business_id']
        if business_id not in dict_count:
            dict_star1[business_id] = 0
            dict_star2[business_id] = 0
            dict_star3[business_id] = 0
            dict_star4[business_id] = 0
            dict_star5[business_id] = 0
            dict_count[business_id] = 0
        dict_count[business_id] += 1
        if stars == 1:
            dict_star1[business_id] += 1
        if stars == 2:
            dict_star2[business_id] += 1
        if stars == 3:
            dict_star3[business_id] += 1
        if stars == 4:
            dict_star4[business_id] += 1
        if stars == 5:
            dict_star5[business_id] += 1

    bss = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION)
    bss.load_all_data()
    business_index = 0
    tmp = '\"_id\": null,'
    for business in bss.cursor:
        business_id = business['business_id']
        star1 = 0
        star2 = 0
        star3 = 0
        star4 = 0
        star5 = 0
        count = 0

        if business_id in dict_count:
            star1 = dict_star1[business_id]
            star2 = dict_star2[business_id]
            star3 = dict_star3[business_id]
            star4 = dict_star4[business_id]
            star5 = dict_star5[business_id]
            count = dict_count[business_id]

        if count > 0:
            aveg = (star1*1.0 + star2*2.0 + star3*3.0 + star4*4.0 + star5*5.0)/count
        else:
            aveg = 0

        business_index += 1
        business['_id'] = None
        business['id'] = business_index
        business['review_count'] = count
        business['average_stars'] = aveg
        business['stars_distribution']['one'] = star1
        business['stars_distribution']['two'] = star2
        business['stars_distribution']['three'] = star3
        business['stars_distribution']['four'] = star4
        business['stars_distribution']['five'] = star5
        print json.dumps(business, indent=1).replace('\n', '').replace(tmp, '')

def res_reviews_update_puser_id():
    u'''Get list of restaurants businesses'''
    dict_us = {}
    res_users = Users(collection_name=Settings.RES_USERS_COLLECTION)
    res_users.load_all_data()
    for user in res_users.cursor:
        dict_us[user['user_id']] = user['id']

    rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    rvs.load_all_data()
    tmp = '\"_id\": null,'
    rv_index = 0
    for review in rvs.cursor:
        if review['user_id'] in dict_us:
            rv_index += 1
            review['_id'] = None
            review['id'] = rv_index
            review['p_user_id'] = dict_us[review['user_id']]
            print json.dumps(review, indent=1).replace('\n', '').replace(tmp, '')

def test_business_stars():
    u'''Test stars_distribution of res_business collection'''
    res_bss = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION)
    res_bss.load_all_data()
    for business in res_bss.cursor:
        print business['id']
        review_count = business['review_count']
        star1 = business['stars_distribution']['one']
        star2 = business['stars_distribution']['two']
        star3 = business['stars_distribution']['three']
        star4 = business['stars_distribution']['four']
        star5 = business['stars_distribution']['five']
        total_stars = star1 + star2 + star3 + star4 + star5
        if total_stars != review_count:
            break

def test_user_stars():
    u'''Test stars_distribution of res_users collection'''
    res_uss = Users(collection_name=Settings.RES_USERS_COLLECTION)
    res_uss.load_all_data()
    for user in res_uss.cursor:
        print user['id']
        review_count = user['review_count']
        star1 = user['stars_distribution']['one']
        star2 = user['stars_distribution']['two']
        star3 = user['stars_distribution']['three']
        star4 = user['stars_distribution']['four']
        star5 = user['stars_distribution']['five']
        total_stars = star1 + star2 + star3 + star4 + star5
        if total_stars != review_count:
            break

def num_reviews_businesses():
    u'''Number of reviews vs number of businesses'''
    res_bss = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION)
    res_bss.load_all_data()
    rv_bs = {}
    for bss in res_bss.cursor:
        rv_count = bss['review_count']
        if rv_count not in rv_bs:
            rv_bs[rv_count] = 1
        else:
            rv_bs[rv_count] += 1
    for key, value in rv_bs.iteritems():
        print key, value

def num_reviews_users():
    u'''Number of reviews vs number of users'''
    res_users = Users(collection_name=Settings.RES_USERS_COLLECTION)
    res_users.load_all_data()
    rv_us = {}
    for user in res_users.cursor:
        rv_count = user['review_count']
        if rv_count not in rv_us:
            rv_us[rv_count] = 1
        else:
            rv_us[rv_count] += 1
    for key, value in rv_us.iteritems():
        print key, value

def num_reviews_votes():
    u'''Number of reviews vs number of votes'''
    res_rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    res_rvs.load_all_data()
    rv_votes = {}
    for review in res_rvs.cursor:
        vote_count = review['votes']['useful'] + review['votes']['funny'] + review['votes']['cool']
        if vote_count not in rv_votes:
            rv_votes[vote_count] = 1
        else:
            rv_votes[vote_count] += 1
    for key, value in rv_votes.iteritems():
        print key, value

def num_reviews_helpful():
    u'''Number of reviews vs number of votes'''
    res_rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    res_rvs.load_all_data()
    rv_votes = {}
    for review in res_rvs.cursor:
        vote_count = review['votes']['useful']
        if vote_count not in rv_votes:
            rv_votes[vote_count] = 1
        else:
            rv_votes[vote_count] += 1
    for key, value in rv_votes.iteritems():
        print key, value

def num_reviews_helpful_percentage():
    u'''Number of reviews vs number of votes'''
    res_rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    res_rvs.load_all_data()
    rv_pt = {}
    for review in res_rvs.cursor:
        useful_count = review['votes']['useful']
        vote_count = review['votes']['useful'] + review['votes']['funny'] + review['votes']['cool']
        if vote_count != 0:
            percentage = float(useful_count) / float(vote_count)
            percentage = int(100*percentage)
            if percentage not in rv_pt:
                rv_pt[percentage] = 1
            else:
                rv_pt[percentage] += 1
    for key, value in rv_pt.iteritems():
        print key, value

def num_received_votes_users():
    u'''Number of received votes vs number of users'''
    res_users = Users(collection_name=Settings.RES_USERS_COLLECTION)
    res_users.load_all_data()
    recv_votes = {}
    for user in res_users.cursor:
        vote_count = user['votes']['useful'] + user['votes']['funny'] + user['votes']['cool']
        if vote_count not in recv_votes:
            recv_votes[vote_count] = 1
        else:
            recv_votes[vote_count] += 1
    for key, value in recv_votes.iteritems():
        print key, value

def num_received_helpful_votes_users():
    u'''Number of received helpful votes vs number of users'''
    res_users = Users(collection_name=Settings.RES_USERS_COLLECTION)
    res_users.load_all_data()
    recv_helpful = {}
    for user in res_users.cursor:
        vote_count = user['votes']['useful']
        if vote_count not in recv_helpful:
            recv_helpful[vote_count] = 1
        else:
            recv_helpful[vote_count] += 1
    for key, value in recv_helpful.iteritems():
        print key, value

def get_quality():
    u'''Quality difference between two reviews'''
    rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    rvs.load_all_data()
    votes_threshold = 10
    for review in rvs.cursor:
        useful_count = review['votes']['useful']
        votes = review['votes']['cool'] + review['votes']['funny'] + review['votes']['useful']
        if votes >= votes_threshold:
            rate = float(useful_count) / float(votes)
            print review['user_id'], review['business_id'], rate

def topics_deviation_hypothesis():
    u'''Determine the effect of topics deviation'''

    collection = Reviews(connection_dir=Settings.MONGO_CONNECTION_STRING, \
        database_name=Settings.YELP_DATABASE, collection_name=Settings.INPUT_COLLECTION).collection
    cursor = collection.find().batch_size(50)
        
    for features in cursor:
        X = np.zeros(8)
        X[0] = float(features['rate_deviation'])
            
        X[1] = float(features['business.global_topics_50_normally'])
        X[2] = float(features['business.global_topics_50_anomaly_minor_major'])
        X[3] = float(features['business.global_topics_50_anomaly_major_minor'])
            
        X[4] = float(features['user.global_topics_50_normally'])
        X[5] = float(features['user.global_topics_50_anomaly_minor_major'])
        X[6] = float(features['user.global_topics_50_anomaly_major_minor'])
        X[7] = float(features['quality'])
        print X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7] 

def review_topics_diff(filename):
    u'''Get the difference of topics distribution between reviews'''
    data = np.genfromtxt(filename, delimiter=' ', dtype=None)
    nr = 1000
    nc = 50
    #count = 0
    for i in range(nr):
        fri = np.zeros(nc)
        qi = float(data[i][nc])
        for m in range(nc):
            fri[m] = float(data[i][m])
        for j in range(i+1, nr):
            qj = float(data[j][nc])
            qdf = qi - qj

            for m in range(nc):
                frj = float(data[j][m])
            fdf = fri - frj
            #diff = np.sqrt(fdf.dot(fdf))
            diff = np.mean(abs(fdf))
            # frimax = np.max(fri)
            # frjmax = np.max(frj)
            # if frimax != 0:
            #     fri = fri / frimax
            # if frjmax != 0:
            #     frj = frj / frjmax
            # normally = 0.0
            # anom1 = 1.0
            # anom2 = 1.0
            # if frimax != 0 or frjmax != 0:
            #     normally = np.max(fri * frj)
            #     anom1 = np.max(fri * (1-frj))
            #     anom2 = np.max(frj * (1-fri))

            # print qdf, diff, normally, anom1, anom2
            # print -qdf, diff, normally, anom1, anom2

            # print KL divergence
            # w1, w2 = 0, 0
            # for m in range(nc):
            #     frj = float(data[j][m])
            #     if frj != 0 and fri[m] != 0:
            #         w1 += fri[m] * np.log(fri[m]/frj)
            #         w2 += frj * np.log(frj/fri[m])
            # w = (w1 + w2) / 2.0
            #count += 1
            print abs(qdf), diff


def merge_features_to_reviews(filename):
    u'''Merge features from filename to reviews'''
    # Read from review topic features file
    data = np.genfromtxt(filename, delimiter=',', dtype=None)
    nr = len(data)
    rv_dict = {}
    for i in range(nr):
        rv_dict[str(data[i][0])] = i
    # Read from database
    collection = Reviews(connection_dir=Settings.MONGO_CONNECTION_STRING, \
        database_name=Settings.YELP_DATABASE, collection_name=Settings.INPUT_COLLECTION).collection
    cursor = collection.find().batch_size(50)
    for features in cursor:
        rv_id = str(features['review_id'])
        if rv_id in rv_dict:
            i = rv_dict[rv_id]
            X = []
            X.append(float(features['num_sent']))
            X.append(float(features['sent_len']))
            X.append(float(features['num_token']))
            X.append(float(features['uniq_word_ratio']))
            X.append(float(features['pos_nn']))
            X.append(float(features['pos_adj']))
            X.append(float(features['pos_comp']))
            X.append(float(features['pos_v']))
            X.append(float(features['pos_rb']))
            X.append(float(features['pos_fw']))
            X.append(float(features['pos_cd']))
            for j in range(50):
                X.append(data[i][j+4])
            #X.append(data[i][54])
            #X.append(data[i][55])
            X.append(float(features['quality']))
            X.append(float(features['votes']))
            # Print features
            print ' '.join([str(x) for x in X])

def merge_json_files(fileleft, fileright, outfile):
    u'''Merges two json files'''
    leftlist = []
    with open(fileleft) as data_file:
        for line in data_file:
            feature = json.loads(line)
            leftlist.append(feature)

    count = 0
    start = time.time()
    with open(outfile, 'w') as _file:
        with open(fileright) as data_file:
            for line in data_file:
                features = json.loads(line)
                leftfeature = leftlist[count]
                features['pos_sen'] = leftfeature['pos_sen']
                features['neg_sen'] = leftfeature['neg_sen']
                count += 1
                _file.write(json.dumps(features, indent=1).replace('\n', ''))
                _file.write('\n')
                if count % 1000 == 0:
                    finish = time.time()
                    print ': Done ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
                         ' sec ~ ' + ('%.2f' % (count / (finish - start))) + '/sec'

if __name__ == '__main__':
    merge_json_files('res_tags_topics_sentiment_votes_1.json', \
        'res_tags_topics_votes_1.json', 'res_tags_topics_votes_nonzero.json')

    #merge_features_to_reviews('review_topic_features.txt')
    #review_topics_diff('./R_scripts/features_topics_only.txt')
    # get_quality()
    # topics_deviation_hypothesis()
    # rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    # rvs.load_all_data()
    # for review in rvs.cursor:
    #     useful_count = review['votes']['useful']
    #     votes = review['votes']['cool'] + review['votes']['funny'] + review['votes']['useful']
    #     rate = 0
    #     if votes >= 10:
    #         rate = float(useful_count) / float(votes)
    #         print votes, rate
    #num_reviews_businesses()
    #num_reviews_users()
    #num_reviews_votes()
    #num_reviews_helpful()
    #num_reviews_helpful_percentage()
    #num_received_votes_users()
    #num_received_helpful_votes_users()

