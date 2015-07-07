#!env python
# -*- coding:utf-8 -*-
'''Extract all features for learning
'''

import numpy as np
import time

from settings import Settings
from data_utils import Reviews, GenCollection

def get_all_features(collection_text, collection_topic, votes_threshold=9):
    u'''Get all features for learning'''
    idx = collection_text.find('_')
    filename = collection_text[0:idx] + '_all_features.txt'
    rvs_top_collection = Reviews(collection_name=collection_topic).collection
    rvs_text_cursor = Reviews(collection_name=collection_text).collection.\
        find({'votes' : {'$gt' : votes_threshold}}).batch_size(50)
    total = rvs_text_cursor.count()
    count = 0
    with open(filename, 'w') as outfile:
        for rvs in rvs_text_cursor:
            count += 1
            print count, total
            if count <= 944840:
                continue
            review_id = rvs['review_id']
            rvs_top = rvs_top_collection.find_one({'review_id' : review_id})
            if rvs_top.get('topics'):
                line = []
                for i in range(Settings.NUMTOPICS):
                    line.append(rvs_top['topics'][str(i)])
                line.append(rvs['sent_len'])
                line.append(rvs['num_sent'])
                line.append(rvs['num_token'])
                line.append(rvs['uniq_word_ratio'])
                line.append(rvs['pos_nn'])
                line.append(rvs['pos_adj'])
                line.append(rvs['pos_comp'])
                line.append(rvs['pos_v'])
                line.append(rvs['pos_rb'])
                line.append(rvs['pos_fw'])
                line.append(rvs['pos_cd'])
                line.append(rvs['pos_sen'])
                line.append(rvs['neg_sen'])
                line.append(rvs['helpful'])
                line.append(rvs['votes'])
                line.append(float(rvs['helpful'])/float(rvs['votes']))
                strline = [str(item) for item in line]
                outfile.write(' '.join(strline))
                outfile.write('\n')

if __name__ == '__main__':
    #get_all_features(collection_text='yelp_text_features', collection_topic='yelp_LDA')
    #get_all_features(collection_text='tripadvisor_text_features', collection_topic='tripadvisor_LDA')
    get_all_features(collection_text='movies_text_features', collection_topic='movies_LDA')
