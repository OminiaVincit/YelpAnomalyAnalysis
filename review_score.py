#!env python
# -*- coding:utf-8 -*-
'''Calculate score for first kth topics
'''

import os
import time
import sys
import multiprocessing
import nltk
import json
import numpy as np

from nltk.tag.stanford import POSTagger
from nltk.corpus import sentiwordnet as swn

from settings import Settings
from data_utils import Reviews, Businesses, Users

from topic_predict import Predict

NUMTOPICS = 50
EPSILON = 1e-7

def load_stopwords():
    u'''Load stop words'''
    stopwords = {}
    with open('stopwords3.txt', 'rU') as _file:
        for line in _file:
            stopwords[line.strip()] = 1
    return stopwords

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

def business_topic_worker(identifier, skip, count):
    u'''Add histogram of topics for each business by worker'''
    done = 0
    start = time.time()
    filename = 'res_businesses_topics_' + str(identifier) + '.json'
    businesses_collection = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION).collection
    reviews_collection = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION).collection
    batch_size = 50
    with open(filename, 'a') as _file:
        for batch in range(0, count, batch_size):
            lm_size = min(batch_size, count - batch)
            businesses_cursor = businesses_collection.find().skip(skip + batch).limit(lm_size)
            for business in businesses_cursor:
                tag = business
                tag['_id'] = None
                if business.get('topics') == None:
                    reviews_cursor = reviews_collection.find({'business_id' : business['business_id']}).batch_size(batch_size)
                    topics_stat = {}

                    for review in reviews_cursor:
                        rv_topics = business_analysis_topics(review)
                        # topics_field = ['global_topics_50', 'local_business_topics_5', \
                        #     'local_business_topics_10', 'local_business_topics_15', \
                        #     'local_business_topics_20']

                        for field, flist in rv_topics.iteritems():
                            if field not in topics_stat:
                                topics_stat[field] = {}
                            for pair in flist:
                                if pair[0] not in topics_stat[field]:
                                    topics_stat[field][pair[0]] = pair[1]
                                else:
                                    topics_stat[field][pair[0]] += pair[1]
                    rv_count = business['review_count']
                    for field, flist in topics_stat.iteritems():
                        sum_norm = 0.0
                        for idx, val in flist.iteritems():
                            sum_norm += val
                        if sum_norm > 0:
                            for key in topics_stat[field].iterkeys():
                                topics_stat[field][key] /= float(sum_norm)
                    tag['topics'] = topics_stat
                _file.write(json.dumps(tag, indent=1).replace('\n', '').replace('\"_id\": null,',''))
                _file.write('\n')
                done += 1
                if done % 100 == 0:
                    finish = time.time()
                    print ' Business topic worker' + str(identifier) + ': Done ' + str(done) + \
                        ' out of ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
                        ' sec ~ ' + ('%.2f' % (done / (finish - start))) + '/sec'
                    sys.stdout.flush()

def business_topic_parallel(num_workers):
    u'''Add histogram of topics for each business by multi-threads'''
    bss = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION)
    bss.load_all_data()
    workers = num_workers
    batch = bss.count / workers
    jobs = []
    for i in range(workers):
        _ps = multiprocessing.Process(target=business_topic_worker, \
            args=((i+1), i*batch, batch))
        jobs.append(_ps)
        _ps.start()

    # Add remainder parts to process
    _rp = multiprocessing.Process(target=business_topic_worker, args=((workers+1), \
        workers*batch, bss.count - workers*batch))
    jobs.append(_rp)
    _rp.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)

def business_analysis_topics(review):
    u'''Rearrange the topic distribution of review '''
    import numpy as np

    top_rs = {}
    topics_field = ['global_topics_50', 'local_business_topics_5', 'local_business_topics_10', \
        'local_business_topics_15', 'local_business_topics_20']

    for field in topics_field:
        if review.get(field):
            top_stat = review[field]
            num_top = len(top_stat)
            scores = np.zeros(num_top)

            for i in range(num_top):
                scores[i] = top_stat[i][1]

            scores *= np.var(scores)
            scores[scores < EPSILON] = 0.0
            sum_norm = np.sum(scores)
            if sum_norm > 0:
                scores /= sum_norm

            for i in range(num_top):
                top_stat[i][1] = scores[i]
            top_rs[field] = top_stat
    return top_rs

def user_topic_worker(identifier, skip, count):
    u'''Add histogram of topics for each user by worker'''
    done = 0
    start = time.time()
    filename = 'res_users_topics_' + str(identifier) + '.json'
    users_collection = Users(collection_name=Settings.RES_USERS_COLLECTION).collection
    reviews_collection = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION).collection
    batch_size = 50
    with open(filename, 'a') as _file:
        for batch in range(0, count, batch_size):
            lm_size = min(batch_size, count - batch)
            users_cursor = users_collection.find().skip(skip + batch).limit(lm_size)
            for user in users_cursor:
                tag = user
                tag['_id'] = None
                if user.get('topics') == None:
                    reviews_cursor = reviews_collection.find({'user_id' : user['user_id']}).batch_size(batch_size)
                    topics_stat = {}

                    for review in reviews_cursor:
                        rv_topics = user_analysis_topics(review)
                        for field, flist in rv_topics.iteritems():
                            if field not in topics_stat:
                                topics_stat[field] = {}
                            for pair in flist:
                                if pair[0] not in topics_stat[field]:
                                    topics_stat[field][pair[0]] = pair[1]
                                else:
                                    topics_stat[field][pair[0]] += pair[1]
                    rv_count = user['review_count']
                    for field, flist in topics_stat.iteritems():
                        sum_norm = 0.0
                        for idx, val in flist.iteritems():
                            sum_norm += val
                        if sum_norm > 0:
                            for key in topics_stat[field].iterkeys():
                                topics_stat[field][key] /= float(sum_norm)
                    tag['topics'] = topics_stat
                _file.write(json.dumps(tag, indent=1).replace('\n', '').replace('\"_id\": null,',''))
                _file.write('\n')
                done += 1
                if done % 100 == 0:
                    finish = time.time()
                    print ' User topic worker' + str(identifier) + ': Done ' + str(done) + \
                        ' out of ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
                        ' sec ~ ' + ('%.2f' % (done / (finish - start))) + '/sec'
                    sys.stdout.flush()

def user_topic_parallel(num_workers):
    u'''Add histogram of topics for each user by multi-threads'''
    uss = Users(collection_name=Settings.RES_USERS_COLLECTION)
    uss.load_all_data()
    workers = num_workers
    batch = uss.count / workers
    jobs = []
    for i in range(workers):
        _ps = multiprocessing.Process(target=user_topic_worker, \
            args=((i+1), i*batch, batch))
        jobs.append(_ps)
        _ps.start()

    # Add remainder parts to process
    _rp = multiprocessing.Process(target=user_topic_worker, args=((workers+1), \
        workers*batch, uss.count - workers*batch))
    jobs.append(_rp)
    _rp.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)

def user_analysis_topics(review):
    u'''Rearrange the topic distribution of review '''
    import numpy as np

    top_rs = {}
    topics_field = ['global_topics_50', 'local_user_topics_5', 'local_user_topics_10', \
        'local_user_topics_15', 'local_user_topics_20']

    for field in topics_field:
        if review.get(field):
            top_stat = review[field]
            num_top = len(top_stat)
            scores = np.zeros(num_top)

            for i in range(num_top):
                scores[i] = top_stat[i][1]

            scores *= np.var(scores)
            scores[scores < EPSILON] = 0.0
            sum_norm = np.sum(scores)
            if sum_norm > 0:
                scores /= sum_norm

            for i in range(num_top):
                top_stat[i][1] = scores[i]
            top_rs[field] = top_stat
    return top_rs

def text_features(text):
    u'''Extract text features for text'''
    
    # CC - Coordinating conjunction
    # CD - Cardinal number
    # DT - Determiner
    # EX - Existential there
    # FW - Foreign word
    # IN - Preposition or subordinating conjunction
    # JJ - Adjective
    # JJR - Adjective, comparative
    # JJS - Adjective, superlative
    # LS - List item marker
    # MD - Modal
    # NN - Noun, singular or mass
    # NNS - Noun, plural
    # NNP - Proper noun, singular
    # NNPS - Proper noun, plural
    # PDT - Predeterminer
    # POS - Possessive ending
    # PRP - Personal pronoun
    # PRP$ - Possessive pronoun (prolog version PRP-S)
    # RB - Adverb
    # RBR - Adverb, comparative
    # RBS - Adverb, superlative
    # RP - Particle
    # SYM - Symbol
    # TO - to
    # UH - Interjection
    # VB - Verb, base form
    # VBD - Verb, past tense
    # VBG - Verb, gerund or present participle
    # VBN - Verb, past participle
    # VBP - Verb, non-3rd person singular present
    # VBZ - Verb, 3rd person singular present
    # WDT - Wh-determiner
    # WP - Wh-pronoun
    # WP$ - Possessive wh-pronoun (prolog version WP-S)
    # WRB - Wh-adverb

    # num_token: total number of tokens
    # num_sent: total number of sentences
    # uniq_word_ratio: ratio of unique words
    # sent_len: averate sentence length
    # cap_ratio: ratio of capitalized sentences
    # pos_nn: ratio of nouns
    # pos_adj: ratio of adjectives
    # pos_comp: ratio of comparatives
    # pos_v: ratio of verbs
    # pos_rb: ratio of adverbs
    # pos_fw: ratio of foreign words
    # pos_cd: ratio of numbers
    # pos_pp: ratio of punctuation symbols
    # klall: conformity
    # pos_sen: sentiment, ratio of positive sentiment words (sentences)
    # neg_sen: ratio of negative sentiment words (sentences)
    # sentiment scores??? (rating / 5???)

    #english_postagger = POSTagger('../postagger/models/wsj-0-18-left3words-distsim.tagger', \
    #    '../postagger/stanford-postagger.jar')
    
    #pos_sent = open('./positive.txt').read()
    #positive_words=pos_sent.split('\n')
    
    #neg_sent = open('./negative.txt').read()
    #negative_words=neg_sent.split('\n')

    features = {}
    sentences = nltk.sent_tokenize(text) 
    num_sent = len(sentences)
    num_token = 0
    sent_len = 0
    words = []
    tags_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP',\
        'NNPS', 'PDT', 'POS', 'PRP', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD',\
        'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'positive', 'negative']

    tags_stat = {}
    for tag in tags_list:
        tags_stat[tag] = 0

    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        num_token += len(tokens)
        sent_len += len(sentence)
        #text_encoded = [word.encode('utf-8') for word in tokens]
        tagged_text = nltk.pos_tag(tokens)
        #tagged_text = english_postagger.tag(tokens)
        for word, tag in tagged_text:
            word = word.encode('utf-8')
            tag = tag.encode('utf-8')
            #words.append({'word': word, 'pos': tag})
            if tag in tags_list:
                tags_stat[tag] += 1
            
            # Check positive, negative score
            tf = tag[0].lower()
            if tf == 'j':
                tf = 'a'
            if tf in ['a', 'v', 'r', 'n']:
                try:
                    sen_ls = swn.senti_synsets(word, tf)
                    if len(sen_ls) != 0:
                        sen_score = sen_ls[0]
                        pos_score = sen_score.pos_score()
                        neg_score = sen_score.neg_score()
                        # obj_score = sen_score.obj_score()
                        if pos_score > neg_score:
                            tags_stat['positive'] += 1
                        if pos_score < neg_score:
                            tags_stat['negative'] += 1
                except UnicodeDecodeError:
                    pass

            #if word in positive_words:
            #    tags_stat['positive'] += 1
            #if word in negative_words:
            #    tags_stat['negative'] += 1
    
    pos_total = 0
    # pos_nn = 0 # number of nouns
    # pos_adj = 0 # number of adjective
    # pos_comp = 0 # number of comparatives
    # pos_v = 0 # number of verbs
    # pos_rb = 0 # number of adverbs
    # pos_fw = 0 # number of foreign words
    # pos_cd = 0 # number of numbers
    # unique_words = 0 # number of unique words
    pos_sen = 0
    neg_sen = 0
    positive_count = 0 # number of positive words
    negative_count = 0 # number of negative words

    for tag in tags_list:
        tf = tag[0]
        stat = tags_stat[tag]
        pos_total += stat
        # if stat == 1:
        #     unique_words += 1
        if tag == 'positive':
            positive_count += stat
        if tag == 'negative':
            negative_count += stat

        # if tag == 'JJR' or tag == 'JJS' \
        #     or tag == 'RBR' or tag == 'RBS':
        #         pos_comp += stat
        # if tag == 'RB' or tag == 'RBR' or tag == 'RBS':
        #     pos_rb += stat
        # if tag == 'CD':
        #     pos_cd += stat
        # if tag == 'FW':
        #     pos_fw += stat
        # if tf == 'N':
        #     pos_nn += stat
        # elif tf == 'V':
        #     pos_v += stat
        # elif tf == 'J':
        #     pos_adj += stat

    if pos_total != 0:
        # ratio
        # pos_nn = float(pos_nn) / float(pos_total)
        # pos_adj = float(pos_adj) / float(pos_total)
        # pos_comp = float(pos_comp) / float(pos_total)
        # pos_v = float(pos_v) / float(pos_total)
        # pos_rb = float(pos_rb) / float(pos_total)
        # pos_fw = float(pos_fw) / float(pos_total)
        # pos_cd = float(pos_cd) / float(pos_total)
        pos_sen = float(positive_count) / float(pos_total)
        neg_sen = float(negative_count) / float(pos_total)
        # unique_words = float(unique_words) / float(pos_total)
    
    # if num_sent > 0:
    #     sent_len = float(sent_len) / float(num_sent)

    # features['sent_len'] = sent_len
    # features['num_sent'] = num_sent
    # features['num_token'] = num_token
    # features['uniq_word_ratio'] = unique_words
    # features['pos_nn'] = pos_nn
    # features['pos_adj'] = pos_adj
    # features['pos_comp'] = pos_comp
    # features['pos_v'] = pos_v
    # features['pos_rb'] = pos_rb
    # features['pos_fw'] = pos_fw
    # features['pos_cd'] = pos_cd
    features['pos_sen'] = pos_sen
    features['neg_sen'] = neg_sen
    #features['words'] = words
    
    return features

def analysis_topics(review):
    u'''Rearrange the topic distribution of review '''
    import numpy as np

    top_rs = {}
    topics_field = ['global_topics_50', 'local_business_topics_5', 'local_business_topics_10', \
        'local_business_topics_15', 'local_business_topics_20', 'local_user_topics_5', \
        'local_user_topics_10', 'local_user_topics_15', 'local_user_topics_20']

    for field in topics_field:
        if review.get(field):
            top_stat = review[field]
            num_top = len(top_stat)
            scores = np.zeros(num_top)

            for i in range(num_top):
                scores[i] = top_stat[i][1]

            scores *= np.var(scores)
            scores[scores < EPSILON] = 0.0
            sum_norm = np.sum(scores)
            if sum_norm > 0:
                scores /= sum_norm

            for i in range(num_top):
                top_stat[i][1] = scores[i]
            top_rs[field] = top_stat
    return top_rs

def analysis_global_topics_50(review):
    u'''Rearrange the topic distribution of review '''
    field = 'global_topics_50'
    if review.get(field):
        top_stat = review[field]
        num_top = len(top_stat)
        scores = np.zeros(num_top)

        for i in range(num_top):
            scores[i] = top_stat[i][1]

        scores *= np.var(scores)
        scores[scores < EPSILON] = 0.0
        sum_norm = np.sum(scores)
        if sum_norm > 0:
            scores /= sum_norm

        for i in range(num_top):
            top_stat[i][1] = scores[i]
    return top_stat

def extract_text_features(filename):
    u'''Extract text features for each review with number of votes >= 10'''
    votes_threshold = 10
    reviews = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    reviews.load_all_data()
    count = 0

    # users and businesses collection
    users_collection = Users(collection_name=Settings.RES_USERS_COLLECTION).collection
    businesses_collection = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION).collection
    topics_field = ['business.global_topics_50', 'business.local_topics_5', 'business.local_topics_10', \
        'business.local_topics_15', 'business.local_topics_20', \
        'user.global_topics_50', 'user.local_topics_5', 'user.local_topics_10', \
        'user.local_topics_15', 'user.local_topics_20']

    with open(filename, 'w') as _file:
        for review in reviews.cursor:
            votes_count = review['votes']['funny'] + review['votes']['useful'] + review['votes']['cool']
            useful_count = review['votes']['useful']
            funny_count = review['votes']['funny']
            cool_count = review['votes']['cool']

            if votes_count >= votes_threshold:
                count += 1

                user = users_collection.find_one({'user_id' : review['user_id']})
                business = businesses_collection.find_one({'business_id' : review['business_id']})
                
                features = text_features(review['text'])
                #features = {}
                features['review_id'] = review['review_id']
                features['user_id'] = review['user_id']
                features['business_id'] = review['business_id']
                features['votes'] = votes_count
                features['useful'] = useful_count
                features['cool'] = cool_count
                features['funny'] = funny_count
                features['quality'] = float(useful_count) / float(votes_count)

                # rate deviation
                features['rate_deviation'] = (review['stars'] - business['average_stars'])/5.0

                # topics deviation
                top_rs = analysis_topics(review)

                for field in topics_field:
                    minor_major = field + '_anomaly_minor_major'
                    major_minor = field + '_anomaly_major_minor'
                    normaly = field + '_normally'

                    features[minor_major] = 0.0
                    features[major_minor] = 0.0
                    features[normaly] = 1.0

                    # find index in top_rs (after .)
                    dot = field.find('.')
                    top_h = field[:dot]
                    top_f = field[(dot+1):]

                    if top_rs.get(top_f):
                        
                        top_stat = top_rs[top_f]
                        num_top = len(top_stat)
                        if num_top > 0:
                            scores = {}

                            for top in top_stat:
                                scores[str(top[0])] = top[1]
                            sc_max = np.max(scores.values())
                            if sc_max > 0:
                                if top_h == 'business':
                                    bs_tops = business['topics'][top_f]
                                else:
                                    bs_tops = user['topics'][top_f]
                                # print 'bs_tops', bs_tops
                                # print 'top_stat', top_stat
                                bs_max = np.max(bs_tops.values())

                                anor_sc1 = {}
                                nor_sc = {}
                                for top in top_stat:
                                    # anormaly minor to major
                                    tp_idx = str(top[0])
                                    wr = top[1] / float(sc_max)
                                    fr = bs_tops[tp_idx]/float(bs_max)
                                    anor_sc1[tp_idx] = wr * (1.0 - fr)
                                    nor_sc[tp_idx] = wr * fr

                                anor_sc2 = {}
                                for key, value in bs_tops.iteritems():
                                    fr = value / float(bs_max)
                                    if key in scores:
                                        wr = scores[key] / float(sc_max)
                                    else:
                                        wr = 0.0
                                    anor_sc2[key] = fr * (1.0 - wr)

                                features[minor_major] = np.max(anor_sc1.values())
                                features[major_minor] = np.max(anor_sc2.values())
                                features[normaly] = np.max(nor_sc.values())

                            else:
                                # sc_max = 0, all weights = 0
                                features[minor_major] = 1.0
                                features[major_minor] = 1.0
                                features[normaly] = 0.0


                _file.write(json.dumps(features, indent=1).replace('\n', ''))
                _file.write('\n')
                print count

def extract_text_features_2(filename, votes_threshold=1):
    u'''Extract text features for each review with number of votes >= votes_threshold'''
    K = 50
    reviews = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    reviews.load_all_data()
    count = 0

    # users and businesses collection
    users_collection = Users(collection_name=Settings.RES_USERS_COLLECTION).collection
    businesses_collection = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION).collection
    
    start = time.time()

    with open(filename, 'w') as _file:
        for review in reviews.cursor:
            votes_count = review['votes']['funny'] + review['votes']['useful'] + review['votes']['cool']
            useful_count = review['votes']['useful']
            funny_count = review['votes']['funny']
            cool_count = review['votes']['cool']

            if votes_count >= votes_threshold:
                count += 1
                if count <= 46865:
                     continue

                #user = users_collection.find_one({'user_id' : review['user_id']})
                #business = businesses_collection.find_one({'business_id' : review['business_id']})
                
                features = text_features(review['text'])
                #features = {}
                # features['review_id'] = review['review_id']
                # features['user_id'] = review['user_id']
                # features['business_id'] = review['business_id']
                # features['votes'] = votes_count
                # features['useful'] = useful_count
                # features['cool'] = cool_count
                # features['funny'] = funny_count

                # rate deviation
                # features['rate_deviation'] = (review['stars'] - business['average_stars'])/5.0

                # topics deviation
                # top_stat = analysis_global_topics_50(review)
                # top_fr = {}
                # for i in range(K):
                #     top_fr[i] = 0
                # for top in top_stat:
                #     top_fr[int(top[0])] = top[1]
                # features['topics'] = top_fr

                _file.write(json.dumps(features, indent=1).replace('\n', ''))
                _file.write('\n')
                if count % 100 == 0:
                     finish = time.time()
                     print ': Done ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
                         ' sec ~ ' + ('%.2f' % (count / (finish - start))) + '/sec'

def extract_review_topics():
    u'''Extract review topics'''
    votes_threshold = 1
    K = 50
    users_collection = Users(collection_name=Settings.RES_USERS_COLLECTION).collection
    businesses_collection = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION).collection

    reviews = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    reviews.load_all_data()
    count = 0
    for review in reviews.cursor: 
        votes_count = review['votes']['funny'] + review['votes']['useful'] + review['votes']['cool']
        useful_count = review['votes']['useful']
        if votes_count >= votes_threshold:
            count += 1

            user = users_collection.find_one({'user_id' : review['user_id']})
            business = businesses_collection.find_one({'business_id' : review['business_id']})

            bs_tops = business['topics']['global_topics_50']
            us_tops = user['topics']['global_topics_50']

            rate = float(useful_count) / float(votes_count)
            top_stat = analysis_global_topics_50(review)
            top_fr = {}
            for i in range(K):
                top_fr[i] = 0
            for top in top_stat:
                top_fr[int(top[0])] = top[1]
            print_list = []
            print_list.append(review['review_id'])
            print_list.append(review['business_id'])
            print_list.append(review['user_id'])
            print_list.append(rate)
            for i in range(K):
                tmp = top_fr[i]
                if bs_tops[str(i)] != 0:
                    tmp = tmp / bs_tops[str(i)]
                print_list.append(tmp)

            # calculate topic KL divergence
            # bw, uw = 0, 0
            # for i in range(K):
            #     if top_fr[i] != 0:
            #         if bs_tops[str(i)] != 0:
            #             bw += top_fr[i] * np.log(float(top_fr[i])/float(bs_tops[str(i)]))
            #         if us_tops[str(i)] != 0:
            #             uw += top_fr[i] * np.log(float(top_fr[i])/float(us_tops[str(i)]))
            # print_list.append(bw)
            # print_list.append(uw)
            print(','.join([str(item) for item in print_list]))

def extract_features_for_learning_from_file(jsonfile=ur'../yelp_dataset_challenge_academic_dataset_20150209/restaurant_only/res_tags_topics_votes_1.json'):
    u'''Extract learning data from json file'''
    import json
    K = 50
    count = 0
    start = time.time()
    with open(jsonfile) as data_file:
        for line in data_file:
            features = json.loads(line)
            fr_list = []
            count += 1

            fr_list.append(float(features['num_sent']))
            fr_list.append(float(features['sent_len']))
            fr_list.append(float(features['num_token']))
            fr_list.append(float(features['uniq_word_ratio']))
            fr_list.append(float(features['pos_nn']))
            fr_list.append(float(features['pos_adj']))
            fr_list.append(float(features['pos_comp']))
            fr_list.append(float(features['pos_v']))
            fr_list.append(float(features['pos_rb']))
            fr_list.append(float(features['pos_fw']))
            fr_list.append(float(features['pos_cd']))
            
            fr_list.append(float(features['rate_deviation']))
            for i in range(K):
                fr_list.append(float(features['topics'][str(i)]))

            fr_list.append(float(features['useful']) / float(features['votes']))
            fr_list.append(float(features['votes']))

            print ' '.join([str(fr) for fr in fr_list])
            # if count % 100 == 0:
            #     finish = time.time()
            #     print ': Done ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
            #         ' sec ~ ' + ('%.2f' % (count / (finish - start))) + '/sec'

def extract_features_for_learning_from_mongo(connection_dir=Settings.MONGO_CONNECTION_STRING, \
        database_name=Settings.YELP_DATABASE, collection_name=Settings.FEATURES_COLLECTION):
    u'''Extract learning data from mongo database'''
    from pymongo import MongoClient
    collection = MongoClient(connection_dir)[database_name][collection_name]
    cursor = collection.find().batch_size(50)
    K = 50
    #count = 0
    start = time.time()
    votes_threshold = 10
    for features in cursor:
        if float(features['votes']) < votes_threshold:
            continue
        fr_list = []
        #count += 1

        # fr_list.append(float(features['num_sent']))
        # fr_list.append(float(features['sent_len']))
        # fr_list.append(float(features['num_token']))
        # fr_list.append(float(features['uniq_word_ratio']))
        # fr_list.append(float(features['pos_nn']))
        # fr_list.append(float(features['pos_adj']))
        # fr_list.append(float(features['pos_comp']))
        # fr_list.append(float(features['pos_v']))
        # fr_list.append(float(features['pos_rb']))
        # fr_list.append(float(features['pos_fw']))
        # fr_list.append(float(features['pos_cd']))
        # fr_list.append(float(features['pos_sen']))
        # fr_list.append(float(features['neg_sen']))
        for i in range(K):
           fr_list.append(float(features['topics'][str(i)]))

        fr_list.append(float(features['useful']) / float(features['votes']))
        fr_list.append(float(features['votes']))

        print ' '.join([str(fr) for fr in fr_list])
        # if count % 100 == 0:
        #     finish = time.time()
        #     print ': Done ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
        #         ' sec ~ ' + ('%.2f' % (count / (finish - start))) + '/sec'
    
if __name__ == '__main__':
    extract_features_for_learning_from_mongo()
    #business_topic_parallel(3)
    #user_topic_parallel(3)
    # Call normal function once more for update correctly
    # business_topic(3)
    #_test()
    # s = 'I have a dream not for Jenny (my friend who is better than me). That is all I want to do.'
    # s2 = 'Let there be no question: Alexions owns the best cheeseburger in the region and they have now for decades. \
    #     Try a burger on Italian bread. The service is flawlessly friendly, the food is amazing, \
    #     and the wings? Oh the wings... but it\'s still about the cheeseburger. \
    #     The atmosphere is inviting, but you can\'t eat atmosphere... so go right now. \
    #     Grab the car keys... you know you\'re hungry for an amazing cheeseburger, \
    #     maybe some wings, and a cold beer! Easily, hands down, the best bar and grill in Pittsburgh'

    # features = text_features(s)
    # print features
    # extract_text_features_2('res_tags_topics_sentiment_2.json')
    # extract_review_topics()


