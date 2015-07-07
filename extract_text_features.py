#!env python
# -*- coding:utf-8 -*-
'''Calculate for review text features score
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
from nltk.corpus.reader.wordnet import WordNetError

from settings import Settings
from data_utils import Reviews, GenCollection

from topic_predict import Predict

def text_features(text):
    u'''Extract text features for review content'''
    
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
    sentences = nltk.sent_tokenize(text.lower()) 
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
        try:
            sentence = sentence.decode('utf-8')
            tokens = nltk.word_tokenize(sentence)
            tagged_text = nltk.pos_tag(tokens)
            #tagged_text = english_postagger.tag(tokens)
            num_token += len(tokens)
            sent_len += len(sentence)
            
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
                    except WordNetError:
                        pass
                            
        except UnicodeEncodeError:
            pass
                
    pos_total = 0
    pos_nn = 0 # number of nouns
    pos_adj = 0 # number of adjective
    pos_comp = 0 # number of comparatives
    pos_v = 0 # number of verbs
    pos_rb = 0 # number of adverbs
    pos_fw = 0 # number of foreign words
    pos_cd = 0 # number of numbers
    unique_words = 0 # number of unique words
    pos_sen = 0
    neg_sen = 0
    positive_count = 0 # number of positive words
    negative_count = 0 # number of negative words

    for tag in tags_list:
        tf = tag[0]
        stat = tags_stat[tag]
        pos_total += stat
        if stat == 1 and tag != 'positive' and tag != 'negative':
             unique_words += 1
        if tag == 'positive':
            positive_count += stat
        if tag == 'negative':
            negative_count += stat

        if tag == 'JJR' or tag == 'JJS' \
            or tag == 'RBR' or tag == 'RBS':
                pos_comp += stat
        if tag == 'RB' or tag == 'RBR' or tag == 'RBS':
            pos_rb += stat
        if tag == 'CD':
            pos_cd += stat
        if tag == 'FW':
            pos_fw += stat
        if tf == 'N':
            pos_nn += stat
        elif tf == 'V':
            pos_v += stat
        elif tf == 'J':
            pos_adj += stat

    if pos_total != 0:
        # ratio
        pos_nn = float(pos_nn) / float(pos_total)
        pos_adj = float(pos_adj) / float(pos_total)
        pos_comp = float(pos_comp) / float(pos_total)
        pos_v = float(pos_v) / float(pos_total)
        pos_rb = float(pos_rb) / float(pos_total)
        pos_fw = float(pos_fw) / float(pos_total)
        pos_cd = float(pos_cd) / float(pos_total)
        pos_sen = float(positive_count) / float(pos_total)
        neg_sen = float(negative_count) / float(pos_total)
        unique_words = float(unique_words) / float(pos_total)
    
    if num_sent > 0:
        sent_len = float(sent_len) / float(num_sent)

    features['sent_len'] = sent_len
    features['num_sent'] = num_sent
    features['num_token'] = num_token
    features['uniq_word_ratio'] = unique_words
    features['pos_nn'] = pos_nn
    features['pos_adj'] = pos_adj
    features['pos_comp'] = pos_comp
    features['pos_v'] = pos_v
    features['pos_rb'] = pos_rb
    features['pos_fw'] = pos_fw
    features['pos_cd'] = pos_cd
    features['pos_sen'] = pos_sen
    features['neg_sen'] = neg_sen
    #features['words'] = words
    
    return features

def extract_text_features_job(collection_name, identifier, skip, count):
    u'''Extract text features'''
    idx = collection_name.find('_')
    name = collection_name[0:idx]
    # Debug time
    done = 0
    start = time.time()
    filename = name + '_text_features_' + str(identifier) + '.json'
    #english_postagger = POSTagger('./postagger/models/wsj-0-18-left3words-distsim.tagger', \
    #    './postagger/stanford-postagger.jar')
    rvs = Reviews(collection_name=collection_name)
    batch_size = 50

    with open(filename, 'w') as _file:
        for batch in range(0, count, batch_size):
            lm_size = min(batch_size, count - batch)
            rvs.cursor = rvs.collection.find().skip(skip + batch).limit(lm_size)
            for review in rvs.cursor:
                # print review['id'], review['text']
                if not review.get('text'):
                	continue
                features = text_features(review['text'])
                features['review_id'] = review['review_id']
                features['item_id'] = review['item_id']
                features['user_id'] = review['user_id']
                features['votes'] = review['votes']
                features['helpful'] = review['helpful']
                _file.write(json.dumps(features, indent=1).replace('\n', ''))
                _file.write('\n')

                done += 1
                if done % 100 == 0:
                    end = time.time()
                    print 'Worker' + str(identifier) + ': Done ' + str(done) + \
                        ' out of ' + str(count) + ' in ' + \
                        ('%.2f' % (end - start)) + ' sec ~ ' + \
                        ('%.2f' % (done / (end - start))) + '/sec'
                    sys.stdout.flush()

def extract_text_features(collection_name, workers=5):
    u'''Extract text features by collection'''
    rvs = Reviews(collection_name=collection_name)
    rvs.load_all_data()
    batch = rvs.count / workers
    jobs = []
    for i in range(workers):
        _ps = multiprocessing.Process(target=extract_text_features_job, \
            args=(collection_name, (i+1), i*batch, batch))
        jobs.append(_ps)
        _ps.start()

    # Add remainder parts to process
    _rp = multiprocessing.Process(target=extract_text_features_job, \
        args=(collection_name, (workers+1), \
        workers*batch, rvs.count-workers*batch))
    jobs.append(_rp)
    _rp.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)
            
def test():
    u'''Test function'''
    #extract_features_for_learning_from_mongo()
    
    s = 'I have a dream not for Jenny (my friend who is better than me). That is all I want to do.'
    s2 = 'Let there be no question: Alexions owns the best cheeseburger in the region and they have now for decades. \
        Try a burger on Italian bread. The service is flawlessly friendly, the food is amazing, \
        and the wings? Oh the wings... but it\'s still about the cheeseburger. \
        The atmosphere is inviting, but you can\'t eat atmosphere... so go right now. \
        Grab the car keys... you know you\'re hungry for an amazing cheeseburger, \
        maybe some wings, and a cold beer! Easily, hands down, the best bar and grill in Pittsburgh'

    features = text_features(s)
    print features
        
if __name__ == '__main__':
    #test()
    extract_text_features(Settings.MOVIES_REVIEWS_COLLECTION)

