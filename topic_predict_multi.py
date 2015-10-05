#!env python
# -*- coding:utf-8 -*-
'''Predict topic for new review
'''

import time
import multiprocessing
import sys
import json

import logging
import os.path
import numpy as np
import argparse

from gensim.models import LdaModel
from gensim import corpora
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

from settings import Settings
from data_utils import Reviews

EXCEPT_CHAR = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9',
u'!', u'"', u'#', u'$', u'%', u'&', u'\'', u'(', u')', u'-', u'=', u'^', u'~', 
u'¥', u'|', u'@', u'`', u'[', u'{', u';', u'+', u':', u'*', u']', u'}', 
u',', u'<', u'>', u'.', u'/', u'?', u'_']
EPSILON = 1e-5

class Predict(object):
    u'''Predict class'''

    def __init__(self, collection_name, lda_num_topics):
        idx = collection_name.find('_')
        name = collection_name[0:idx]
        dictionary_path = '../Dataset/models/dictionary_' + name + '_corpus.dict'
        lda_model_path = '../Dataset/models/lda_model_'+ str(lda_num_topics) + \
            '_topics_' + name + '_corpus.lda'
        u'''Initialize predict class'''
        self.dictionary = corpora.Dictionary.load(dictionary_path)
        self.lda = LdaModel.load(lda_model_path)
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self):
        u'''Load stopwords'''
        stopwords = {}
        with open('stopwords.txt', 'rU') as _file:
            for line in _file:
                stopwords[line.strip()] = 1
        return stopwords

    def extract_lemmatized_nouns(self, new_review):
        u'''Extract lemmatize nouns'''
        lem = WordNetLemmatizer()
        nouns = []
        sentences = nltk.sent_tokenize(new_review.lower())
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            #text = [word for word in tokens if word not in self.stopwords]
            tagged_text = nltk.pos_tag(tokens)

            for word, tag in tagged_text:
                if len(word) > 2 and word not in self.stopwords \
                    and tag in ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
                        save_flag = True
                        for ch in word:
                            if not save_flag:
                                break
                            if ch in EXCEPT_CHAR:
                                save_flag = False
                        if save_flag:
                            nouns.append(lem.lemmatize(word))
        return nouns

    def run(self, new_review):
        u'''Run model for predict topic'''
        nouns = self.extract_lemmatized_nouns(new_review)
        new_review_bow = self.dictionary.doc2bow(nouns)
        new_review_lda = self.lda[new_review_bow]

        return new_review_lda


def _test():
    u'''Main function'''
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    new_review = "It's like eating with a big Italian family. " \
                 "Great, authentic Italian food, good advice when asked, and terrific service. " \
                 "With a party of 9, last minute on a Saturday night, we were sat within 15 minutes. " \
                 "The owner chatted with our kids, and made us feel at home. " \
                 "They have meat-filled raviolis, which I can never find. " \
                 "The Fettuccine Alfredo was delicious. We had just about every dessert on the menu. " \
                 "The tiramisu had only a hint of coffee, the cannoli was not overly sweet, " \
                 "and they had this custard with wine that was so strangely good. " \
                 "It was an overall great experience!"

    new_review2 = 'I arrived around 10 am on a Saturday morning.  '\
        'I was greeted by the hostess,who was polite and friendly, '\
        'and I informed her that I was alone.  She had just arrived, '\
        'as well, and wasn\'t exactly sure what was going on yet, '\
        'so she asked (the manager?) where she should seat me.  '\
        'After receiving guidance, she brought me to a table towards '\
        'the back.  It had not yet been cleaned after the previous'\
        ' guests had dined.  So, she quickly got it cleared off, '\
        'grabbed a rag, and wiped it down.\n\nThe waitress came '\
        'rather quickly to take my drink order.  She was extremely '\
        'friendly.  I ordered coffee and water, which she went and '\
        'got while I looked over the menu.  She returned, and took '\
        'my order.  I ordered the Cinnamon Swirl French Toast Special'\
        ' - Three slices of French toast made with Pittsburgh\'s '\
        'own Jenny Lee® Cinnamon Swirl, two farm-fresh eggs, with '\
        'bacon or sausage (I chose sausage), and your choice of potato '\
        '(I went with the cheesy hash brown casserole).  I also added '\
        'an orange juice.  She went and put my order in, while I waited, '\
        'and came back with it after not too long.   The eggs were cooked '\
        'exactly how I wanted them, the cheesy hash brown casserole '\
        'and the french toast were both delicious.  I also enjoyed the '\
        'sausage which was pretty typical.\n\nKings Family Restaurant '\
        'featured a very friendly staff, great prices, and tasty food.  '\
        'I am pleased and will definitely come back again.'

    new_review3 = 'Sashimi here is wonderful and melt in your mouth.' \
        'I often get the sashimi tasting here.'\
        'I do not think the rolls here are that great.'\
        'I think the udon noodle sets and other items on the menu are very good but it changes from time to time.'\
        'If you have a group over 4 then this may not be a place for you. Agashi tofu was great here.'

    new_review4 = 'Khong lien quan'
    predict = Predict(collection_name=Settings.YELP_REVIEWS_COLLECTION)
    topics = predict.run(str(new_review3).decode('utf-8'))
    print topics
    top_arr = np.zeros(NUMTOPICS)
    
    for topic in topics:
        top_arr[topic[0]] = topic[1]
    top_arr *= np.var(top_arr)
    top_arr[top_arr < EPSILON] = 0
    total = np.sum(top_arr)
    if total > 0:
        top_arr /= total
    print 'Top arr: ', top_arr

    top_dict = {x:0 for x in range(NUMTOPICS)}
    for i in range(NUMTOPICS):
        top_dict[i] = top_arr[i]
    print 'Top dic: ', top_dict

def process(topics, lda_num_topics):
    u'''Process topics to array'''
    top_arr = np.zeros(lda_num_topics)
    
    for topic in topics:
        top_arr[topic[0]] = topic[1]
    top_arr *= np.var(top_arr)
    top_arr[top_arr < EPSILON] = 0
    total = np.sum(top_arr)
    if total > 0:
        top_arr /= total
    top_dict = {x:0 for x in range(lda_num_topics)}
    for i in range(lda_num_topics):
        top_dict[i] = top_arr[i]
    return top_dict

def predict_worker_to_file(collection_name, identifier, skip, count, lda_num_topics):
    u'''Predict review worker for multi-processing'''
    done = 0
    start = time.time()
    global_predict = Predict(collection_name=collection_name, lda_num_topics=lda_num_topics)
    filename = collection_name + '_topics_' + str(lda_num_topics) + '_' \
        + str(identifier) + '.json'
    reviews = Reviews(collection_name=collection_name)
    batch_size = 50
    
    with open(filename, 'w') as _file:
        for batch in range(0, count, batch_size):
            lm_size = min(batch_size, count - batch)
            reviews_cursor = reviews.collection.find().skip(skip + batch).limit(lm_size)
            for review in reviews_cursor:
                tag = {}
                tag['review_id'] = review['review_id']
                tag['user_id'] = review['user_id']
                tag['item_id'] = review['item_id']
                tag['votes'] = review['votes']
                tag['helpful'] = review['helpful']
                tag['topics'] = process(global_predict.run(review['text']), lda_num_topics=lda_num_topics)

                _file.write(json.dumps(tag, indent=1).replace('\n', ''))
                _file.write('\n')
                done += 1
                if done % 100 == 0:
                    finish = time.time()
                    print ' Predict review worker' + str(identifier) + ': Done ' + str(done) + \
                        ' out of ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
                        ' sec ~ ' + ('%.2f' % (done / (finish - start))) + '/sec'
                    sys.stdout.flush()

def predict_worker_to_file_multi(collection_name, identifier, skip, count):
    u'''Predict review worker for multi-processing'''
    done = 0
    start = time.time()
    global_predict_51 = Predict(collection_name=collection_name, lda_num_topics=51)
    global_predict_50 = Predict(collection_name=collection_name, lda_num_topics=50)
    # global_predict_64 = Predict(collection_name=collection_name, lda_num_topics=64)
    # global_predict_100 = Predict(collection_name=collection_name, lda_num_topics=100)
    # global_predict_144 = Predict(collection_name=collection_name, lda_num_topics=144)
    # global_predict_196 = Predict(collection_name=collection_name, lda_num_topics=196)
    # global_predict_256 = Predict(collection_name=collection_name, lda_num_topics=256)

    filename = collection_name + '_topics_distribution_' + str(identifier) + '.json'
    reviews = Reviews(collection_name=collection_name)
    batch_size = 50
    
    with open(filename, 'w') as _file:
        for batch in range(0, count, batch_size):
            lm_size = min(batch_size, count - batch)
            reviews_cursor = reviews.collection.find().skip(skip + batch).limit(lm_size)
            for review in reviews_cursor:
                tag = {}
                tag['review_id'] = review['review_id']
                tag['user_id'] = review['user_id']
                tag['item_id'] = review['item_id']
                tag['votes'] = review['votes']
                tag['helpful'] = review['helpful']
                tag['topics_51'] = process(global_predict_51.run(review['text']), lda_num_topics=51)
                tag['topics_50'] = process(global_predict_50.run(review['text']), lda_num_topics=50)

                # tag['topics_64'] = process(global_predict_64.run(review['text']), lda_num_topics=64)
                # tag['topics_100'] = process(global_predict_100.run(review['text']), lda_num_topics=100)
                # tag['topics_144'] = process(global_predict_144.run(review['text']), lda_num_topics=144)
                # tag['topics_196'] = process(global_predict_196.run(review['text']), lda_num_topics=196)
                # tag['topics_256'] = process(global_predict_256.run(review['text']), lda_num_topics=256)

                _file.write(json.dumps(tag, indent=1).replace('\n', ''))
                _file.write('\n')
                done += 1
                if done % 100 == 0:
                    finish = time.time()
                    print ' Predict review worker' + str(identifier) + ': Done ' + str(done) + \
                        ' out of ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
                        ' sec ~ ' + ('%.2f' % (done / (finish - start))) + '/sec'
                    sys.stdout.flush()

def extract_topics_features(collection_name, lda_num_topics):
    u'''Main function'''
    rvs = Reviews(collection_name=collection_name)
    rvs.load_all_data()
    workers = 3
    batch = rvs.count / workers
    jobs = []
    for i in range(workers):
        _ps = multiprocessing.Process(target=predict_worker_to_file, \
            args=(collection_name, (i+1), i*batch, batch, lda_num_topics))
        jobs.append(_ps)
        _ps.start()

    # Add remainder parts to process
    _rp = multiprocessing.Process(target=predict_worker_to_file, args=(collection_name, (workers+1), \
        workers*batch, rvs.count - workers*batch, lda_num_topics))
    jobs.append(_rp)
    _rp.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)

def extract_topics_features_multi(collection_name):
    u'''Main function'''
    rvs = Reviews(collection_name=collection_name)
    rvs.load_all_data()
    workers = 3
    batch = rvs.count / workers
    jobs = []
    for i in range(workers):
        _ps = multiprocessing.Process(target=predict_worker_to_file_multi, \
            args=(collection_name, (i+1), i*batch, batch))
        jobs.append(_ps)
        _ps.start()

    # Add remainder parts to process
    _rp = multiprocessing.Process(target=predict_worker_to_file_multi, args=(collection_name, (workers+1), \
        workers*batch, rvs.count - workers*batch))
    jobs.append(_rp)
    _rp.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--num_topics', type=int, default=64)
    parser.add_argument('--site', type=str, default='yelp',
                      choices=['yelp', 'tripadvisor', 'movies'])
    args = parser.parse_args()

    if args.site == 'tripadvisor':
        extract_topics_features_multi(Settings.TRIPADVISOR_REVIEWS_COLLECTION)
    elif args.site == 'yelp':
        extract_topics_features_multi(Settings.YELP_REVIEWS_COLLECTION)
    else:
        pass
        #extract_topics_features_multi(Settings.MOVIES_REVIEWS_COLLECTION)
#_test()
