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

from gensim.models import LdaModel
from gensim import corpora
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

from settings import Settings
from data_utils import Reviews

NUMTOPICS = 50

class Predict(object):
    u'''Predict class'''

    def __init__(self, dictionary_path = '../models/dictionary.dict',\
        lda_model_path = '../models/lda_model_50_topics.lda'):
        u'''Initialize predict class'''
        self.dictionary = corpora.Dictionary.load(dictionary_path)
        self.lda = LdaModel.load(lda_model_path)
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self):
        u'''Load stopwords'''
        stopwords = {}
        with open('stopwords2.txt', 'rU') as _file:
            for line in _file:
                stopwords[line.strip()] = 1
        return stopwords

    def extract_lemmatized_nouns(self, new_review):
        u'''Extract lemmatize nouns'''
        words = []
        sentences = nltk.sent_tokenize(new_review.lower())
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            #text = [word for word in tokens if word not in self.stopwords]
            tagged_text = nltk.pos_tag(tokens)

            for word, tag in tagged_text:
                if word not in self.stopwords:
                    words.append({'word': word, 'pos': tag})

        lem = WordNetLemmatizer()
        nouns = []
        for word in words:
            if word['pos'] in ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
                nouns.append(lem.lemmatize(word['word']))

        return nouns

    def run(self, new_review):
        u'''Run model for predict topic'''
        nouns = self.extract_lemmatized_nouns(new_review)
        new_review_bow = self.dictionary.doc2bow(nouns)
        new_review_lda = self.lda[new_review_bow]

        return new_review_lda


def _test():
    u'''Main function'''
    import numpy as np

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
    predict = Predict()
    topics = predict.run(str(new_review3).decode('utf-8'))
    print topics

    top_dict = {x:0 for x in range(NUMTOPICS)}
    for topic in topics:
        top_dict[topic[0]] = topic[1]
    print 'Top dict list: ', top_dict
    top_arr = np.zeros(NUMTOPICS)
    for i in range(NUMTOPICS):
        top_arr[i] = top_dict[i]
    print 'Top array: ', top_arr
    top_arr *= np.var(top_arr)
    top_arr = sorted(top_arr, reverse=True)
    print 'Processed array ', top_arr
    #top_arr[top_arr < EPSILON] = 0

def predict_worker(identifier, skip, count):
    u'''Predict review worker for multi-processing'''
    done = 0
    start = time.time()
    predict = Predict()
    reviews = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    batch_size = 100
    for batch in range(0, count, batch_size):
        lm_size = min(batch_size, count - batch)
        reviews_cursor = reviews.collection.find().skip(skip + batch).limit(lm_size)
        for review in reviews_cursor:
            if review.get('global_topics_50') == None:
                topics = predict.run(review['text'])
                top_dict = {x:0 for x in range(NUMTOPICS)}
                for tp in topics:
                    top_dict[tp[0]] = tp[1]
                reviews.collection.update({'_id' : review['_id']}, \
                    {'$set' : {'global_topics_50' : {\
                        '1': top_dict[0], '2': top_dict[1], '3': top_dict[2], \
                        '4': top_dict[3], '5': top_dict[4], '6': top_dict[5], \
                        '7': top_dict[6], '8': top_dict[7], '9': top_dict[8], \
                        '10': top_dict[9], '11': top_dict[10], '12': top_dict[11], \
                        '13': top_dict[12], '14': top_dict[13], '15': top_dict[14], \
                        '16': top_dict[15], '17': top_dict[16], '18': top_dict[17], \
                        '19': top_dict[18], '20': top_dict[19], '21': top_dict[20], \
                        '22': top_dict[21], '23': top_dict[22], '24': top_dict[23], \
                        '25': top_dict[24], '26': top_dict[25], '27': top_dict[26], \
                        '28': top_dict[27], '29': top_dict[28], '30': top_dict[29], \
                        '31': top_dict[30], '32': top_dict[31], '33': top_dict[32], \
                        '34': top_dict[33], '35': top_dict[34], '36': top_dict[35], \
                        '37': top_dict[36], '38': top_dict[37], '39': top_dict[38], \
                        '40': top_dict[39], '41': top_dict[40], '42': top_dict[41], \
                        '43': top_dict[42], '44': top_dict[43], '45': top_dict[44], \
                        '46': top_dict[45], '47': top_dict[46], '48': top_dict[47], \
                        '49': top_dict[48], '50': top_dict[49]\
                    }}}, \
                    False, True)
            done += 1
            if done % 100 == 0:
                finish = time.time()
                print ' Predict review worker' + str(identifier) + ': Done ' + str(done) + \
                    ' out of ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
                    ' sec ~ ' + ('%.2f' % (done / (finish - start))) + '/sec'
                sys.stdout.flush()

def predict_worker_to_file(identifier, skip, count):
    u'''Predict review worker for multi-processing'''
    done = 0
    start = time.time()
    global_predict = Predict()
    filename = 'res_reviews_topics_' + str(identifier) + '.json'
    reviews = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    batch_size = 50
    lda_num_topics_list = [5, 10, 15, 20]

    with open(filename, 'a') as _file:
        for batch in range(0, count, batch_size):
            lm_size = min(batch_size, count - batch)
            reviews_cursor = reviews.collection.find().skip(skip + batch).limit(lm_size)
            for review in reviews_cursor:
                tag = {}
                tag = review
                tag['_id'] = None
                tag['global_topics_50'] = global_predict.run(review['text'])
                
                business_id = review['business_id']
                user_id = review['user_id']

                for lda_num_topics in lda_num_topics_list:
                    business_topics = 'business_local_topics_' + str(lda_num_topics)
                    lda_business_model_path = '../models/' + 'lda_model_' + str(lda_num_topics) + \
                        '_topics_business_' + str(business_id) + '.lda'
                    business_dictionary_path = '../models/dictionary_business_' + str(business_id) + '.dict'

                    # Check if model and distionary existed or not
                    if os.path.isfile(lda_business_model_path) and \
                        os.path.isfile(business_dictionary_path):
                        # Run predict for business model
                        local_predict = Predict(dictionary_path=business_dictionary_path, \
                            lda_model_path=lda_business_model_path)
                        local_id = 'local_business_topics_' + str(lda_num_topics)
                        tag[local_id] = local_predict.run(review['text'])
                        # print review['id'], business_id

                    user_topics = 'user_local_topics_' + str(lda_num_topics)
                    lda_user_model_path = '../models/' + 'lda_model_' + str(lda_num_topics) + \
                        '_topics_user_' + str(user_id) + '.lda'
                    user_dictionary_path = '../models/dictionary_user_' + str(user_id) + '.dict'

                    # Check if model and distionary existed or not
                    if os.path.isfile(lda_user_model_path) and \
                        os.path.isfile(user_dictionary_path):
                        # Run predict for user model
                        local_predict = Predict(dictionary_path=user_dictionary_path, \
                            lda_model_path=lda_user_model_path)
                        local_id = 'local_user_topics_' + str(lda_num_topics)
                        tag[local_id] = local_predict.run(review['text']) 
                        # print review['id'], user_id
                
                _file.write(json.dumps(tag, indent=1).replace('\n', '').replace('\"_id\": null,',''))
                _file.write('\n')
                done += 1
                if done % 100 == 0:
                    finish = time.time()
                    print ' Predict review worker' + str(identifier) + ': Done ' + str(done) + \
                        ' out of ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
                        ' sec ~ ' + ('%.2f' % (done / (finish - start))) + '/sec'
                    sys.stdout.flush()

def main():
    u'''Main function'''
    rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    rvs.load_all_data()
    workers = 3
    batch = rvs.count / workers
    jobs = []
    for i in range(workers):
        _ps = multiprocessing.Process(target=predict_worker_to_file, \
            args=((i+1), i*batch, batch))
        jobs.append(_ps)
        _ps.start()

    # Add remainder parts to process
    _rp = multiprocessing.Process(target=predict_worker_to_file, args=((workers+1), \
        workers*batch, rvs.count - workers*batch))
    jobs.append(_rp)
    _rp.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)

if __name__ == '__main__':
    #main()
    _test()
