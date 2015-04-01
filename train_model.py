#!env python
# -*- coding:utf-8 -*-
'''Feeds the reviews corpus to the gensim LDA model
'''

import logging
import gensim
from gensim.corpora import BleiCorpus
from gensim.models import LdaModel
from gensim import corpora
import os.path

from settings import Settings
from data_utils import CorpusCollection, Businesses, Users

class Corpus(object):
    u'''Corpus class'''
    def __init__(self, cursor, corpus_dictionary, corpus_path):
        u'''Initialize corpus'''
        self.cursor = cursor
        self.corpus_dictionary = corpus_dictionary
        self.corpus_path = corpus_path

    def __iter__(self):
        u'''Corpus iterator'''
        self.cursor.rewind()
        for corpus in self.cursor:
            yield self.corpus_dictionary.doc2bow(corpus['words'])

    def serialize(self):
        u'''Serialize corpus'''
        BleiCorpus.serialize(self.corpus_path, self, \
            id2word=self.corpus_dictionary)
        return self


class Dictionary(object):
    u'''Dictionary class'''
    def __init__(self, cursor, dictionary_path):
        u'''Initialize Dictionary class'''
        self.cursor = cursor
        self.dictionary_path = dictionary_path

    def build(self):
        u'''Build dictionary'''
        self.cursor.rewind()
        dictionary = corpora.Dictionary(review['words'] \
            for review in self.cursor)
        dictionary.filter_extremes(keep_n=10000)
        dictionary.compactify()
        corpora.Dictionary.save(dictionary, self.dictionary_path)

        return dictionary


class Train:
    u'''Training class'''
    def __init__(self):
        pass

    @staticmethod
    def run(lda_model_path, corpus_path, num_topics, id2word):
        u'''Training to create LDA model'''
        corpus = corpora.BleiCorpus(corpus_path)
        lda = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=id2word)
        lda.save(lda_model_path)

        return lda

def get_lda_model_business(business_id, lda_num_topics_list):
    u'''Get lda model for each business'''
    logging.basicConfig(format='%(asctime)s: %(levelname)s :%(message)s', level=logging.INFO)
    dictionary_path = 'models/dictionary_business_' + str(business_id) + '.dict'
    corpus_path = 'models/corpus_business_' + str(business_id) + '.lda-c'
    corpus_collection = CorpusCollection(collection_name=Settings.RES_CORPUS_COLLECTION).collection
    corpus_cursor = corpus_collection.find({'business_id' : business_id}).batch_size(50)

    lda_model_path_5 = 'models/' + 'lda_model_5' + \
                '_topics_business_' + str(business_id) + '.lda'
    lda_model_path_10 = 'models/' + 'lda_model_10' + \
            '_topics_business_' + str(business_id) + '.lda'
    lda_model_path_15 = 'models/' + 'lda_model_15' + \
            '_topics_business_' + str(business_id) + '.lda'
    lda_model_path_20 = 'models/' + 'lda_model_20' + \
            '_topics_business_' + str(business_id) + '.lda'

    if os.path.isfile(lda_model_path_5) and os.path.isfile(lda_model_path_10) \
        and os.path.isfile(lda_model_path_15) and os.path.isfile(lda_model_path_20):
        pass
    elif corpus_cursor.count() > 0:
        if not os.path.isfile(dictionary_path):
            dictionary = Dictionary(corpus_cursor, dictionary_path).build()
        else:
            dictionary = corpora.Dictionary.load(dictionary_path)
        if not os.path.isfile(corpus_path):
            Corpus(corpus_cursor, dictionary, corpus_path).serialize()
        for lda_num_topics in lda_num_topics_list:
            lda_model_path = 'models/' + 'lda_model_' + str(lda_num_topics) + \
                '_topics_business_' + str(business_id) + '.lda'
            if not os.path.isfile(lda_model_path):
                Train.run(lda_model_path, corpus_path, lda_num_topics, dictionary)

def get_lda_model_user(user_id, lda_num_topics_list):
    u'''Get lda model for each user'''
    logging.basicConfig(format='%(asctime)s: %(levelname)s :%(message)s', level=logging.INFO)
    dictionary_path = 'models/dictionary_user_' + str(user_id) + '.dict'
    corpus_path = 'models/corpus_user_' + str(user_id) + '.lda-c'
    corpus_collection = CorpusCollection(collection_name=Settings.RES_CORPUS_COLLECTION).collection
    corpus_cursor = corpus_collection.find({'user_id' : user_id}).batch_size(50)
    
    lda_model_path_5 = 'models/' + 'lda_model_5' + \
                '_topics_user_' + str(user_id) + '.lda'
    lda_model_path_10 = 'models/' + 'lda_model_10' + \
            '_topics_user_' + str(user_id) + '.lda'
    lda_model_path_15 = 'models/' + 'lda_model_15' + \
            '_topics_user_' + str(user_id) + '.lda'
    lda_model_path_20 = 'models/' + 'lda_model_20' + \
            '_topics_user_' + str(user_id) + '.lda'

    if os.path.isfile(lda_model_path_5) and os.path.isfile(lda_model_path_10) \
        and os.path.isfile(lda_model_path_15) and os.path.isfile(lda_model_path_20):
        pass
    elif corpus_cursor.count() > 0:
        if not os.path.isfile(dictionary_path):
            dictionary = Dictionary(corpus_cursor, dictionary_path).build()
        else:
            dictionary = corpora.Dictionary.load(dictionary_path)
        if not os.path.isfile(corpus_path):
            Corpus(corpus_cursor, dictionary, corpus_path).serialize()
        for lda_num_topics in lda_num_topics_list:
            lda_model_path = 'models/' + 'lda_model_' + str(lda_num_topics) + \
                '_topics_user_' + str(user_id) + '.lda'
            if not os.path.isfile(lda_model_path):
                Train.run(lda_model_path, corpus_path, lda_num_topics, dictionary)

def main():
    u'''Main function'''
    logging.basicConfig(format='%(asctime)s: %(levelname)s :%(message)s', level=logging.INFO)
    dictionary_path = 'models/dictionary.dict'
    corpus_path = 'models/corpus.lda-c'
    lda_num_topics = 50
    lda_model_path = 'models/lda_model_50_topics.lda'

    corpus_collection = CorpusCollection(collection_name=Settings.RES_CORPUS_COLLECTION)
    corpus_collection.load_all_data()
    corpus_cursor = corpus_collection.cursor

    dictionary = Dictionary(corpus_cursor, dictionary_path).build()
    Corpus(corpus_cursor, dictionary, corpus_path).serialize()
    Train.run(lda_model_path, corpus_path, lda_num_topics, dictionary)

def test():
    get_lda_model_business('P1fJb2WQ1mXoiudj8UE44w', 10)
    get_lda_model_user('DfJ8B8T1fgNA6_ZT-aeUlQ', 10)

def display():
    for i in [10]:
        print i
        business_id = 'P1fJb2WQ1mXoiudj8UE44w'
        lda_model_path = 'models/' + 'lda_model_' + str(i) + \
            '_topics_business_' + str(business_id) + '.lda'
        lda1 = LdaModel.load(lda_model_path)
        print lda1.show_topics(num_topics=i, num_words=10, log=False, formatted=True)
        user_id = 'DfJ8B8T1fgNA6_ZT-aeUlQ'
        lda_model_path_2 = 'models/' + 'lda_model_' + str(i) + \
            '_topics_user_' + str(user_id) + '.lda'
        lda2 = LdaModel.load(lda_model_path_2)
        print lda2.show_topics(num_topics=i, num_words=10, log=False, formatted=True)

def get_business_local_models(lda_num_topics_list):
    u'''Get local model for each business'''
    u'''For only case with review count >= 50'''
    business_threshold = 30

    business_collection = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION).collection
    business_cursor = business_collection.find({'review_count': {'$gt': business_threshold}}).batch_size(50)
    count = 0
    for business in business_cursor:
        count += 1
        print count, business['id'], business['name']
        get_lda_model_business(business['business_id'], lda_num_topics_list)
        
    
def get_user_local_models(lda_num_topics_list):
    u'''Get local model for each user'''
    u'''For only case with review count >= 50'''
    user_threshold = 30
    
    user_collection = Users(collection_name=Settings.RES_USERS_COLLECTION).collection
    user_cursor = user_collection.find({'review_count': {'$gt': user_threshold}}).batch_size(50)
    count = 0
    for user in user_cursor:
        count += 1
        print count, user['id'], user['name']
        get_lda_model_user(user['user_id'], lda_num_topics_list)

if __name__ == '__main__':
    get_user_local_models([5, 10, 15, 20])
    get_business_local_models([5, 10, 15, 20])

    #test()
    #display()
    #main()

    








