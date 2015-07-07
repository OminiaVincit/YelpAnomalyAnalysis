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
from data_utils import GenCollection

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

def make_model(collection_name):
    u'''Main function'''
    logging.basicConfig(format='%(asctime)s: %(levelname)s :%(message)s', level=logging.INFO)
    dictionary_path = 'models/dictionary_' + collection_name + '.dict'
    corpus_path = 'models/corpus_' + collection_name + '.lda-c'
    lda_num_topics = 50
    lda_model_path = 'models/lda_model_50_topics_' + collection_name + '.lda'

    corpus_collection = GenCollection(collection_name=collection_name)
    corpus_collection.load_all_data()
    corpus_cursor = corpus_collection.cursor

    dictionary = Dictionary(corpus_cursor, dictionary_path).build()
    Corpus(corpus_cursor, dictionary, corpus_path).serialize()
    Train.run(lda_model_path, corpus_path, lda_num_topics, dictionary)

def test():
    pass

def display():
    u'''Display hidden topics'''
    lda_model_path = '../Dataset/models/lda_model_50_topics_movies_corpus.lda'
    lda1 = LdaModel.load(lda_model_path)
    top_list = lda1.show_topics(num_topics=50, num_words=10, log=False, formatted=True)
    index = 0
    for top in top_list:
        index += 1
        print index,
        #scores = []
        #words = []
        topwords = top.split(' + ')
        for topword in topwords:
            member = topword.split('*')
            print member[1],
            #words.append(member[1])
            #scores.append(member[0])
        print ''
        

if __name__ == '__main__':
#    make_model(Settings.MOVIES_CORPUS_COLLECTION)
    display()


    








