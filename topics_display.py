#!env python
# -*- coding:utf-8 -*-
'''Display topics from trained LDA models
'''

import logging

from gensim.models import LdaModel
#from gensim import corpora

def display():
    u'''Display topics'''
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #dictionary_path = 'models/dictionary.dict'
    #corpus_path = 'models/corpus.lda-c'
    lda_num_topics = 50
    lda_model_path = 'models/lda_model_50_topics.lda'

    #dictionary = corpora.Dictionary.load(dictionary_path)
    #corpus = corpora.BleiCorpus(corpus_path)
    lda = LdaModel.load(lda_model_path)

    i = 0
    for topic in lda.show_topics(num_topics=lda_num_topics):
        print '#' + str(i) + ': ' + topic
        i += 1


if __name__ == '__main__':
    display()
