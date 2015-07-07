#!env python
# -*- coding:utf-8 -*-

'''Load data utils
'''

from pymongo import MongoClient
from settings import Settings

def load_stopwords():
    u'''Load stop words'''
    stopwords = {}
    with open('stopwords.txt', 'rU') as _file:
        for line in _file:
            stopwords[line.strip()] = 1
    return stopwords

class GenCollection(object):
    u'''Holds a general collection'''
    
    def __init__(self, connection_dir=Settings.MONGO_CONNECTION_STRING, \
                 database_name=Settings.DATABASE, collection_name=Settings.YELP_REVIEWS_COLLECTION):
        '''Init Reviews collection'''
        self.collection = MongoClient(connection_dir)[database_name][collection_name]
        self.cursor = None
        self.count = 0
    
    def load_all_data(self):
        '''Load cursor'''
        self.cursor = self.collection.find()
        self.count = self.cursor.count()

class Reviews(object):
    u'''Holds a set of reviews'''

    def __init__(self, connection_dir=Settings.MONGO_CONNECTION_STRING, \
        database_name=Settings.DATABASE, collection_name=Settings.YELP_REVIEWS_COLLECTION):
        '''Init Reviews collection'''
        self.collection = MongoClient(connection_dir)[database_name][collection_name]
        self.cursor = None
        self.count = 0

    def load_all_data(self):
        '''Load review cursor'''
        self.cursor = self.collection.find()
        self.count = self.cursor.count()

