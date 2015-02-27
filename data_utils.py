#!env python
# -*- coding:utf-8 -*-

'''Load data utils
'''

from pymongo import MongoClient
from settings import Settings

class Reviews(object):
    u'''Holds a set of reviews'''

    def __init__(self, connection_dir=Settings.MONGO_CONNECTION_STRING, \
        database_name=Settings.YELP_DATABASE, collection_name=Settings.REVIEWS_COLLECTION):
        '''Init Reviews collection'''
        self.collection = MongoClient(connection_dir)[database_name][collection_name]
        self.cursor = None
        self.count = 0

    def load_all_data(self):
        '''Load review cursor'''
        self.cursor = self.collection.find()
        self.count = self.cursor.count()


class Businesses(object):
    u'''Holds a set of businesses'''
    def __init__(self, connection_dir=Settings.MONGO_CONNECTION_STRING, \
        database_name=Settings.YELP_DATABASE, collection_name=Settings.BUSINESSES_COLLECTION):
        '''Init Businesses collection'''
        self.collection = MongoClient(connection_dir)[database_name][collection_name]
        self.cursor = None
        self.count = 0

    def load_all_data(self):
        '''Load business cursor'''
        self.cursor = self.collection.find()
        self.count = self.cursor.count()


class Users(object):
    u'''Holds a set of users'''
    def __init__(self, connection_dir=Settings.MONGO_CONNECTION_STRING, \
        database_name=Settings.YELP_DATABASE, collection_name=Settings.USERS_COLLECTION):
        '''Init Users collection'''
        self.collection = MongoClient(connection_dir)[database_name][collection_name]
        self.cursor = None
        self.count = 0

    def load_all_data(self):
        '''Load user cursor'''
        self.cursor = self.collection.find()
        self.count = self.cursor.count()
