import multiprocessing
import os
import time
import sys

from pymongo import MongoClient
from settings import Settings

done = 758475

businesses_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.BUSINESSES_COLLECTION]
businesses_cursor = businesses_collection.find()
businesses_count = businesses_cursor.count()
businesses_cursor.batch_size(1000)

users_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.USERS_COLLECTION]
users_cursor = users_collection.find()
users_count = users_cursor.count()
users_cursor.batch_size(1000)

reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.REVIEWS_COLLECTION]
reviews_cursor = reviews_collection.find().skip(done-1)
reviews_count = reviews_cursor.count()

workers = 8
businesses_batch = businesses_count / workers
users_batch = users_count / workers

print str(businesses_count) + ' ' + str(businesses_batch) + ' ' + str(reviews_count / workers)

for item in reviews_cursor:
	print str(item["id"]) 
	if item["id"] != done:
		break
	done += 1