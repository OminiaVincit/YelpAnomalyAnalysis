import os
import time

from pymongo import MongoClient
import nltk

from settings import Settings

reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.REVIEWS_COLLECTION]
businesses_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.BUSINESSES_COLLECTION]
users_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.USERS_COLLECTION]

start = time.time()
reviews_cursor = reviews_collection.find()
revews_count = reviews_cursor.count()
reviews_cursor.batch_size(1000)

for review in reviews_cursor:
	business_id = review["business_id"]
	user_id = review["user_id"]
	review_stars = review["stars"]

finish = time.time()
duration = finish - start	
print duration

