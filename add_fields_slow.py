#  Add extra fields to database
#    With users collection
#      1 star count, 2 stars count, 3 stars count, 4 stars count, 5 stars count
#      
#    With businesses collection
#      1 star count, 2 stars count, 3 stars count, 4 stars count, 5 stars count
#      
# This stupid program takes about 22h to finish this task in MacBook Air Autumn 2013 
# -> See add_fields_ver2 for very shorter version (about 1 min)
#
# Initialize first by 
# db.businesses.update({},{$set : {"stars_distribution" : {"one" : 0, "two" : 0, "three" : 0, "four" : 0, "five" : 0}}},false,true)
# db.users.update({},{$set : {"stars_distribution" : {"one" : 0, "two" : 0, "three" : 0, "four" : 0, "five" : 0}}},false,true)

import multiprocessing
import os
import time
import sys

from pymongo import MongoClient

from settings import Settings

def worker(identifier, collection, skip, count):
	done = 0
	start = time.time()
	reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.REVIEWS_COLLECTION]

	if str(collection) == "businesses":
		items_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.BUSINESSES_COLLECTION]
		item_str = "business_id"
	else:
		items_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.USERS_COLLECTION]
		item_str = "user_id"

	batch_size = 100
	for batch in range(0, count, batch_size):
		items_cursor = items_collection.find().skip(skip + batch).limit(batch_size)
		for item in items_cursor:
			item_id = item[item_str]
			rv_cursor = reviews_collection.find({item_str : item_id})
			dict = {}
			dict[1] = 0
			dict[2] = 0
			dict[3] = 0
			dict[4] = 0
			dict[5] = 0
			for rv in rv_cursor:
				stars = rv["stars"]
				dict[stars] += 1

			items_collection.update({item_str : item_id},{'$set' : {"stars_distribution" : {"one" : dict[1], "two" : dict[2], "three" : dict[3], "four" : dict[4], "five" : dict[5]}}},False,True)
			done += 1
			if done % 100 == 0:
				finish = time.time()
				print str(collection) + ' worker' + str(identifier) + ': Done ' + str(done) + ' out of ' + str(count) + ' in ' + ("%.2f" % (finish - start)) + ' sec ~ ' + ("%.2f" % (done / (finish - start))) + '/sec'
				sys.stdout.flush()

def main():
	start = time.time()
	businesses_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.BUSINESSES_COLLECTION]
	businesses_cursor = businesses_collection.find()
	businesses_count = businesses_cursor.count()

	users_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.USERS_COLLECTION]
	users_cursor = users_collection.find()
	users_count = users_cursor.count()

	workers = 4
	businesses_batch = businesses_count / workers
	users_batch = users_count / workers

	jobs = []
	for i in range(workers):
		bp = multiprocessing.Process(target=worker,args=((i+1), "businesses", i * businesses_batch, businesses_count / workers))
		up = multiprocessing.Process(target=worker,args=((i+1), "users", i * users_batch, users_count / workers))
		jobs.append(bp)
		bp.start()
		jobs.append(up)
		up.start()

	
	for j in jobs:
		j.join()
		print '%s.exitcode = %s' % (j.name, j.exitcode)

	finish = time.time()
	print "Duration in " + ("%.2f" % (finish - start)) 

if __name__ == '__main__':
	main()
