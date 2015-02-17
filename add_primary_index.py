# This task takes about 15 mins
# Before run this scrip, do the following commands for update fields at database
# db.users.update({},{$set : {"id" : 0}}, false, true)
# db.businesses.update({},{$set : {"id" : 0}}, false, true)
# db.reviews.update({},{$set : {"id" : 0, "p_user_id" : 0, "p_business_id" : 0}}, false, true)

import multiprocessing
import os
import time
import sys

from pymongo import MongoClient
from settings import Settings

bs_dict = {}
us_dict = {}

businesses_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.BUSINESSES_COLLECTION]
businesses_cursor = businesses_collection.find()
businesses_count = businesses_cursor.count()
businesses_cursor.batch_size(1000)

users_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.USERS_COLLECTION]
users_cursor = users_collection.find()
users_count = users_cursor.count()
users_cursor.batch_size(1000)

reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.REVIEWS_COLLECTION]
reviews_cursor = reviews_collection.find()
reviews_count = reviews_cursor.count()

def worker(identifier, collection, skip, count):
	done = 0
	start = time.time()
	
	if str(collection) == "businesses":
		items_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.BUSINESSES_COLLECTION]
		#item_str = "business_id"
	elif str(collection) == "users":
		items_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.USERS_COLLECTION]
		#item_str = "user_id"
	else:
		items_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.REVIEWS_COLLECTION]
		#item_str = "review_id"

	batch_size = 500
	for batch in range(0, count, batch_size):
		items_cursor = items_collection.find().skip(skip + batch).limit(batch_size)
		counter = skip + batch + 1
		for item in items_cursor:
			items_collection.update({"_id" : item["_id"]},{'$set' : {"id":counter}}, False, True)
			#dict[ str(item[ item_str ]) ] = counter
			counter += 1
			done += 1
			if done % 1000 == 0:
				finish = time.time()
				print str(collection) + ' worker' + str(identifier) + ': Done ' + str(done) + ' out of ' + str(count) + ' in ' + ("%.2f" % (finish - start)) + ' sec ~ ' + ("%.2f" % (done / (finish - start))) + '/sec'
				sys.stdout.flush()

def review_worker(identifier, skip, count):
	done = 0
	start = time.time()
	
	batch_size = 1000
	for batch in range(0, count, batch_size):
		reviews_cursor = reviews_collection.find().skip(skip + batch).limit(batch_size)
		counter = skip + batch + 1
		for review in reviews_cursor:
			#p_business_id = businesses_collection.find_one({"business_id" : review["business_id"]})["id"]
			#p_user_id = users_collection.find_one({"user_id":review["user_id"]})["id"]
			p_business_id = bs_dict[ str(review["business_id"]) ]
			p_user_id = us_dict[ str(review["user_id"]) ]
			reviews_collection.update({"_id" : review["_id"]},{'$set' : {"id" : counter, "p_business_id" : p_business_id, "p_user_id" : p_user_id}}, False, True)
			counter += 1
			done += 1
			if done % 1000 == 0:
				finish = time.time()
				print ' Review worker' + str(identifier) + ': Done ' + str(done) + ' out of ' + str(count) + ' in ' + ("%.2f" % (finish - start)) + ' sec ~ ' + ("%.2f" % (done / (finish - start))) + '/sec'
				sys.stdout.flush()

def main():
	start = time.time()
	workers = 4
	businesses_batch = businesses_count / workers
	users_batch = users_count / workers

	# Add primary index to users and businesses collections
	jobs = []
	for i in range(workers):
		bp = multiprocessing.Process(target=worker,args=((i+1), "businesses", i * businesses_batch, businesses_batch))
		up = multiprocessing.Process(target=worker,args=((i+1), "users", i * users_batch, users_batch))
		jobs.append(bp)
		bp.start()
		jobs.append(up)
		up.start()
	
	# Add remainder parts to process
	bp = multiprocessing.Process(target=worker,args=((workers+1), "businesses", workers * businesses_batch, businesses_count - workers * businesses_batch))
	up = multiprocessing.Process(target=worker,args=((workers+1), "users", workers * users_batch, users_count - workers * users_batch))
	jobs.append(bp)
	bp.start()
	jobs.append(up)
	up.start()

	for j in jobs:
		j.join()
		print '%s.exitcode = %s' % (j.name, j.exitcode)

	stage1 = time.time()

	# Store index to dict
	
	for business in businesses_cursor:
		bs_dict[str(business["business_id"])] = business["id"]

	for user in users_cursor:
		us_dict[str(user["user_id"])] = user["id"]

	# Update business_primary_index and user_primary_index
	workers = 8
	reviews_batch = reviews_count / workers
	
	rvjobs = []
	for i in range(workers):
		p = multiprocessing.Process(target=review_worker, args=((i+1), i * reviews_batch, reviews_batch))
		rvjobs.append(p)
		p.start()

	# Add remainder parts to process
	p = multiprocessing.Process(target=review_worker, args=((workers+1), workers * reviews_batch, reviews_count - workers * reviews_batch))
	rvjobs.append(p)
	p.start()

	for j in rvjobs:
		j.join()
		print '%s.exitcode = %s' % (j.name, j.exitcode)

	stage2 = time.time()
	print "Add index to businesses and users collection, duration in " + ("%.2f" % (stage1 - start)) 
	print "Update reviews collection, duration in " + ("%.2f" % (stage2 - stage1)) 


if __name__ == '__main__':
	main()