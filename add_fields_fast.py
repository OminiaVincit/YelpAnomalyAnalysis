#  Add extra fields to database
#    With users collection
#      1 star count, 2 stars count, 3 stars count, 4 stars count, 5 stars count
#      
#    With businesses collection
#      1 star count, 2 stars count, 3 stars count, 4 stars count, 5 stars count
#      
# It takes about 1 min to finish this task
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

businesses_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.BUSINESSES_COLLECTION]
businesses_cursor = businesses_collection.find()
businesses_count = businesses_cursor.count()

users_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.USERS_COLLECTION]
users_cursor = users_collection.find()
users_count = users_cursor.count()

# Initialize global us_dict and bs_dict
us_dict = {(x,y):0 for x in range(1, users_count+1) for y in range(1,6)}
bs_dict = {(x,y):0 for x in range(1, businesses_count+1) for y in range(1,6)}

def bs_worker(identifier, skip, count):
	done = 0
	start = time.time()
	#items_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.BUSINESSES_COLLECTION]

	batch_size = 1000
	for batch in range(0, count, batch_size):
		items_cursor = businesses_collection.find().skip(skip + batch).limit(batch_size)
		for item in items_cursor:
			item_id = item["id"]
			dict = {}
			for i in range(1,6):
				dict[i] = bs_dict[item_id, i]

			businesses_collection.update({"_id" : item["_id"]},{'$set' : {"stars_distribution" : {"one" : dict[1], "two" : dict[2], "three" : dict[3], "four" : dict[4], "five" : dict[5]}}},False,True)
			done += 1
			if done % 1000 == 0:
				finish = time.time()
				print 'Business' + ' worker' + str(identifier) + ': Done ' + str(done) + ' out of ' + str(count) + ' in ' + ("%.2f" % (finish - start)) + ' sec ~ ' + ("%.2f" % (done / (finish - start))) + '/sec'
				sys.stdout.flush()

def us_worker(identifier, skip, count):
	done = 0
	start = time.time()
	#items_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.USERS_COLLECTION]

	batch_size = 1000
	for batch in range(0, count, batch_size):
		items_cursor = users_collection.find().skip(skip + batch).limit(batch_size)
		for item in items_cursor:
			item_id = item["id"]
			dict = {}
			for i in range(1,6):
				dict[i] = 0
				for i in range(1,6):
					dict[i] = us_dict[item_id, i]

			users_collection.update({"_id" : item["_id"]},{'$set' : {"stars_distribution" : {"one" : dict[1], "two" : dict[2], "three" : dict[3], "four" : dict[4], "five" : dict[5]}}},False,True)
			done += 1
			if done % 1000 == 0:
				finish = time.time()
				print 'User' + ' worker' + str(identifier) + ': Done ' + str(done) + ' out of ' + str(count) + ' in ' + ("%.2f" % (finish - start)) + ' sec ~ ' + ("%.2f" % (done / (finish - start))) + '/sec'
				sys.stdout.flush()

def main():
	start = time.time()

	reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.REVIEWS_COLLECTION]
	reviews_cursor = reviews_collection.find()
	reviews_count = reviews_cursor.count()
	reviews_cursor.batch_size(1000)

	done = 0

	for rv in reviews_cursor:
		us_dict[ rv["p_user_id"], rv["stars"] ] += 1
		bs_dict[ rv["p_business_id"], rv["stars"] ] += 1
		done += 1
		if done % 1000 == 0:
			end = time.time()
			print 'Add to dict ' + ': Done ' + str(done) + ' out of ' + str(reviews_count) + ' in ' + ("%.2f" % (end - start)) + ' sec ~ ' + ("%.2f" % (done / (end - start))) + '/sec'

	stage1 = time.time()

	workers = 4
	businesses_batch = businesses_count / workers
	users_batch = users_count / workers

	jobs = []
	for i in range(workers):
		bp = multiprocessing.Process(target=bs_worker,args=((i+1), i * businesses_batch, businesses_batch))
		up = multiprocessing.Process(target=us_worker,args=((i+1), i * users_batch, users_batch))
		jobs.append(bp)
		bp.start()
		jobs.append(up)
		up.start()

	# Add remainder parts to process
	bp = multiprocessing.Process(target=bs_worker,args=((workers+1), workers * businesses_batch, businesses_count - workers * businesses_batch))
	up = multiprocessing.Process(target=us_worker,args=((workers+1), workers * users_batch, users_count - workers * users_batch))
	jobs.append(bp)
	bp.start()
	jobs.append(up)
	up.start()

	for j in jobs:
		j.join()
		print '%s.exitcode = %s' % (j.name, j.exitcode)

	finish = time.time()
	print "State 1 duration in " + ("%.2f" % (stage1 - start)) 
	print "State 2 duration in " + ("%.2f" % (finish - stage1)) 

if __name__ == '__main__':
	main()
