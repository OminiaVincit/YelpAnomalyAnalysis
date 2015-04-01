#  Add extra fields to database
#    With users collection
#      1 star count, 2 stars count, 3 stars count, 4 stars count, 5 stars count
#      
#    With businesses collection
#      1 star count, 2 stars count, 3 stars count, 4 stars count, 5 stars count
#      
# db.businesses.update({},{$set : {"stars_distribution" : {"one" : 0, "two" : 0, "three" : 0, "four" : 0, "five" : 0}}},false,true)
# db.users.update({},{$set : {"stars_distribution" : {"one" : 0, "two" : 0, "three" : 0, "four" : 0, "five" : 0}}},false,true)

import os
import time

from pymongo import MongoClient

from settings import Settings

reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.REVIEWS_COLLECTION]
businesses_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.BUSINESSES_COLLECTION]
users_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.YELP_DATABASE][Settings.USERS_COLLECTION]

reviews_cursor = reviews_collection.find()
businesses_cursor = businesses_collection.find()
users_cursor = users_collection.find()

reviews_count = reviews_cursor.count()

reviews_cursor.batch_size(1000)
businesses_cursor.batch_size(1000)
users_cursor.batch_size(1000)

# Initialize first by 
businesses_collection.update({},{'$set' : {"stars_distribution" : {"one" : 0, "two" : 0, "three" : 0, "four" : 0, "five" : 0}}},False,True)
users_collection.update({},{'$set' : {"stars_distribution" : {"one" : 0, "two" : 0, "three" : 0, "four" : 0, "five" : 0}}},False,True)
rviter = 0

start = time.time()
for review in reviews_cursor:
	business_id = review["business_id"]
	user_id = review["user_id"]
	review_stars = review["stars"]
	business = businesses_collection.find_one({"business_id" : business_id})
	user = users_collection.find_one({"user_id" : user_id})

	if review_stars == 1:
		index = "stars_distribution.one"
	if review_stars == 2: 
		index = "stars_distribution.two"
	if review_stars == 3:
		index = "stars_distribution.three"
	if review_stars == 4:
		index = "stars_distribution.four"
	if review_stars == 5:
		index = "stars_distribution.five"
	#businesses_collection.update({'business_id':business_id},{'$inc':{index:1}},upsert=False, multi=False)
	#users_collection.update({'user_id':user_id},{'$inc':{index:1}},upsert=False, multi=False)
	rviter = rviter+1
	print rviter
	if rviter > 10:
		break

finish = time.time()
duration = (finish - start)*reviews_count/(10.0*3600)	
print duration
