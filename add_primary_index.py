#!env python
# -*- coding:utf-8 -*-

''' Add primary index to items in collections
This task takes about 15 mins in my env (slow)
Before run this scrip, do the following commands for update fields at database
    db.users.update({},{$set : {'id' : 0}}, false, true)
    db.businesses.update({},{$set : {'id' : 0}}, false, true)
    db.reviews.update({},{$set : {'id' : 0, 'p_user_id' : 0, 'p_business_id' : 0}}, false, true)
'''

import multiprocessing
import time
import sys
from data_utils import Users
from data_utils import Businesses
from data_utils import Reviews

bs_dict = {}
us_dict = {}


def worker(identifier, collection, skip, count):
    u'''Worker for multi-processing collections'''
    done = 0
    start = time.time()
    if str(collection) == 'businesses':
        items_collection = Businesses().collection
    elif str(collection) == 'users':
        items_collection = Users().collection
    else:
        items_collection = Reviews().collection

    batch_size = 500
    for batch in range(0, count, batch_size):
        items_cursor = items_collection.find().skip(skip + batch).limit(batch_size)
        counter = skip + batch + 1
        for item in items_cursor:
            items_collection.update({'_id' : item['_id']}, {'$set' : {'id':counter}}, False, True)
            counter += 1
            done += 1
            if done % 1000 == 0:
                finish = time.time()
                print str(collection) + ' worker' + str(identifier) + ': Done ' + str(done) + \
                    ' out of ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + ' sec ~ ' + \
                    ('%.2f' % (done / (finish - start))) + '/sec'
                sys.stdout.flush()

def review_worker(identifier, skip, count):
    u'''Review worker for multi-processing'''
    done = 0
    start = time.time()
    reviews_collection = Reviews().collection
    batch_size = 1000
    for batch in range(0, count, batch_size):
        reviews_cursor = reviews_collection.find().skip(skip + batch).limit(batch_size)
        counter = skip + batch + 1
        for review in reviews_cursor:
            p_business_id = bs_dict[str(review['business_id'])]
            p_user_id = us_dict[str(review['user_id'])]
            reviews_collection.update({'_id' : review['_id']}, \
                {'$set' : \
                {'id' : counter, 'p_business_id' : p_business_id, 'p_user_id' : p_user_id}}, \
                False, True)
            counter += 1
            done += 1
            if done % 1000 == 0:
                finish = time.time()
                print ' Review worker' + str(identifier) + ': Done ' + str(done) + \
                    ' out of ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
                    ' sec ~ ' + ('%.2f' % (done / (finish - start))) + '/sec'
                sys.stdout.flush()

def main():
    u'''Main function'''
    start = time.time()

    bss = Businesses()
    bss.load_all_data()

    uss = Users()
    uss.load_all_data()

    rvs = Reviews()
    rvs.load_all_data()

    workers = 4
    businesses_batch = bss.count / workers
    users_batch = uss.count / workers

    # Add primary index to users and businesses collections
    jobs = []
    for i in range(workers):
        _bp = multiprocessing.Process(target=worker, args=((i+1), 'businesses', \
            i*businesses_batch, businesses_batch))
        _up = multiprocessing.Process(target=worker, args=((i+1), 'users', \
            i*users_batch, users_batch))
        jobs.append(_bp)
        _bp.start()
        jobs.append(_up)
        _up.start()
    # Add remainder parts to process
    _bp = multiprocessing.Process(target=worker, args=((workers+1), 'businesses', \
        workers*businesses_batch, bss.count-workers*businesses_batch))
    _up = multiprocessing.Process(target=worker, args=((workers+1), 'users', \
        workers*users_batch, uss.count-workers*users_batch))
    jobs.append(_bp)
    _bp.start()
    jobs.append(_up)
    _up.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)

    stage1 = time.time()

    # Store index to dict
    for business in bss.cursor:
        bs_dict[str(business['business_id'])] = business['id']

    for user in uss.cursor:
        us_dict[str(user['user_id'])] = user['id']

    # Update business_primary_index and user_primary_index
    workers = 8
    reviews_batch = rvs.count / workers
    rvjobs = []
    for i in range(workers):
        _rp = multiprocessing.Process(target=review_worker, args=((i+1), i*reviews_batch, reviews_batch))
        rvjobs.append(_rp)
        _rp.start()

    # Add remainder parts to process
    _rp = multiprocessing.Process(target=review_worker, args=((workers+1), \
        workers*reviews_batch, rvs.count-workers*reviews_batch))
    rvjobs.append(_rp)
    _rp.start()

    for j in rvjobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)
    stage2 = time.time()
    print 'Add index to businesses and users collection, duration in ' + ('%.2f' % (stage1 - start))
    print 'Update reviews collection, duration in ' + ('%.2f' % (stage2 - stage1))


if __name__ == '__main__':
    main()
