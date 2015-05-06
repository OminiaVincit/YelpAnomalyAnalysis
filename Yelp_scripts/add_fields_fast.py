#!env python
# -*- coding:utf-8 -*-

''' Add extra fields to database
    With users collection
        1 star count, 2 stars count, 3 stars count, 4 stars count, 5 stars count
    With businesses collection
        1 star count, 2 stars count, 3 stars count, 4 stars count, 5 stars count

It takes about 1 min to finish this task
Initialize first by
   db.businesses.update({},{$set : {'stars_distribution' : \
    {'one' : 0, 'two' : 0, 'three' : 0, 'four' : 0, 'five' : 0}}},false,true)
   db.users.update({},{$set : {'stars_distribution' : \
    {'one' : 0, 'two' : 0, 'three' : 0, 'four' : 0, 'five' : 0}}},false,true)
'''

import multiprocessing
import time
import sys

from data_utils import Users
from data_utils import Businesses
from data_utils import Reviews

# Initialize global us_dict and bs_dict
BSS = Businesses()
BSS.load_all_data()

USS = Users()
USS.load_all_data()

us_dict = {(x, y):0 for x in range(1, USS.count+1) for y in range(1, 6)}
bs_dict = {(x, y):0 for x in range(1, BSS.count+1) for y in range(1, 6)}


def bs_worker(identifier, skip, count):
    u'''Business workers for multiprocessing'''
    done = 0
    start = time.time()
    businesses_collection = Businesses().collection
    batch_size = 1000
    for batch in range(0, count, batch_size):
        lm_size = min(batch_size, count-batch)
        items_cursor = businesses_collection.find().skip(skip+batch).limit(lm_size)
        for item in items_cursor:
            item_id = item['id']
            tmp_dict = {}
            for i in range(1, 6):
                tmp_dict[i] = bs_dict[item_id, i]

            businesses_collection.update({'_id' : item['_id']}, \
                {'$set' : {'stars_distribution' : \
                {'one' : tmp_dict[1], 'two' : tmp_dict[2], 'three' : tmp_dict[3], \
                'four' : tmp_dict[4], 'five' : tmp_dict[5]}}}, \
                False, True)
            done += 1
            if done % 1000 == 0:
                finish = time.time()
                print 'Business' + ' worker' + str(identifier) + ': Done ' + str(done) + \
                    ' out of ' + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
                    ' sec ~ ' + ('%.2f' % (done / (finish - start))) + '/sec'
                sys.stdout.flush()


def us_worker(identifier, skip, count):
    u'''User workers for multiprocessing'''
    done = 0
    start = time.time()
    users_collection = Users().collection
    batch_size = 1000
    for batch in range(0, count, batch_size):
        lm_size = min(batch_size, count-batch)
        items_cursor = users_collection.find().skip(skip + batch).limit(lm_size)
        for item in items_cursor:
            item_id = item['id']
            tmp_dict = {}
            for i in range(1, 6):
                tmp_dict[i] = us_dict[item_id, i]

            users_collection.update({'_id' : item['_id']}, \
                {'$set' : {'stars_distribution' : \
                {'one' : tmp_dict[1], 'two' : tmp_dict[2], 'three' : tmp_dict[3], \
                'four' : tmp_dict[4], 'five' : tmp_dict[5]}}}, False, True)
            done += 1
            if done % 1000 == 0:
                finish = time.time()
                print 'User' + ' worker' + str(identifier) + ': Done ' + str(done) + ' out of ' \
                    + str(count) + ' in ' + ('%.2f' % (finish - start)) + \
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

    done = 0

    for review in rvs.cursor:
        us_dict[review['p_user_id'], review['stars']] += 1
        bs_dict[review['p_business_id'], review['stars']] += 1
        done += 1
        if done % 1000 == 0:
            end = time.time()
            print 'Add to dict ' + ': Done ' + str(done) + ' out of ' + str(rvs.count) + \
                ' in ' + ('%.2f' % (end - start)) + ' sec ~ ' + \
                ('%.2f' % (done / (end - start))) + '/sec'

    stage1 = time.time()
    workers = 4
    businesses_batch = bss.count / workers
    users_batch = uss.count / workers

    jobs = []
    for i in range(workers):
        _bp = multiprocessing.Process(target=bs_worker, \
            args=((i+1), i*businesses_batch, businesses_batch))
        _up = multiprocessing.Process(target=us_worker, \
            args=((i+1), i*users_batch, users_batch))
        jobs.append(_bp)
        _bp.start()
        jobs.append(_up)
        _up.start()

    # Add remainder parts to process
    _bp = multiprocessing.Process(target=bs_worker, args=((workers+1), \
        workers*businesses_batch, bss.count-workers*businesses_batch))
    _up = multiprocessing.Process(target=us_worker, args=((workers+1), \
        workers*users_batch, uss.count-workers*users_batch))
    jobs.append(_bp)
    _bp.start()
    jobs.append(_up)
    _up.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)

    finish = time.time()
    print 'State 1 duration in ' + ('%.2f' % (stage1 - start))
    print 'State 2 duration in ' + ('%.2f' % (finish - stage1))

if __name__ == '__main__':
    main()
