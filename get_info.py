#!env python
# -*- coding:utf-8 -*-
'''Get info module for YELP challenge data set
Extract restaurant related data
'''

import json

from settings import Settings
from data_utils import Users
from data_utils import Businesses
from data_utils import Reviews


def _test():
    u'''Test for performance'''
    bss = Businesses()
    bss.load_all_data()
    items = []
    for business in bss.cursor:
        for catego in business['categories']:
            items.append(catego)
    chains = ' '.join(items)
    chains = chains.lower()
    chains = chains.replace('(', '')
    chains = chains.replace(')', '')
    chains = chains.replace('&', '')
    chains = chains.replace('/', ' ')
    chains = chains.replace('\\', ' ')
    print chains

def res_businesses_extract():
    u'''Extract restaurant businesses'''
    bss = Businesses()
    bss.load_all_data()
    bs_index = 0
    tmp = '\"_id\": null,'
    for business in bss.cursor:
        for catego in business['categories']:
            if catego.lower() == 'restaurants':
                bs_index += 1
                business['id'] = bs_index
                business['_id'] = None
                print json.dumps(business, indent=1).replace('\n', '').replace(tmp, '')

def res_reviews_extract():
    u'''Get list of restaurant businesses'''
    dict_bs = {}
    rbss = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION)
    rbss.load_all_data()
    for res_business in rbss.cursor:
        dict_bs[res_business['business_id']] = res_business['id']

    rvs = Reviews()
    rvs.load_all_data()
    rv_index = 0
    tmp = '\"_id\": null,'
    for review in rvs.cursor:
        if review['business_id'] in dict_bs:
            rv_index += 1
            review['_id'] = None
            review['id'] = rv_index
            review['p_business_id'] = dict_bs[review['business_id']]
            print json.dumps(review, indent=1).replace('\n', '').replace(tmp, '')

def res_users_extract():
    u'''Get list of users reviewed for restaurants'''
    dict_count = {}
    dict_star1 = {}
    dict_star2 = {}
    dict_star3 = {}
    dict_star4 = {}
    dict_star5 = {}

    rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    rvs.load_all_data()
    for review in rvs.cursor:
        user_id = review['user_id']
        if user_id not in dict_count:
            dict_star1[user_id] = 0
            dict_star2[user_id] = 0
            dict_star3[user_id] = 0
            dict_star4[user_id] = 0
            dict_star5[user_id] = 0
            dict_count[user_id] = 0

        stars = review['stars']
        dict_count[user_id] += 1
        if stars == 1:
            dict_star1[user_id] += 1
        if stars == 2:
            dict_star2[user_id] += 1
        if stars == 3:
            dict_star3[user_id] += 1
        if stars == 4:
            dict_star4[user_id] += 1
        if stars == 5:
            dict_star5[user_id] += 1

    users = Users()
    users.load_all_data()
    user_index = 0
    tmp = '\"_id\": null,'
    for user in users.cursor:
        user_id = user['user_id']
        star1 = 0
        star2 = 0
        star3 = 0
        star4 = 0
        star5 = 0
        count = 0

        if user_id in dict_count:
            star1 = dict_star1[user_id]
            star2 = dict_star2[user_id]
            star3 = dict_star3[user_id]
            star4 = dict_star4[user_id]
            star5 = dict_star5[user_id]
            count = dict_count[user_id]

        if count > 0:
            user_index += 1
            aveg = (star1*1.0 + star2*2.0 + star3*3.0 + star4*4.0 + star5*5.0)/count

            user['_id'] = None
            user['id'] = user_index
            user['review_count'] = count
            user['average_stars'] = aveg
            user['stars_distribution']['one'] = star1
            user['stars_distribution']['two'] = star2
            user['stars_distribution']['three'] = star3
            user['stars_distribution']['four'] = star4
            user['stars_distribution']['five'] = star5
            print json.dumps(user, indent=1).replace('\n', '').replace(tmp, '')

def res_businesses_extract2():
    u'''Get list of restaurants'''
    dict_count = {}
    dict_star1 = {}
    dict_star2 = {}
    dict_star3 = {}
    dict_star4 = {}
    dict_star5 = {}

    rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    rvs.load_all_data()
    for review in rvs.cursor:
        stars = review['stars']
        business_id = review['business_id']
        if business_id not in dict_count:
            dict_star1[business_id] = 0
            dict_star2[business_id] = 0
            dict_star3[business_id] = 0
            dict_star4[business_id] = 0
            dict_star5[business_id] = 0
            dict_count[business_id] = 0
        dict_count[business_id] += 1
        if stars == 1:
            dict_star1[business_id] += 1
        if stars == 2:
            dict_star2[business_id] += 1
        if stars == 3:
            dict_star3[business_id] += 1
        if stars == 4:
            dict_star4[business_id] += 1
        if stars == 5:
            dict_star5[business_id] += 1

    bss = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION)
    bss.load_all_data()
    business_index = 0
    tmp = '\"_id\": null,'
    for business in bss.cursor:
        business_id = business['business_id']
        star1 = 0
        star2 = 0
        star3 = 0
        star4 = 0
        star5 = 0
        count = 0

        if business_id in dict_count:
            star1 = dict_star1[business_id]
            star2 = dict_star2[business_id]
            star3 = dict_star3[business_id]
            star4 = dict_star4[business_id]
            star5 = dict_star5[business_id]
            count = dict_count[business_id]

        if count > 0:
            aveg = (star1*1.0 + star2*2.0 + star3*3.0 + star4*4.0 + star5*5.0)/count
        else:
            aveg = 0

        business_index += 1
        business['_id'] = None
        business['id'] = business_index
        business['review_count'] = count
        business['average_stars'] = aveg
        business['stars_distribution']['one'] = star1
        business['stars_distribution']['two'] = star2
        business['stars_distribution']['three'] = star3
        business['stars_distribution']['four'] = star4
        business['stars_distribution']['five'] = star5
        print json.dumps(business, indent=1).replace('\n', '').replace(tmp, '')

def res_reviews_update_puser_id():
    u'''Get list of restaurants businesses'''
    dict_us = {}
    res_users = Users(collection_name=Settings.RES_USERS_COLLECTION)
    res_users.load_all_data()
    for user in res_users.cursor:
        dict_us[user['user_id']] = user['id']

    rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    rvs.load_all_data()
    tmp = '\"_id\": null,'
    rv_index = 0
    for review in rvs.cursor:
        if review['user_id'] in dict_us:
            rv_index += 1
            review['_id'] = None
            review['id'] = rv_index
            review['p_user_id'] = dict_us[review['user_id']]
            print json.dumps(review, indent=1).replace('\n', '').replace(tmp, '')

def test_business_stars():
    u'''Test stars_distribution of res_business collection'''
    res_bss = Businesses(collection_name=Settings.RES_BUSINESSES_COLLECTION)
    res_bss.load_all_data()
    for business in res_bss.cursor:
        print business['id']
        review_count = business['review_count']
        star1 = business['stars_distribution']['one']
        star2 = business['stars_distribution']['two']
        star3 = business['stars_distribution']['three']
        star4 = business['stars_distribution']['four']
        star5 = business['stars_distribution']['five']
        total_stars = star1 + star2 + star3 + star4 + star5
        if total_stars != review_count:
            break

def test_user_stars():
    u'''Test stars_distribution of res_users collection'''
    res_uss = Users(collection_name=Settings.RES_USERS_COLLECTION)
    res_uss.load_all_data()
    for user in res_uss.cursor:
        print user['id']
        review_count = user['review_count']
        star1 = user['stars_distribution']['one']
        star2 = user['stars_distribution']['two']
        star3 = user['stars_distribution']['three']
        star4 = user['stars_distribution']['four']
        star5 = user['stars_distribution']['five']
        total_stars = star1 + star2 + star3 + star4 + star5
        if total_stars != review_count:
            break

if __name__ == '__main__':
    #res_reviews_extract2()
    #test_user_stars()
    #test_business_stars()
    #res_businesses_extract()
    #res_reviews_extract()
    #res_users_extract()
    #res_businesses_extract2()
    res_reviews_update_puser_id()
    #_test()
