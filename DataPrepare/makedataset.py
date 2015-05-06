#!env python
# -*- coding:utf-8 -*-
u'''Make dataset for our predictor'''

import os
import json
import uuid

def make_from_yelp(infile, outfile):
    u'''Make dataset from json yelp data file'''
    count = 0
    YELP_DATA_DIR = ur'../../Dataset/yelp_dataset_challenge_academic_dataset_20150209/restaurant_only'
    inpath = os.path.join(YELP_DATA_DIR, infile)
    with open(outfile, 'w') as out_file:
        with open(inpath, 'r') as in_file:
            for line in in_file:
                features = json.loads(line)
                review = {}
                review['review_id'] = features['review_id']
                review['user_id'] = features['user_id']
                review['item_id'] = features['business_id']
                review['text'] = features['text']
                review['rating'] = float(features['stars'])
                review['votes'] = int(features['votes']['useful']) + \
                    int(features['votes']['cool']) + int(features['votes']['funny'])
                review['helpful'] = int(features['votes']['useful'])
                out_file.write(json.dumps(review, indent=1).replace('\n', ''))
                out_file.write('\n')
                count += 1
                print count

def make_from_tripadvisor(infolder=ur'../../Dataset/TripAdvisorReviews', 
    outfile='tripadvisor_reviews.json'):
    u'''Make dataset from tripadvisor data folders'''
    count = 0
    with open(outfile, 'w') as out_file:
        infiles = os.listdir(infolder)
        for infile in infiles:
            inpath = os.path.join(infolder, infile)
            review = {}
            with open(inpath, 'r') as in_file:
                for line in in_file:
                    line = line.replace('\r\n','')
                    if len(line) > 0:
                        bg = line.find('<')
                        ed = line.find('>')
                        if bg >= 0 and ed >= 0:
                            tag = line[(bg+1):ed]
                            if tag == 'Author':
                                review['user_id'] = line[(ed+1):]
                            if tag == 'Content':
                                review['text'] = line[(ed+1):]
                            if tag == 'No. Reader':
                                review['votes'] = int(line[(ed+1):])
                            if tag == 'No. Helpful':
                                review['helpful'] = int(line[(ed+1):])
                            if tag == 'Overall':
                                review['rating'] = float(line[(ed+1):])
                    else:
                        if len(review) > 0:
                            review['item_id'] = infile[6:(-4)]
                            review['review_id'] = str(uuid.uuid4())
                            # Print review
                            out_file.write(json.dumps(review, indent=1).replace('\n', ''))
                            out_file.write('\n')
                            count += 1
                            print count

                        review = {}

def make_from_webmovies(infile, outfile):
    u'''Make dataset from web movies data folders'''
    MOVIES_DATA_DIR = ur'../../Dataset'
    count = 0
    inpath = os.path.join(MOVIES_DATA_DIR, infile)
    with open(outfile, 'w') as out_file:
        review = {}
        with open(inpath, 'r') as in_file:
            for line in in_file:
                line = line.replace('\n','')
                if len(line) > 0:
                    bg = line.find('/')
                    ed = line.find(':')
                    if bg >= 0 and ed >= 0:
                        tag = line[(bg+1):ed]
                        if tag == 'productId':
                            review['item_id'] = line[(ed+1):]
                        if tag == 'userId':
                            review['user_id'] = line[(ed+1):]
                        if tag == 'text':
                            content = line[(ed+1):].replace('<br />', ' ')
                            content = content.replace('"', ' ')
                            content = content.replace('&quot', '')
                            review['text'] = content
                        if tag == 'helpfulness':
                            rt = line[(ed+1):]
                            sl = rt.find('/')
                            review['votes'] = int(rt[(sl+1):])
                            review['helpful'] = int(rt[:sl])
                        if tag == 'score':
                            review['rating'] = float(line[(ed+1):])
                else:
                    if len(review) > 0:
                        review['review_id'] = str(uuid.uuid4())
                        # Print review
                        try:
                            out_file.write(json.dumps(review, indent=1).replace('\n', ''))
                            out_file.write('\n')
                            count += 1
                            print count
                        except UnicodeDecodeError:
                            print 'Oops! UnicodeDecodeError'

                    review = {}

if __name__ == '__main__':
    #make_from_yelp('res_reviews.json', 'yelp_reviews.json')
    make_from_webmovies('movies.txt', 'movies_reviews.json')


