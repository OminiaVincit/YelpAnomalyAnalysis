#!env python
# -*- coding:utf-8 -*-
'''Extract tags from reviews set
'''

import json
import time
import sys

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import regexp
import re

#from nltk.tag.stanford import POSTagger
import multiprocessing

from settings import Settings
from data_utils import Reviews

def extract_tags_job(collection_name, identifier, skip, count):
    u'''Extract tags'''
    idx = collection_name.find('_')
    name = collection_name[0:idx]
    # Debug time
    done = 0
    start = time.time()
    filename = name + '_tags_' + str(identifier) + '.json'

    # Load stopwords and tokenizer
    stopwds = stopwords.words('english')
    tokenizer = regexp.RegexpTokenizer("[\w’]+", flags=re.UNICODE)

    #english_postagger = POSTagger('./postagger/models/wsj-0-18-left3words-distsim.tagger', \
    #    './postagger/stanford-postagger.jar')
    rvs = Reviews(collection_name=collection_name)
    batch_size = 50

    with open(filename, 'w') as _file:
        for batch in range(0, count, batch_size):
            lm_size = min(batch_size, count-batch)
            rvs.cursor = rvs.collection.find().skip(skip + batch).limit(lm_size)
            for review in rvs.cursor:
                words = []
                if not review.get('text'):
                    continue
                
                tokens = tokenizer.tokenize(review['text'])
                tokens = [token for token in tokens if len(token) > 2]
                tokens = [token.lower() for token in tokens if token not in stopwds]
                
                tagged_text = nltk.pos_tag(tokens)
                #tagged_text = english_postagger.tag(tokens)
                for word, tag in tagged_text:
                    words.append({'word': word, 'pos': tag})

                # sentences = nltk.sent_tokenize(review['text'].lower())
                # for sentence in sentences:
                #     try:
                #         sentence = sentence.decode('utf-8')
                #         tokens = nltk.word_tokenize(sentence)
                #         text = [word for word in tokens if word not in stopwords]
                #         tagged_text = nltk.pos_tag(text)
                #         #tagged_text = english_postagger.tag(tokens)
                #         for word, tag in tagged_text:
                #             words.append({'word': word, 'pos': tag})
                #     except UnicodeEncodeError:
                #         pass

                tag = {}
                tag['review_id'] = review['review_id']
                tag['item_id'] = review['item_id']
                tag['user_id'] = review['user_id']
                tag['rating'] = review['rating']
                tag['votes'] = review['votes']
                tag['helpful'] = review['helpful']
                tag['text'] = review['text']
                tag['words'] = words
                _file.write(json.dumps(tag, indent=1).replace('\n', ''))
                _file.write('\n')

                done += 1
                if done % 100 == 0:
                    end = time.time()
                    print 'Worker' + str(identifier) + ': Done ' + str(done) + \
                        ' out of ' + str(count) + ' in ' + \
                        ('%.2f' % (end - start)) + ' sec ~ ' + \
                        ('%.2f' % (done / (end - start))) + '/sec'
                    sys.stdout.flush()

def extract_tags(collection_name):
    u'''Extract tags by collection'''
    rvs = Reviews(collection_name=collection_name)
    rvs.load_all_data()
    workers = 4
    batch = rvs.count / workers
    jobs = []
    for i in range(workers):
        _ps = multiprocessing.Process(target=extract_tags_job, \
            args=(collection_name, (i+1), i*batch, batch))
        jobs.append(_ps)
        _ps.start()

    # Add remainder parts to process
    _rp = multiprocessing.Process(target=extract_tags_job, args=(collection_name, (workers+1), \
        workers*batch, rvs.count-workers*batch))
    jobs.append(_rp)
    _rp.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)

def test_review(collection_name):
    u'''Test field in collection_name'''
    rvs = Reviews(collection_name=collection_name)
    rvs.load_all_data()
    for review in rvs.cursor:
        if not review.get('text') or not review.get('item_id'):
            print review

if __name__ == '__main__':
    extract_tags(Settings.YELP_REVIEWS_COLLECTION)
    extract_tags(Settings.TRIPADVISOR_REVIEWS_COLLECTION)
    #test_review(Settings.MOVIES_REVIEWS_COLLECTION)
