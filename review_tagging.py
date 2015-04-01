#!env python
# -*- coding:utf-8 -*-
'''Extract tags from reviews set
'''

import json
import time
import sys
import nltk
# from nltk.tag.stanford import POSTagger
import multiprocessing

from settings import Settings
from data_utils import Reviews

def load_stopwords():
    u'''Load stop words'''
    stopwords = {}
    with open('stopwords2.txt', 'rU') as _file:
        for line in _file:
            stopwords[line.strip()] = 1
    return stopwords

def extract_tags(identifier, skip, count):
    u'''Extract tags'''
    # Debug time
    done = 0
    start = time.time()
    stopwords = load_stopwords()
    filename = 'res_tags_' + str(identifier) + '.json'
    english_postagger = POSTagger('./postagger/models/wsj-0-18-left3words-distsim.tagger', \
        './postagger/stanford-postagger.jar')
    rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    batch_size = 50

    with open(filename, 'a') as _file:
        for batch in range(0, count, batch_size):
            lm_size = min(batch_size, count-batch)
            rvs.cursor = rvs.collection.find().skip(skip + batch).limit(lm_size)
            for review in rvs.cursor:
                # print review['id'], review['text']
                words = []
                sentences = nltk.sent_tokenize(review['text'].lower().encode('utf-8'))
                for sentence in sentences:
                    tokens = nltk.word_tokenize(sentence)
                    # text = [word for word in tokens if word not in stopwords]
                    # text = [word.encode('utf-8') for word in tokens]
                    # tagged_text = nltk.pos_tag(tokens)
                    tagged_text = english_postagger.tag(tokens)
                    for word, tag in tagged_text:
                        if word not in stopwords:
                            words.append({'word': word, 'pos': tag})
                tag = {}
                tag['review_id'] = review['review_id']
                tag['business_id'] = review['business_id']
                tag['user_id'] = review['user_id']
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

def main():
    u'''Main function'''
    rvs = Reviews(collection_name=Settings.RES_REVIEWS_COLLECTION)
    rvs.load_all_data()
    workers = 3
    batch = rvs.count / workers
    jobs = []
    for i in range(workers):
        _ps = multiprocessing.Process(target=extract_tags, \
            args=((i+1), i*batch, batch))
        jobs.append(_ps)
        _ps.start()

    # Add remainder parts to process
    _rp = multiprocessing.Process(target=extract_tags, args=((workers+1), \
        workers*batch, rvs.count-workers*batch))
    jobs.append(_rp)
    _rp.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)

if __name__ == '__main__':
    main()

