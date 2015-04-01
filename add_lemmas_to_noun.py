#!env python
# -*- coding:utf-8 -*-
'''Extract tags from reviews set
'''

import json
import time
import sys
from nltk.stem.wordnet import WordNetLemmatizer
import multiprocessing

from settings import Settings
from data_utils import Users
from data_utils import Businesses
from data_utils import Reviews
from data_utils import Tags

def add_lemmas():
    u'''Add lemmatize to review_tags data'''
    lem = WordNetLemmatizer()
    tags = Tags(collection_name=Settings.RES_TAGS_COLLECTION)
    tags.load_all_data()
    tags.cursor.batch_size(5000)
    done = 0
    start = time.time()
    with open('res_corpus.json', 'a') as _file:
        for tag in tags.cursor:
            nouns = []
            words = [word for word in tag['words'] if word['pos'] in ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RBR', 'RBS']]
            words = [word for word in words if word not in ['(', ')', '{', '}', '$', '#', '&', '~', '¥', '"']]

            for word in words:
                nouns.append(lem.lemmatize(word['word']))
            corpus = {}
            corpus['review_id'] = tag['review_id']
            corpus['business_id'] = tag['business_id']
            corpus['user_id'] = tag['user_id']
            corpus['text'] = tag['text']
            corpus['words'] = nouns
            _file.write(json.dumps(corpus, indent=1).replace('\n', ''))
            _file.write('\n')

            done += 1
            if done % 100 == 0:
                end = time.time()
                print 'Done ' + str(done) + \
                    ' out of ' + str(tags.count) + ' in ' + \
                    ('%.2f' % (end - start)) + ' sec ~ ' + \
                    ('%.2f' % (done / (end - start))) + '/sec'
                sys.stdout.flush()

if __name__ == '__main__':
    add_lemmas()
