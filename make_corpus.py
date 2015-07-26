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
from data_utils import GenCollection

from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import sentiwordnet as swn

def add_lemmas(collection_name):
    u'''Add lemmatize to review_tags data'''
    lem = WordNetLemmatizer()
    tags = GenCollection(collection_name=collection_name)
    tags.load_all_data()
    tags.cursor.batch_size(50)
    done = 0
    start = time.time()
    outfile = collection_name + '_corpus2.json'
    with open(outfile, 'a') as _file:
        for tag in tags.cursor:
            nouns = []
            words = [word for word in tag['words'] if word['pos'] in ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RBR', 'RBS']]
            words = [word for word in words if len(word['word']) > 1]

            if len(words) >= 10:
                for word in words:
                    tf = word['pos'][0].lower()
                    if tf == 'j':
                        tf = 'a'
                    nouns.append(lem.lemmatize(word['word']))
                    try:
                        sen_ls = swn.senti_synsets(word['word'], tf)
                        if len(sen_ls) != 0:
                            sen_score = sen_ls[0]
                            pos_score = sen_score.pos_score()
                            neg_score = sen_score.neg_score()
                            if pos_score - neg_score > 0.5:
                                nouns.append('POSREVIEW')
                            elif neg_score - pos_score > 0.5:
                                nouns.append('NEGREVIEW')
                    except WordNetError:
                        pass

            # if len(nouns) >= 10:
            #     rating = 3
            #     if tag.get('rating'):
            #         rating = int(tag['rating'])
            #     if rating > 3:
            #         nouns.append('HIGHRATE')
            #     elif rating < 3:
            #         nouns.append('LOWRATE')
            
            corpus = {}
            corpus['review_id'] = tag['review_id']
            corpus['item_id'] = tag['item_id']
            corpus['user_id'] = tag['user_id']
            #corpus['text'] = tag['text']
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
    add_lemmas(Settings.TRIPADVISOR_TAGS_COLLECTION)
