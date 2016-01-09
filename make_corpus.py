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
from data_utils import Reviews, GenCollection

from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import sentiwordnet as swn

EXCEPT_CHAR = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9',
u'!', u'"', u'#', u'$', u'%', u'&', u'\'', u'(', u')', u'-', u'=', u'^', u'~', 
u'¥', u'|', u'@', u'`', u'[', u'{', u';', u'+', u':', u'*', u']', u'}', 
u',', u'<', u'>', u'.', u'/', u'?', u'_']

def corpus_condition(word):
    """
    Condition of word to add to corpus
    """
    save_flag = False
    if word['pos'] in ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
        if len(word['word']) > 2:
            save_flag = True
            for ch in word['word']:
                if ch in EXCEPT_CHAR:
                    save_flag = False
                    break
    return save_flag


def add_lemmas(collection_name):
    u'''Add lemmatize to review_tags data'''
    lem = WordNetLemmatizer()
    tags = GenCollection(collection_name=collection_name)
    tags.load_all_data()
    tags.cursor.batch_size(50)
    done = 0
    start = time.time()
    outfile = collection_name + '_corpus.json'
    
    # Get the info of votes and helpful
    if True:
        vocal = {}
        idx = collection_name.find('_')
        site = collection_name[0:idx]
        name = site + '_reviews'
        rvs = Reviews(collection_name=name)
        rvs.cursor = rvs.collection.find()
        for review in rvs.cursor:
            rvid = review['review_id']
            votes = review['votes']
            helpful = review['helpful']
            vocal[rvid] = {'votes': votes, 'helpful': helpful}
        print 'Finish create vocal'

    with open(outfile, 'w') as _file:
        for tag in tags.cursor:
            nouns = []
            #words = [word for word in tag['words'] if word['pos'] in ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RBR', 'RBS']]
            #words = [word for word in words if len(word['word']) > 2]
            words = [word for word in tag['words'] if corpus_condition(word)]

            if len(words) >= 10:
                for word in words:
                    # tf = word['pos'][0].lower()
                    # if tf == 'j':
                    #     tf = 'a'
                    nouns.append(lem.lemmatize(word['word']))
                    # try:
                    #     sen_ls = swn.senti_synsets(word['word'], tf)
                    #     if len(sen_ls) != 0:
                    #         sen_score = sen_ls[0]
                    #         pos_score = sen_score.pos_score()
                    #         neg_score = sen_score.neg_score()
                    #         if pos_score - neg_score > 0.5:
                    #             nouns.append('POSREVIEW')
                    #         elif neg_score - pos_score > 0.5:
                    #             nouns.append('NEGREVIEW')
                    # except WordNetError:
                    #     pass

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
            corpus['rating'] = tag['rating']
            if not tag.get('votes') or not tag.get('helpful'):
                # Get helpful and votes information from review collection
                rvid = tag['review_id']
                votes = vocal[rvid]['votes']
                helpful = vocal[rvid]['helpful']
            else:
                votes = tag['votes']
                helpful = tag['helpful']

            corpus['votes'] = votes
            corpus['helpful'] = helpful

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
    word = {'word':'fighting', 'pos':'NN'}
    print corpus_condition(word)

    word = {'word':'fight+ing', 'pos':'NN'}
    print corpus_condition(word)

    add_lemmas(Settings.TRIPADVISOR_TAGS_COLLECTION)
    #add_lemmas(Settings.YELP_TAGS_COLLECTION)