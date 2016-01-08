#!env python
# -*- coding:utf-8 -*-
'''Extract topics features
'''

import os
import time
import sys
import json
import numpy as np
import argparse
import json
from settings import Settings

def extract_topics_features_from_file(in_file, out_file, num_features):
  topics = 'topics_' + str(num_features)
  with open(out_file, 'w') as out_f:
    with open(in_file, 'r') as in_f:    
        for line in in_f:
          data = json.loads(line)
          votes = int(data['votes'])
          helpful = int(data['helpful'])
          if votes >= 10:
            rate = float(helpful) / float(votes)
            numls = []
            for i in range(num_features):
              numls.append(data[topics][str(i)])
            numls.append(helpful)
            numls.append(votes)
            numls.append(rate)
            wline = ' '.join([str(n) for n in numls]) + '\n'
            out_f.write(wline)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datadir', type=str, default='/home/zoro/work/Dataset')
  parser.add_argument('--site', type=str, default='yelp',
                      choices=['yelp', 'tripadvisor'])
  parser.add_argument('--num_features', type=int, default=64)
  args = parser.parse_args()
  print args

  for st in ['yelp', 'tripadvisor']:
    in_file = os.path.join(args.datadir, st + '_reviews_topics_distribution.json')
    for num_f in [100, 144, 196, 256]:
      out_file = os.path.join(args.datadir, st + '_topics_' \
                          + str(num_f) + '_features.txt')
      print st, num_f
      extract_topics_features_from_file(in_file, out_file, num_f)
      
