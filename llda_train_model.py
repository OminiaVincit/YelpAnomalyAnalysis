import sys, string, random, numpy
from nltk.corpus import reuters
from llda import LLDA
from optparse import OptionParser

from settings import Settings
from data_utils import GenCollection

LABELS = ['Terrible', 'Bad', 'Normal', 'Good', 'Excellent']
parser = OptionParser()
parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
parser.add_option("-k", dest="K", type="int", help="number of topics", default=50)
parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
parser.add_option("-s", dest="seed", type="int", help="random seed", default=None)
parser.add_option("-n", dest="samplesize", type="int", help="dataset sample size", default=100)
(options, args) = parser.parse_args()
random.seed(options.seed)
numpy.random.seed(options.seed)

collection_name = Settings.YELP_CORPUS_COLLECTION
corpus_collection = GenCollection(collection_name=collection_name)
corpus_collection.load_all_data()
corpus_cursor = corpus_collection.cursor

labels = []
corpus = []
for review in corpus_cursor:
    votes = review['votes']
    if votes < 10:
        continue
    helpful = review['helpful']
    label = int(helpful / float(votes*0.2))
    if label == 5:
        label = 4
    labels.append([LABELS[label]])
    corpus.append(review['words'])
labelset = list(set(reduce(list.__add__, labels)))
print labelset

llda = LLDA(options.K, options.alpha, options.beta)
llda.set_corpus(labelset, corpus, labels)

print "M=%d, V=%d, L=%d, K=%d" % (len(corpus), len(llda.vocas), len(labelset), options.K)

for i in range(options.iteration):
    sys.stderr.write("-- %d : %.4f\n" % (i, llda.perplexity()))
    llda.inference()
print "perplexity : %.4f" % llda.perplexity()

phi = llda.phi()
for k, label in enumerate(labelset):
    print "\n-- label %d : %s" % (k, label)
    for w in numpy.argsort(-phi[k])[:20]:
        print "%s: %.4f" % (llda.vocas[w], phi[k,w])