u'''Setting for environment'''

class Settings:
    u''' For environmental settings '''

    def __init__(self):
        u'''Init'''
        pass

    DATASET_FILE = ''
    MONGO_CONNECTION_STRING = 'mongodb://localhost:27017'
    DATABASE = 'predict'
    
    YELP_REVIEWS_COLLECTION = 'yelp_reviews'
    TRIPADVISOR_REVIEWS_COLLECTION = 'tripadvisor_reviews'
    MOVIES_REVIEWS_COLLECTION = 'movies_reviews'

    YELP_TAGS_COLLECTION = 'yelp_tags'
    TRIPADVISOR_TAGS_COLLECTION = 'tripadvisor_tags'
    MOVIES_TAGS_COLLECTION = 'movies_tags'

    YELP_CORPUS_COLLECTION = 'yelp_corpus'
    TRIPADVISOR_CORPUS_COLLECTION = 'tripadvisor_corpus'
    MOVIES_CORPUS_COLLECTION = 'movies_corpus'
    
    YELP_TFIDF_COLLECTION = 'yelp_tfidf_tokens'
    TRIPADVISOR_TFIDF_COLLECTION = 'tripadvisor_tfidf_tokens'

    TFIDF_DIM = 1024
    GALC_DIM = 39
    LIWC_DIM = 64
    INQUIRER_DIM = 182

    NUMTOPICS = 50
    EPSILON = 1e-7

    DATA_DIR = r'/home/zoro/work/Dataset'
