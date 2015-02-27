u'''Setting for environment'''

class Settings:
    u''' For environmental settings '''

    def __init__(self):
        u'''Init'''
        pass

    DATASET_FILE = ''
    MONGO_CONNECTION_STRING = 'mongodb://localhost:27017'
    YELP_DATABASE = 'yelp'
    REVIEWS_COLLECTION = 'reviews'
    BUSINESSES_COLLECTION = 'businesses'
    USERS_COLLECTION = 'users'

    RES_BUSINESSES_COLLECTION = 'res_businesses'
    RES_REVIEWS_COLLECTION = 'res_reviews'
    RES_USERS_COLLECTION = 'res_users'
    