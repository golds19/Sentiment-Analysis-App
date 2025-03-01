import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sentiment_model.pkl')
    VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
    DEBUG = True