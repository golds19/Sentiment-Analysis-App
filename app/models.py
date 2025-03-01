import joblib
from config import Config
from app.utils import clean_func

class SentimentModel:
    def __init__(self):
        self.model = joblib.load(Config.MODEL_PATH)
        self.vectorizer = joblib.load(Config.VECTORIZER_PATH)
    
    def predict(self, text):
        clean_text = clean_func(text)
        text_vec = self.vectorizer.transform([clean_text])
        pred = self.model.predict(text_vec)
        return pred[0]