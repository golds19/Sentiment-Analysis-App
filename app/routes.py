from flask import render_template, request
from .models import SentimentModel

def init_routes(app):
    model = SentimentModel()  # Load model once
    
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        text = request.form['text']
        prediction = model.predict(text)
        return render_template('index.html', prediction=prediction)