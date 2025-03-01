import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import warnings

warnings.filterwarnings('ignore')
from app.utils import clean_func

# loading the data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data = pd.read_csv('train.csv', encoding='latin1')

# drop missing values
data.dropna(inplace=True)

# cleaning the data
data['selected_text'] = data['selected_text'].apply(clean_func)

# splitting the data into training and testing sets
X = data['selected_text']
y = data['sentiment']

# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the data
vectorization = TfidfVectorizer()
XV_train = vectorization.fit_transform(X_train)
XV_test = vectorization.transform(X_test)

# Train the model
lr = LogisticRegression(n_jobs=-1)
lr.fit(XV_train, y_train)

# saving the model and vectorizer
joblib.dump(lr, 'models/sentiment_model.pkl')
joblib.dump(vectorization, 'models/vectorizer.pkl')
print("Model trained and saved successfully")
