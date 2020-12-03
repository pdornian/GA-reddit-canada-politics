from flask import Flask, render_template, request, jsonify
import requests
import numpy as np 
import pandas as pd
import pickle

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.naive_bayes import BernoulliNB

app = Flask(__name__)

#I'm just training the damn model here due to dependency issues in loading a pickled pipeline.

snow=SnowballStemmer("english")
def snowball_tokens2(text):
    text_processed = re.sub(r'[^A-Za-z]', ' ', text).split()
    tokens = [snow.stem(word) for word in text_processed]
    return tokens
final_model = pickle.load(open('./models/final_model.p','rb'))

canada_df = pd.read_csv('..\\data\\canada_subreddit_comments.csv')

custom_stopwords= stopwords.words('english')
custom_stopwords.extend(['people', 'like', 'canada'])
custom_stopwords = [snow.stem(word) for word in custom_stopwords]
extra_stopwords=['get', 'would', 'gt', 'one', 'go', 'make', 
                 'actual', 'also', 'back', 'us', 'use', 'could', 'say', 'said', 'see', 'back', 'come',
                'canadian', 'look']

custom_stopwords.extend(extra_stopwords)

X=canada_df['body_processed']
y=canada_df['subreddit_bin']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1920)

bnb_final= Pipeline([('cv', CountVectorizer(tokenizer=snowball_tokens2,
                                            max_features=4000,
                                            stop_words=custom_stopwords,
                                            binary=True)), 
                     ('bnb', BernoulliNB())])


bnb_final.fit(X_train, y_train);

#=========================FUNCTIONS================================================
def pred_post(text, model):
    label_map=['r/Canada', 'r/OnGuardForThee']
    post=[text]
    prediction=model.predict([text])[0]
    pred_text=label_map[prediction]
    pred_prob=round(model.predict_proba([text])[0,prediction] * 100, 1)
    return f"The model thinks this seems like an {pred_text} comment with {pred_prob}% certainty"





#==================================FLASK PAGE=============================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=["POST"])
def classify():
	comment = request.form.get('comment')
	pred_text=pred_post(comment, bnb_final)
	return render_template('index.html', classification=pred_text)
