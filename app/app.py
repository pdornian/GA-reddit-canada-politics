from flask import Flask, render_template, request, jsonify
import requests
import numpy as np 
import pandas as pd
import pickle
import re
import sklearn 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

app = Flask(__name__, static_url_path='/static')
#models and associated saved files
#the trained pickle model doesn't seem to record the stemmer in the pipeline and otherwise breaks on import
#so defining the stemmer manually before import
snow=SnowballStemmer("english")
def snowball_tokens2(text):
    text_processed = re.sub(r'[^A-Za-z]', ' ', text).split()
    tokens = [snow.stem(word) for word in text_processed]
    return tokens
final_model = pickle.load(open('./models/final_model.p','rb'))



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
	pred_text=pred_post(comment, final_model)
	return render_template('index.html', classification=pred_text)

# Call app.run(debug=True) when python script is called
if __name__=='__main__':
	app.run(debug=True)