import pickle
#import sqlite3
#import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

load_pickle_file = open("/home/ml_rejolut/Desktop/ML/Sentiment_Analysis_Python_NLP/sentiment_analysis.pickle", "rb")
classifier = pickle.load(load_pickle_file)
load_pickle_file.close()

pickle_file = open("/home/ml_rejolut/Desktop/ML/Sentiment_Analysis_Python_NLP/bow.pickle", "rb")
model = pickle.load(pickle_file)
pickle_file.close()

@app.route('/')
def home():
    return "Machine Learning model for Sentiment Analysis."
    
@app.route('/enter_input_text', methods = ['POST'])
def enter_input_text():
    content = request.json
    print(content)
    # newReview = input('Type the review: ')
    newReview = content['newReview']
    new_review = model.transform([newReview]).toarray()
    prediction =  classifier.predict(new_review)
    print(prediction)
    
    if newReview == ' ' :
        return 'Not a valid review.'
    elif prediction[0] == 'positive':
        return 'True'
    else:
        return 'Flase'

if __name__ == '__main__':
    app.run(debug=True)
    




