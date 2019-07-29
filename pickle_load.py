import pickle
import sqlite3
import pandas as pd


con = sqlite3.connect("/home/ml_rejolut/Desktop/ML/Sentiment_Analysis_Python_NLP/final.sqlite")
cleaned_data = pd.read_sql_query("select * from Reviews LIMIT 364000", con)
    
data_pos = cleaned_data[cleaned_data["Score"] == "positive"].sample(n = 160000, replace = True)
data_neg = cleaned_data[cleaned_data["Score"] == "negative"].sample(n = 160000, replace = True)
final_data = pd.concat([data_pos, data_neg])

load_pickle_file = open("/home/ml_rejolut/Desktop/ML/Sentiment_Analysis_Python_NLP/sentiment_analysis.pickle", "rb")
classifier = pickle.load(load_pickle_file)
load_pickle_file.close()
    
load_pickle_file1 = open("/home/ml_rejolut/Desktop/ML/Sentiment_Analysis_Python_NLP/bow.pickle", "rb")
classifier1 = pickle.load(load_pickle_file1)
load_pickle_file1.close()
          

class_label = ["negative", "positive"]

def predict():
    newReview = input('Type the review: ')
    if newReview == ' ' :
            print('Not a valid review')
    else:
        new_review = classifier1.transform([newReview]).toarray()
        prediction =  classifier.predict(new_review)
        print(prediction)
        
        if prediction[0] == 'positive':
            print('Positive Review')
        else:
            print('Negative Review')
            
    
predict()
    




