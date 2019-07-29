#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:34:59 2019

@author: rejolut404
"""
import pickle
import nltk
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# using the SQLite Table to read data.
connect = sqlite3.connect('/home/ml_rejolut/Desktop/ML/Sentiment_Analysis_Python_NLP/amazon-fine-food-reviews/database.sqlite')

#filtering only positive and negative reviews i.e. 
#not taking into consideration those reviews with Score=3
filtered_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3 LIMIT 568454""", connect) 

# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.
def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'

#changing reviews with score less than 3 to be positive and vice-versa
actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition) 
filtered_data['Score'] = positiveNegative

filtered_data.shape #looking at the number of attributes and size of the data
filtered_data.head()

#as we can see 42,640 rows are eliminated which had neutral review
print(filtered_data)

#show the duplicated rows of the data it returns dtype = boolean.
duplicated_Data = filtered_data.duplicated()
print(duplicated_Data)

#remove duplicate data for unbiased result of the analyzed data.
display = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3 AND UserId = "AR5J8UI46CURR" ORDER BY ProductId LIMIT 568454""", connect) 
print(display)

#Sorting data according to ProductId in ascending order
sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

#Deduplication of entries
final = sorted_data.drop_duplicates(subset = {"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
a = final.shape
print(a)

#Checking to see how much % of data still remains
b = (final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
print(b)

#here these two product ids have value of numerator greater than value of denominator so we are deleting it from the table.
display1 = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score !=3 AND Id = 44737 OR Id = 64422 ORDER BY ProductId LIMIT 568454""", connect)
print(display1)

#the value of numerator should be less than or equal to the denominator score.
final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]

#lets see the final shape of our data
c = final.shape
print(c)

#see the number of positive and negative number of words count in our data.
d = final['Score'].value_counts()
print(d)
'''
Hence in the Preprocessing phase we do the following in the order below:-

    1.Begin by removing the html tags
    2.Remove any punctuations or limited set of special characters like , or . or # etc.
    3.Check if the word is made up of english letters and is not alpha-numeric
    4.Check to see if the length of the word is greater than 2 (as it was researched that there is no adjective in 2-letters)
    5.Convert the word to lowercase
    6.Remove Stopwords
    7.Finally Snowball Stemming the word (it was obsereved to be better than Porter Stemming)
After which we collect the words used to describe positive and negative reviews
'''
#remove html tags
import re   #regular expression.
i = 0
for sent in final['Text'].values:
    if(len(re.findall('<.*?>', sent))):
        print(i)
        print(sent)
        break
    i += 1

import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stop = set(stopwords.words('english'))  #set of stopwords.
#note the snowball stemmer is used for the information retrieval system.
snow = nltk.stem.SnowballStemmer('english')  #initialising the snowball stemmer.

#function to clean the word of any html-tags
def cleanhtml(sentence):
    cleanr = re.compile('<.*?>')    #look fr the all the html tags
    cleantext = re.sub(cleanr, ' ', sentence)   #substitute the whitespace for the tags found in the text.
    return cleantext

#function to clean the word of any punctuation or special characters
def cleanpunc(sentence): 
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned
print(stop)   #print the stopwords of the doc.
print('************************************')

#Code for implementing step-by-step the checks mentioned in the pre-processing phase
#Note this code takes a while to run as it needs to run on 500k sentences.
i=0
str1=' '
final_string=[]
# store words from +ve reviews here
all_positive_words=[] 
# store words from -ve reviews here.
all_negative_words=[] 
s=''
for sent in final['Text'].values:
    filtered_sentence=[]
    #print(sent);
    sent=cleanhtml(sent)    # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(snow.stem(cleaned_words.lower())).encode('utf8')     #unicode transformation formation of 8-bit values.
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 'positive': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue 
    #final string of cleaned words
    str1 = b" ".join(filtered_sentence) 
    final_string.append(str1)
    i+=1
    
#adding a column of CleanedText which displays the data after pre-processing of the review 
final['CleanedText'] = final_string

#below the processed review can be seen in the CleanedText Column 
final.head(3)

# store final table into an SQlLite table for future.
conn = sqlite3.connect('/home/ml_rejolut/Desktop/ML/Sentiment_Analysis_Python_NLP/final.sqlite')
c=conn.cursor()  #allows python code to execute sql command in databse session. 
conn.text_factory = str  #again a sql command 
final.to_sql('Reviews', conn, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)

con = sqlite3.connect("/home/ml_rejolut/Desktop/ML/Sentiment_Analysis_Python_NLP/final.sqlite")
cleaned_data = pd.read_sql_query("select * from Reviews LIMIT 568454", con)

d = cleaned_data.shape
print(d)

cleaned_data.head()

cleaned_data['Score'].value_counts()

# To randomly sample 500k points from both class
data_pos = cleaned_data[cleaned_data["Score"] == "positive"].sample(n = 200000, replace = True)
data_neg = cleaned_data[cleaned_data["Score"] == "negative"].sample(n = 200000, replace = True)
final_500k = pd.concat([data_pos, data_neg])
final_500k.shape

# Sort data based on time
final_500k["Time"] = pd.to_datetime(final_500k["Time"], unit = "s")
final_500k = final_500k.sort_values(by = "Time")
final_500k.head()

# Bag of Words.
# Naive-Bayes Algorithm for text classification.
# alpha is the hyperparameter i.e. parameter that controls the model flow itself.
def naive_bayes(X_train, y_train):
    
    #np.arange(start or optional, stop number, step, dtype:None)
    alpha_values = np.arange(1, 400, 0.5)
    
    # This empty list will hold the cross-validation score.
    crossvalidation_scores = []
    
    #cross-validation is resampling process to evaluate ML models on limited data sample.
    # perform 10-fold cross validation
    for alpha in alpha_values:
        multinomialnb = MultinomialNB(alpha = alpha)
        #x is training data to be fit and whereas y is the target variable. 
        scores = cross_val_score(multinomialnb, X_train, y_train, cv = 10, scoring = 'accuracy')
        #append mean of the scores to the empty list declared cv_scores
        crossvalidation_scores.append(scores.mean())

    # changing to misclassification error
    MCE = [1 - x for x in crossvalidation_scores]

    # determining the optimum value of alpha
    optimum_alpha = alpha_values[MCE.index(min(MCE))]
    print('\nThe optimal number of alpha is %d.' % optimum_alpha)

    # plot misclassification error vs alpha
    plt.plot(alpha_values, MCE, marker = '*') 
    plt.title("Misclassification Error vs Alpha")
    plt.xlabel('Value of Alpha')
    plt.ylabel('Misclassification Error')
    plt.show()
    return optimum_alpha

# 100k data which will use to train model after vectorization
X = final_500k["CleanedText"]
print("shape of X:", X.shape)

# class label
y = final_500k["Score"]
print("shape of y:", y.shape)

from sklearn.model_selection import train_test_split
#training data is 70% and testing data is 30%
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
print(X_train.shape, y_train.shape, x_test.shape)

from sklearn.feature_extraction.text import CountVectorizer
bagofwords = CountVectorizer()    #count vectorizer builds a vocabulary of known words in form of matrix. 
X_train = bagofwords.fit_transform(X_train)
print(X_train)

# Test Vectorizor
x_test = bagofwords.transform(x_test)
print(x_test.shape)

# To choose optimal_alpha using cross validation
optimum_alpha_bagofwords = naive_bayes(X_train, y_train)
print(optimum_alpha_bagofwords)

# instantiate learning model alpha = optimal_alpha
nb_optimum =  MultinomialNB(alpha = optimum_alpha_bagofwords)

# fitting the model
nb_optimum.fit(X_train, y_train)
#nb_optimal.fit(bow_data, y_train)

# predict the response
pred = nb_optimum.predict(x_test)

# To get all the features name 
bagofwords_features = bagofwords.get_feature_names()

# To count feature for each class while fitting the model
feature_count = nb_optimum.feature_count_
print(feature_count.shape)

# Accuracy on train data
training_accuracy_bagofwords = nb_optimum.score(X_train, y_train)
print("Train accuracy", training_accuracy_bagofwords)

# Error on training data
train_error_bow = 1-training_accuracy_bagofwords 
print("Train Error %f%%" % (train_error_bow))

#evaluate accuracy on test data.
accuracy_bagofwords = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the naive bayes classifier for alpha = %d is %f%%' % (optimum_alpha_bagofwords, accuracy_bagofwords))

#confusion matrix.
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, pred)
print(conf_matrix)

# plottig confusion matrix to describe the performance of classifier.
import seaborn as sns   #this is the datavisualization library based on matpltotlib
class_label = ["negative", "positive"]
df_conf_matrix = pd.DataFrame(conf_matrix, index = class_label, columns = class_label)
sns.heatmap(df_conf_matrix, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

#Main classification report.
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

#save the binary classifier model.
save_classifier = open('/home/ml_rejolut/Desktop/ML/Sentiment_Analysis_Python_NLP/sentiment_analysis.pickle', 'wb')
pickle.dump(nb_optimum, save_classifier)  # first parameter is what you are dumping and second param is where you are dumping it.
save_classifier.close()

#save bag of words values to pickle.
save_classifier1 = open('/home/ml_rejolut/Desktop/ML/Sentiment_Analysis_Python_NLP/bow.pickle', 'wb')
pickle.dump(bagofwords, save_classifier1)
save_classifier1.close()


def predictReview():
    newReview = input('Type the review: ')
    
    if newReview == ' ' :
        print('Not a valid review')
        
    else:
        new_review = bagofwords.transform([newReview]).toarray()  
        prediction =  nb_optimum.predict(new_review)
        print(prediction)
        
        if prediction[0] == 'positive':
            print('Positive Review')
        else:
            print('Negative Review')

predictReview()

# Tf-IDF is used for understanding how important a word is in a document by giving a weight to a word.
# An algortihm cannot understand words so it is neccessary to convert the raw words to numeric values.

#initialize the cleaned data for tf-idf
X = final_500k['CleanedText']

#get the target from the cleaned data i.e. the "Score" 
y = final_500k['Score']

#split the cleaned text data in train and test size of 70:30
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
print(X_train.shape, x_test.shape, y_train.shape, y_test.shape)

    




