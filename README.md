1.  Import Sql database using sqlite3 library python.
2. Seperate data based on Score.
3. assign label to data positive negative depending on the score.
4. Do text cleaning removing punctuation, removing html tags, removing stopwords, stemming, removing alphanumeric words.
5. parse the cleaned data to numeric value such as array.
6. apply cleaned data to the naive bayes classifier.
7. make count vectorizer or bag of words to store the keywords.
8. split data into train and test of 70 and 30 %.
9. train the data using the NB classifier.
10. Save the classified model using pickle which is again a python library.
11. Predict the data from the new derived final.sqlite(cleaned data after preprocessing) file which was generated after training.
12. Integrate with flask api and check the output with postman.