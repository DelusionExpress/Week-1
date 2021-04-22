import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
X_test = []
y_test = []
with open('preprocess_data/test_data.txt') as test_data_file:
    for line in test_data_file:
        words = line.strip().split()
        y_test.append(words[0])
        X_test.append(' '.join(words[1:]))
encoder = LabelEncoder()
encoder.classes_ = np.load('trained_model/classes.npy')
y_test = encoder.transform(y_test)
# Naive
model = pickle.load(open("trained_model/nb_bow.pkl", 'rb'))
y_predict = model.predict(X_test)
print('Naive Bayes-BoW, Accuracy =', np.mean(y_predict == y_test))
model = pickle.load(open("trained_model/nb_tfidf.pkl","rb"))
y_predict = model.predict(X_test)
print('Naive Bayes-TFIDF, Accuracy =', np.mean(y_predict == y_test))

# SVM
model = pickle.load(open("trained_model/svm_bow.pkl", 'rb'))
y_predict = model.predict(X_test)
print('SVM-BoW, Accuracy =', np.mean(y_predict == y_test))
model = pickle.load(open("trained_model/svm_tfidf.pkl","rb"))
y_predict = model.predict(X_test)
print('SVM-TFIDF, Accuracy =', np.mean(y_predict == y_test))

# Random Forrest
model = pickle.load(open("trained_model/rf_bow.pkl", 'rb'))
y_predict = model.predict(X_test)
print('Random Forrest-BoW, Accuracy =', np.mean(y_predict == y_test))
model = pickle.load(open("trained_model/rf_tfidf.pkl","rb"))
y_predict = model.predict(X_test)
print('Random Forrest-TFIDF, Accuracy =', np.mean(y_predict == y_test))

