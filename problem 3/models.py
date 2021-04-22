import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

contents = []
labels  =  []
with open('preprocess_data/train_data.txt') as train_data_file:
    for line in train_data_file:
        words = line.strip().split()
        labels.append(words[0])
        contents.append(' '.join(words[1:]))
X_train,X_validate ,y_train,y_validate = train_test_split(contents,labels,test_size=0.2,random_state=42)


with open('preprocess_data/train.txt', 'w') as fp:
    for x, y in zip(X_train, y_train):
        fp.write('{} {}\n'.format(y, x))
 
with open('preprocess_data/validate.txt', 'w') as fp:
    for x, y in zip(X_validate, y_validate):
        fp.write('{} {}\n'.format(y, x))

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
np.save('trained_model/classes.npy', label_encoder.classes_)

nb_bow = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), 
                     ('clf', MultinomialNB())
                    ])
nb_tfidf= Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultinomialNB())])

svm_bow = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), 
                     ('clf', SVC())
                    ])
svm_tfidf= Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', SVC())])
rf_bow = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), 
                     ('clf',RandomForestClassifier(n_estimators=20))
                    ])
rf_tfidf= Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', RandomForestClassifier(n_estimators=20))])
nb_bow.fit(X_train,y_train)
nb_tfidf.fit(X_train, y_train)
svm_bow.fit(X_train,y_train)
svm_tfidf.fit(X_train, y_train)
rf_bow.fit(X_train,y_train)
rf_tfidf.fit(X_train, y_train)

pickle.dump(nb_tfidf, open("trained_model/nb_tfidf.pkl","wb"))
pickle.dump(nb_bow, open("trained_model/nb_bow.pkl", 'wb'))
pickle.dump(svm_tfidf, open("trained_model/svm_tfidf.pkl", 'wb'))
pickle.dump(svm_bow, open("trained_model/svm_bow.pkl", 'wb'))
pickle.dump(rf_tfidf, open("trained_model/rf_tfidf.pkl", 'wb'))
pickle.dump(rf_bow, open("trained_model/rf_bow.pkl", 'wb'))