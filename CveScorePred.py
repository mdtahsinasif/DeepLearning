# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 15:34:49 2020

@author: tahsin.asif
"""

import pandas as pd
from sklearn import tree, model_selection, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import os
from sklearn.externals import joblib
from sklearn import tree, model_selection, preprocessing, ensemble, feature_selection, neighbors, naive_bayes

df = pd.read_csv("C:/Users/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/CveScorePrediction/CveScore.csv",encoding='ISO-8859-1')
df.head(5)
df.dropna(subset = ["baseScore"], inplace=True)  
random_state = 100
kfold = model_selection.StratifiedKFold(n_splits=10)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.description).toarray()
labels = df.baseScore
features.shape

print(df.baseScore)

X_train, X_test, y_train, y_test = train_test_split(df['description'], df['baseScore'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(y_train)

RandomForestRegressor = ensemble.RandomForestRegressor(random_state=100, n_jobs=1, verbose=1, oob_score=True)
RandomForestRegressorAfterFit = RandomForestRegressor.fit(X_train_tfidf, y_train)

path = 'C:/Users/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/CveScorePrediction/'
joblib.dump(RandomForestRegressorAfterFit,os.path.join(path, 'regression_model-v3.pkl') )

#cross check the dumped model with load
classifier_loaded = joblib.load(os.path.join(path, 'regression_model-v3.pkl') )
inputData = input("Please enter the input text::") 
print(classifier_loaded.predict(count_vect.transform([inputData])))
