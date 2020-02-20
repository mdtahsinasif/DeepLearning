# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:34:49 2020

@author: tahsin.asif
"""

import pandas as pd
df = pd.read_csv("C:/Users/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/CveScorePrediction/CveScore.csv",encoding='ISO-8859-1')
df.head()



from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.description).toarray()
labels = df.baseScore
features.shape

print(df.columns)


  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['description'], df['baseScore'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(y_train)
y_train= y_train.fillna(7)
clf = MultinomialNB().fit(X_train_tfidf, y_train)  
##################################################


import os
from sklearn.externals import joblib
path = 'C:/Users/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/CveScorePrediction/'
#copy the model to pkl file and keep the model file at required server location
joblib.dump(clf,os.path.join(path, 'clf-v1.pkl') )
#joblib.dump(rfc,os.path.join(path, 'rfc_model-v1.pkl') )

#cross check the dumped model with load
classifier_loaded = joblib.load(os.path.join(path, 'clf-v1.pkl') )

#################################################





print(clf.predict(count_vect.transform(["The vMX Series software uses a predictable IP ID Sequence Number. This leaves the system as well as clients connecting through the device susceptible to a family of attacks which rely on the use of predictable IP ID sequence numbers as their base method of attack. This issue was found during internal product security testing. Affected releases are Juniper Networks Junos OS: 15.1 versions prior to 15.1F5 on vMX Series."])))

#############################
#Ensemble Model
#hard voting ensemble
from sklearn import tree, model_selection, preprocessing, ensemble, feature_selection, neighbors, naive_bayes

dt_estimator = tree.DecisionTreeClassifier(random_state=100)

knn_estimator = neighbors.KNeighborsClassifier()

nb_estimator = naive_bayes.GaussianNB()

ada_estimator = ensemble.AdaBoostClassifier(random_state=100)

rf_estimator = ensemble.RandomForestClassifier(random_state=100)

estimators = [('dt', dt_estimator), ('knn', knn_estimator), ('nb', nb_estimator) , ('rf', rf_estimator), ('ada', ada_estimator)]
hvoting_estimator = ensemble.VotingClassifier(estimators)
voting_grid = {'dt__max_depth':[3,7], 'knn__n_neighbors':[3,5,7], 'ada__n_estimators':[50, 100], 'rf__n_estimators':[50, 100] }
hvoting_grid_estimator = model_selection.GridSearchCV(hvoting_estimator, voting_grid, scoring='accuracy', cv=10, return_train_score=True)
hvoting_grid_estimator.fit(X_train_tfidf.toarray(), y_train)
print(hvoting_grid_estimator.best_score_)
print(hvoting_grid_estimator.best_params_)
final_estimator = hvoting_grid_estimator.best_estimator_
print(final_estimator.estimators_)
print(final_estimator.score(X_train_tfidf, y_train))

joblib.dump(final_estimator,os.path.join(path, 'final_estimator-v1.pkl') )

#cross check the dumped model with load
classifier_loaded = joblib.load(os.path.join(path, 'final_estimator-v1.pkl') )

#############################
