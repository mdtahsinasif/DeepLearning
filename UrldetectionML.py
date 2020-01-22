# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:53:46 2020

@author: tahsin.asif
"""

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import os
#print(os.listdir("../input"))
#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/URLDetection/")

urldata = pd.read_csv("C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/URLDetection/urldata.csv")
urldata.head()

#Removing the unnamed columns as it is not necesary.
urldata = urldata.drop('Unnamed: 0',axis=1)
urldata.shape

urldata.info()

urldata.isnull().sum()

#pip install tld

#Importing dependencies
from urllib.parse import urlparse
from tld import get_tld
import os.path

print('urldata----------->', urldata['url'])
urldata['url_length'] = urldata['url'].apply(lambda i: len(str(i)))
print(urldata['url_length'] )


#Hostname Length
urldata['hostname_length'] = urldata['url'].apply(lambda i: len(urlparse(i).netloc))
print (urldata['hostname_length'] )

#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

urldata['fd_length'] = urldata['url'].apply(lambda i: fd_length(i))

print(urldata['fd_length'])

#Length of Top Level Domain
urldata['tld'] = urldata['url'].apply(lambda i: get_tld(i,fail_silently=True))
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

urldata['tld_length'] = urldata['tld'].apply(lambda i: tld_length(i))
print(urldata['tld_length']) 

urldata.head()

urldata = urldata.drop("tld",1)
urldata.head()

urldata['count-'] = urldata['url'].apply(lambda i: i.count('-'))
urldata['count@'] = urldata['url'].apply(lambda i: i.count('@'))
urldata['count?'] = urldata['url'].apply(lambda i: i.count('?'))
urldata['count%'] = urldata['url'].apply(lambda i: i.count('%'))
urldata['count.'] = urldata['url'].apply(lambda i: i.count('.'))
urldata['count='] = urldata['url'].apply(lambda i: i.count('='))
urldata['count-http'] = urldata['url'].apply(lambda i : i.count('http'))
urldata['count-https'] = urldata['url'].apply(lambda i : i.count('https'))
urldata['count-www'] = urldata['url'].apply(lambda i: i.count('www'))

def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits
urldata['count-digits']= urldata['url'].apply(lambda i: digit_count(i))

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters
urldata['count-letters']= urldata['url'].apply(lambda i: letter_count(i))

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')
urldata['count_dir'] = urldata['url'].apply(lambda i: no_of_dir(i))

urldata.head()

import re
#Use of IP or not in domain
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return -1
    else:
        # print 'No matching pattern found'
        return 1
urldata['use_of_ip'] = urldata['url'].apply(lambda i: having_ip_address(i))
print(urldata['use_of_ip'] )


def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return -1
    else:
        return 1
    
    
urldata['short_url'] = urldata['url'].apply(lambda i: shortening_service(i))    

urldata.head()

#Heatmap
corrmat = urldata.corr()
f, ax = plt.subplots(figsize=(25,19))
sns.heatmap(corrmat, square=True, annot = True, annot_kws={'size':10})

plt.figure(figsize=(15,5))
sns.countplot(x='label',data=urldata)
plt.title("Count Of URLs",fontsize=20)
plt.xlabel("Type Of URLs",fontsize=18)
plt.ylabel("Number Of URLs",fontsize=18)

print("Percent Of Malicious URLs:{:.2f} %".format(len(urldata[urldata['label']=='malicious'])/len(urldata['label'])*100))
print("Percent Of Benign URLs:{:.2f} %".format(len(urldata[urldata['label']=='benign'])/len(urldata['label'])*100))

plt.figure(figsize=(20,5))
plt.hist(urldata['url_length'],bins=50,color='LightBlue')
plt.title("URL-Length",fontsize=20)
plt.xlabel("Url-Length",fontsize=18)
plt.ylabel("Number Of Urls",fontsize=18)
plt.ylim(0,1000)

plt.figure(figsize=(20,5))
plt.hist(urldata['hostname_length'],bins=50,color='Lightgreen')
plt.title("Hostname-Length",fontsize=20)
plt.xlabel("Length Of Hostname",fontsize=18)
plt.ylabel("Number Of Urls",fontsize=18)
plt.ylim(0,1000)

plt.figure(figsize=(20,5))
plt.hist(urldata['tld_length'],bins=50,color='Lightgreen')
plt.title("TLD-Length",fontsize=20)
plt.xlabel("Length Of TLD",fontsize=18)
plt.ylabel("Number Of Urls",fontsize=18)
plt.ylim(0,1000)

plt.figure(figsize=(15,5))
plt.title("Number Of Directories In Url",fontsize=20)
sns.countplot(x='count_dir',data=urldata)
plt.xlabel("Number Of Directories",fontsize=18)
plt.ylabel("Number Of URLs",fontsize=18)

plt.figure(figsize=(15,5))
plt.title("Use Of IP In Url",fontsize=20)
plt.xlabel("Use Of IP",fontsize=18)

sns.countplot(urldata['use_of_ip'])
plt.ylabel("Number of URLs",fontsize=18)

plt.figure(figsize=(15,5))
plt.title("Use Of IP In Url",fontsize=20)
plt.xlabel("Use Of IP",fontsize=18)
plt.ylabel("Number of URLs",fontsize=18)
sns.countplot(urldata['use_of_ip'],hue='label',data=urldata)
plt.ylabel("Number of URLs",fontsize=18)


plt.figure(figsize=(15,5))
plt.title("Use Of http In Url",fontsize=20)
plt.xlabel("Use Of IP",fontsize=18)
plt.ylim((0,1000))
sns.countplot(urldata['count-http'])
plt.ylabel("Number of URLs",fontsize=18)


plt.figure(figsize=(15,5))
plt.title("Use Of http In Url",fontsize=20)
plt.xlabel("Count Of http",fontsize=18)
plt.ylabel("Number of URLs",fontsize=18)
plt.ylim((0,1000))
sns.countplot(urldata['count-http'],hue='label',data=urldata)
plt.ylabel("Number of URLs",fontsize=18)

plt.figure(figsize=(15,5))
plt.title("Use Of http In Url",fontsize=20)
plt.xlabel("Count Of http",fontsize=18)

sns.countplot(urldata['count-http'],hue='label',data=urldata)

plt.ylabel("Number of URLs",fontsize=18)

plt.figure(figsize=(15,5))
plt.title("Use Of WWW In URL",fontsize=20)
plt.xlabel("Count Of WWW",fontsize=18)
sns.countplot(urldata['count-www'])
plt.ylim(0,1000)
plt.ylabel("Number Of URLs",fontsize=18)


plt.figure(figsize=(15,5))
plt.title("Use Of WWW In URL",fontsize=20)
plt.xlabel("Count Of WWW",fontsize=18)

sns.countplot(urldata['count-www'],hue='label',data=urldata)
plt.ylim(0,1000)
plt.ylabel("Number Of URLs",fontsize=18)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

#Predictor Variables
x = urldata[['hostname_length',
        'fd_length', 'tld_length', 'count-', 'count@', 'count?',
       'count%', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]

#Target Variable
y = urldata['result']

x.shape

y.shape

#Splitting the data into Training and Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)

#Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)

dt_predictions = dt_model.predict(x_test)

accuracy_score(y_test,dt_predictions)

print(confusion_matrix(y_test,dt_predictions))

#Random Forest
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

rfc_predictions = rfc.predict(x_test)
accuracy_score(y_test, rfc_predictions)

print(confusion_matrix(y_test,rfc_predictions))

#Logistic Regression
log_model = LogisticRegression()
log_model.fit(x_train,y_train)

log_predictions = log_model.predict(x_test)
accuracy_score(y_test,log_predictions)

print(confusion_matrix(y_test,log_predictions))


from sklearn.externals import joblib
path = 'C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/URLDetection/'
#copy the model to pkl file and keep the model file at required server location
joblib.dump(log_model,os.path.join(path, 'log_model-v1.pkl') )
joblib.dump(rfc,os.path.join(path, 'rfc_model-v1.pkl') )

#cross check the dumped model with load
classifier_loaded = joblib.load(os.path.join(path, 'log_model-v1.pkl') )
classifier_loaded = joblib.load(os.path.join(path, 'rfc_model-v1.pkl') )

#titanic_test = pd.read_csv("C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/URLDetection/TestingUrlNew.csv")
#X_test = titanic_test.drop(['label'],1)

#titanic_test['result'] = dt_model.predict(X_test)
#titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)
# Hard Voting

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
hvoting_grid_estimator.fit(x_train, y_train)
print(hvoting_grid_estimator.best_score_)
print(hvoting_grid_estimator.best_params_)
final_estimator = hvoting_grid_estimator.best_estimator_
print(final_estimator.estimators_)
print(final_estimator.score(x_train, y_train))

joblib.dump(final_estimator,os.path.join(path, 'final_estimator-v1.pkl') )

#cross check the dumped model with load
classifier_loaded = joblib.load(os.path.join(path, 'final_estimator-v1.pkl') )

# postman input {"data":"https://xrpinvesting.com/wp-includes/js/tinymce/home/login.php"}

# post param - http://localhost:8086/predict
