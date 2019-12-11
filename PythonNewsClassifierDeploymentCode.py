# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:51:40 2019

@author: tahsin.asif
"""
#pip install flask
from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import os
import sklearn
from nltk.corpus import stopwords
import json
from django.http import JsonResponse
import re



json_obj = ''
obj =''

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
     json_= request.json
     print ('json:',json_)
    # print('json_g---->',json_g)
     json_obj = json.dumps(json_)
     print('json_obj',json_obj)
     number_reviews_json = len(json_obj) #Calculating the number of reviews  json_obj.encode("utf-8"))
     print ('number_reviews_json',number_reviews_json)
     vectorize = sklearn.feature_extraction.text.CountVectorizer(analyzer = "word")
     #cleanHeadlines_test =  get_cleanHeadlines_test(number_reviews_json)
     for i in range(0,number_reviews_json):
         headlines_onlyletters = re.sub("[^a-zA-Z]", " ",json_obj) #Remove everything other than letters     
         words = headlines_onlyletters.lower().split() #Convert to lower case, split into individual words    
        # stops = set(stopwords.words("english"))  #Convert the stopwords to a set for improvised performance                 
        # meaningful_words = [w.lower() for w in words if not w in stops]   #Removing stopwords
        # print('meaningful_words---->',meaningful_words)
         cleanHeadlines_request.append(words) 
         #print('cleanHeadlines_request',cleanHeadlines_request)
        # print('cleanHeadlines_test:---->',cleanHeadlines_test)
     vectorize._validate_vocabulary()
     bagOfWords_json = vectorize.fit_transform(words)
     #bagOfWords_json1 = vectorize.transform(words)
     print('bagOfWords_json---->',bagOfWords_json)
     X_train1 = bagOfWords_json.toarray()
    # query_df = pd.DataFrame(json_,index=[0])
     #query_df_array =  query_df.to_numpy()
     #query_df_array= numpy.asarray(query_df)
     print('query_df_array::----->',X_train1)
     prediction = dt_estimator.predict(X_train1)
     print('Predicted Value;--->',prediction)
     return jsonify (pd.Series(prediction).to_json(orient='values'))
 #({'prediction': (prediction)})
 



#def get_cleanHeadlines_test(number_reviews_json):
 #   for i in range(0,number_reviews_json):
       # cleanHeadline = get_words(json_obj) #Processing the data and getting words with no special characters, numbers or html tags
  #      headlines_onlyletters = re.sub("[^a-zA-Z]", " ",json_obj) #Remove everything other than letters     
   #     words = headlines_onlyletters.lower().split() #Convert to lower case, split into individual words    
    #    stops = set(stopwords.words("english"))  #Convert the stopwords to a set for improvised performance                 
     #   meaningful_words = [w for w in words if not w in stops]   #Removing stopwords
      #  print('meaningful_words---->',meaningful_words)
       # cleanHeadlines_request.append( meaningful_words ) 
        #print('cleanHeadlines_request',cleanHeadlines_request)
        #return (cleanHeadlines_request)   
    
#def get_words( headlines ):   
 #   print('json_obj---->',json_obj)            
  #  headlines_onlyletters = re.sub("[^a-zA-Z]", " ",headlines) #Remove everything other than letters     
   # words = headlines_onlyletters.lower().split() #Convert to lower case, split into individual words    
    #stops = set(stopwords.words("english"))  #Convert the stopwords to a set for improvised performance                 
    #meaningful_words = [w for w in words if not w in stops]   #Removing stopwords
    #return( " ".join( meaningful_words )) #Joining the words

cleanHeadlines_request= []    
 
MODEL_DIR = 'C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/UCI_NEWS'
MODEL_FILE = 'Lr-v1.pkl'
if __name__ == '__main__':
     os.chdir(MODEL_DIR)
     dt_estimator = joblib.load(MODEL_FILE)
     app.run(port=8086)
