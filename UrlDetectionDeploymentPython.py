# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:30:50 2020

@author: tahsin.asif
"""

from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import os
import json
#Importing dependencies
from urllib.parse import urlparse
from tld import get_tld
# Postman input ---{"data":"https://zyxytr.com/acompanhamento/"}
# post  ---- params - http://localhost:8086/predict

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
     print ('Length of _json',number_reviews_json)
     ############################
     #testing = {'Url':https://zyxytr.com/acompanhamento}
     url_test = pd.DataFrame([json_])
    # url_test = pd.DataFrame('https://zyxytr.com/acompanhamento')
     print('url_test item data---->',url_test['data'])
     #Hostname Length
    # url_test['hostname_length'] = json_.apply(lambda i: len(urlparse(i).netloc))
     url_test['hostname_length'] = len(urlparse(json_obj).netloc)
     print ('------------>',url_test['hostname_length'] )
     # fd length 
     def fd_length(url):
         urlpath= urlparse(url).path
         try:
             return len(urlpath.split('/')[1])
         except:
             return 0

    # url_test['fd_length'] = url_test.apply(lambda i: fd_length(i))
     url_test['fd_length'] = fd_length(json_obj)
     print('====================>>>',url_test['fd_length'])
     
     # tld length 
     #Length of Top Level Domain
     #url_test['tld'] = url_test.apply(lambda i: get_tld(i,fail_silently=True))
     url_test['tld'] = get_tld(json_obj,fail_silently=True)
     def tld_length(tld):
         try:
             return len(tld)
         except:
             return -1
     print('====================>>>',url_test['tld']) 
            
         
     #url_test['tld_length'] = url_test['tld'].apply(lambda i: tld_length(i))
     url_test['tld_length'] = tld_length(url_test['tld'])
     print('tld Length------------------>',url_test['tld_length'])       

     url_test['count-'] = json_obj.count('-')
     print ('Count---------------->',url_test['count-'])
    # url_test['count-'] = url_test.apply(lambda i: i.count('-'))
     url_test['count@'] = 'https://zyxytr.com/acompanhamento@/'.count('@')
     print ('Count-@---------------->',url_test['count@'])
     #url_test['count@'] = url_test.apply(lambda i: i.count('@'))
     url_test['count?'] = json_obj.count('?')
     print ('Count?---------------->',url_test['count?'])
     #url_test['count?'] = url_test.apply(lambda i: i.count('?'))
     url_test['count%'] = json_obj.count('%')
     print ('Count%---------------->',url_test['count%'])
    # url_test['count%'] = url_test.apply(lambda i: i.count('%'))
     url_test['count.'] = json_obj.count('.')
     print ('Count..---------------->',url_test['count.'])
    # url_test['count.'] = url_test.apply(lambda i: i.count('.'))
     url_test['count='] = json_obj.count('=')
     print ('Count=---------------->',url_test['count='])
    # url_test['count='] = url_test.apply(lambda i: i.count('='))
     url_test['count-http'] = json_obj.count('http')
     print ('Count-http---------------->',url_test['count-http'])
    # url_test['count-http'] = url_test.apply(lambda i : i.count('http'))
     url_test['count-https'] = json_obj.count('https')
     print ('Count-https---------------->',url_test['count-https'])
    # url_test['count-https'] = url_test.apply(lambda i : i.count('https'))
     url_test['count-www'] = json_obj.count('www')
     print ('Count-www---------------->',url_test['count-www'])
    # url_test['count-www'] = url_test.apply(lambda i: i.count('www'))
     
     def digit_count(url):
         digits = 0
         for i in url:
             if i.isnumeric():
                 digits = digits + 1
             return digits
    # url_test['count-digits']= url_test.apply(lambda i: digit_count(i))
     url_test['count-digits']= digit_count('https://zyxytr.com/acompanhamento')
     

     #url_test['count-letters'] = url_test.apply(lambda i: letter_count(i))
    
     def letter_count(url):
         letters = 0
         for i in url:
             if i.isalpha():
                 letters = letters + 1
             return letters   
     url_test['count-letters'] = letter_count('https://zyxytr.com/acompanhamento')
      
      
     def no_of_dir(url):
         urldir = urlparse(url).path
         return urldir.count('/')  
     
    # url_test['count_dir'] = url_test.apply(lambda i: no_of_dir(i))
     url_test['count_dir'] = no_of_dir('https://zyxytr.com/acompanhamento')
     
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
        
     #url_test['use_of_ip'] = url_test.apply(lambda i: having_ip_address(i))
     url_test['use_of_ip'] = having_ip_address('https://zyxytr.com/acompanhamento')
     print(url_test['use_of_ip'] )   
   
     #Predictor Variables
     x = url_test[['hostname_length',
        'fd_length', 'tld_length', 'count-', 'count@', 'count?',
       'count%', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]
     ############################
     
     print('query_df_array::----->',x)
     prediction = log_estimator.predict(x)
     print('Predicted Value;--->',prediction)
     return jsonify (pd.Series(prediction).to_json(orient='values'))
 

cleanHeadlines_request= []    
 MODEL_DIR = 'C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/URLDetection/'
#MODEL_FILE = 'log_model-v1.pkl'
MODEL_FILE ='final_estimator-v1.pkl'
#MODEL_FILE = 'rfc_model-v1.pkl'
if __name__ == '__main__':
     os.chdir(MODEL_DIR)
     log_estimator = joblib.load(MODEL_FILE)
     app.run(port=8086)
