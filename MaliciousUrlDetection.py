# -*- coding: utf-8 -*-
"""
Created on Mon July 29 17:30:50 2020
@author: tahsin.asif
"""

from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import os
import json
# Importing dependencies
from urllib.parse import urlparse
from tld import get_tld
import logging

# Postman input ---{"data":"https://zyxytr.com/acompanhamento/"}
# post  ---- params - http://localhost:8086/predict

json_obj = ''
obj = ''

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    print('json:', json_)
    logger.info('json_ %s'%json_)
    # print('json_g---->',json_g)
    json_obj = json.dumps(json_)
    print('json_obj', json_['data'])
    logger.info('json_ %s'%json_obj)
    number_reviews_json = len(json_obj)  # Calculating the number of reviews  json_obj.encode("utf-8"))
    print('Length of _json', number_reviews_json)
    logger.info('Length of _json %s'% number_reviews_json)
    # testing = {'Url':https://zyxytr.com/acompanhamento}
    url_test = pd.DataFrame([json_])
    print('url_test item data---->', url_test['data'])
    logger.info('url_test item data----> %s'% str(url_test['data']))
    # Hostname Length
    url_test['hostname_length'] = len(urlparse(json_['data']).netloc)
    print('hostname_length------------>', url_test['hostname_length'])
    logger.info('hostname_length----> %s'% str(url_test['hostname_length']))

    # fd length
    def fd_length(url):
        urlpath = urlparse(url).path
        try:
            return len(urlpath.split('/')[1])
        except:
            return 0

    url_test['fd_length'] = fd_length(json_['data'])
    print('fd_length====================>>>', url_test['fd_length'])
    logger.info('fd_length---->%s'% str(url_test['fd_length']))

    # tld length
    # Length of Top Level Domain
    url_test['tld'] = get_tld(json_['data'], fail_silently=True)
    print('tld_name====================>>>', url_test['tld'])
    logger.info('tld_name----> %s'% str(url_test['tld']))

    def tld_length(tld):
        try:
            return len(tld)
        except:
            return -1

    url_test['tld_length'] = tld_length(url_test['tld'])
    print('tld Length------------------>', url_test['tld_length'])
    logger.info('tld Length----> %s'% str(url_test['tld_length']))
    url_test['count-'] = json_['data'].count('-')
    print('Count---------------->', url_test['count-'])
    logger.info('Count- ---> %s'% str(url_test['count-']))
    url_test['count@'] = json_['data'].count('@')
    print('Count-@---------------->',(url_test['count@']))
    logger.info('Count-@ ---> %s'%  url_test['count@'])
    url_test['count?'] = json_['data'].count('?')
    print('Count?---------------->',(url_test['count?']))
    logger.info('Count-? --->%s'%  url_test['count?'])
    url_test['count%'] = json_['data'].count('%')
    print('Count%---------------->', url_test['count%'])
    logger.info('Count ---> %s'% str(url_test['count%']))
    url_test['count.'] = json_['data'].count('.')
    print('Count..---------------->', url_test['count.'])
    logger.info('Coun.. --->%s'% str(url_test['count.']))
    url_test['count='] = json_['data'].count('=')
    print('Count=---------------->', url_test['count='])
    logger.info('Cout= ---> %s'% str(url_test['count=']))
    url_test['count-http'] = json_['data'].count('http')
    print('Count-http---------------->', url_test['count-http'])
    logger.info('Cout-http %s'% str(url_test['count-http']))
    url_test['count-https'] = json_['data'].count('https')
    print('Count-https---------------->', url_test['count-https'])
    logger.info('Cout-https %s'% str(url_test['count-https']))
    url_test['count-www'] = json_['data'].count('www')
    print('Count-www---------------->', url_test['count-www'])
    logger.info('Cout-www %s'% str(url_test['count-www']))

    def digit_count(url):
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
            return digits

    url_test['count-digits'] = digit_count(json_['data'])
    logger.info('count-digits %s' % str(url_test['count-digits']))

    def letter_count(url):
        letters = 0
        for i in url:
            if i.isalpha():
                letters = letters + 1
            return letters

    url_test['count-letters'] = letter_count(json_['data'])
    logger.debug('count-letters %s' % str(url_test['count-letters']))

    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')


    url_test['count_dir'] = no_of_dir(json_['data'])
    print("url_test['count_dir']============>",url_test['count_dir'])
    logger.debug('count_dir %s' % str(url_test['count_dir']))

    import re
    # Use of IP or not in domain
    def having_ip_address(url):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  # IPv4 in hexadecimal
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
        if match:
            # print match.group()
            return -1
        else:
            # print 'No matching pattern found'
            return 1

    # url_test['use_of_ip'] = url_test.apply(lambda i: having_ip_address(i))
    url_test['use_of_ip'] = having_ip_address(json_['data'])
    print("Use OF IP=============>",url_test['use_of_ip'])
    logger.debug('Use OF IP %s' % str(url_test['use_of_ip']))

    # Predictor Variables
    x = url_test[['hostname_length',
                  'fd_length', 'tld_length', 'count-', 'count@', 'count?',
                  'count%', 'count=', 'count-http', 'count-https', 'count-www', 'count-digits',
                  'count-letters', 'count_dir', 'use_of_ip']]


    print('query_df_array::----->', x)
    prediction = log_estimator.predict(x)
    print('Predicted Value--->', prediction)
    logger.debug('Predicted Value %s' % str(prediction))
    logger.info('Predicted Value %s' % str(prediction))
    return jsonify(pd.Series(prediction).to_json(orient='values'))


cleanHeadlines_request = []
#MODEL_DIR = '............../Asif/AI/URLDetection/'
MODEL_FILE = 'log_model-v1.pkl'
#MODEL_FILE = 'final_estimator-v1.pkl'
# MODEL_FILE = 'rfc_model-v1.pkl'
if __name__ == '__main__':
    logging.basicConfig(filename="MaliciouusUrlDetection.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    #os.chdir(MODEL_DIR)
    log_estimator = joblib.load(MODEL_FILE)
    app.run(port=8086)
