import logging
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
import time
#pip install langdetect
from langdetect import detect
#pip install google_trans_new
from google_trans_new  import google_translator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
porter=PorterStemmer()

# -*- coding: utf-8 -*-

class BrandImpersonationDb():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    def dbConnection(self):
        try:
            self.logger.info("====== Inside dbConnection Method ========")
            count = 0
            masterDomaindict = {}
            mydoc = collection.find()
            for x in mydoc:
                    originalDomaindict = {}
                    impersonatedDomaindict = {}
                    commonWords = []
                    if ('http' not in str(x['domain'])):
                         imdomainVal = 'https://'+(x['domain'])+'/'
                         imperdomaindict = brandImpersonationDbObj.calculateWord(imdomainVal)
                         impersonatedDomaindict[imdomainVal] = imperdomaindict.get('data')
                         conutryValue = imperdomaindict.get('country')

                         collection.update_one({"country": {"$exists": False}}, {'$set': {"country": conutryValue}})
                         collection.update_one({"pageContent": {"$exists": False}},
                                               {'$set': {"pageContent": imperdomaindict.get('data')}})

                    if (((imperdomaindict.get('data')) is not None)):
                         if ('http' not in str(x['asset_value'])):
                            ordomainVal = 'https://' + x['asset_value']
                            orgdomaindict = brandImpersonationDbObj.calculateWord(ordomainVal)
                            originalDomaindict[ordomainVal] = orgdomaindict.get('data')
                            masterDomaindict[ordomainVal] = orgdomaindict.get('data')
                            countdict = brandImpersonationDbObj.compareWords(impersonatedDomaindict, originalDomaindict)
                            commonWords = str(countdict.get('comwords'))
                            collection.update_one({"_id": x["_id"] },
                                              {'$set': {"matchingContent": (commonWords)}})

                            matchingCountVal = str(countdict.get('matchingCount'))
                            collection.update_one({"_id": x["_id"] },
                                              {'$set': {"matchingCount": (matchingCountVal)}})
                            impersonatedPageCount = len(orgdomaindict.get('data'))
                            matchingContentCount = countdict.get('matchingCount')
                            similarityPercentage = (matchingContentCount/impersonatedPageCount)*100
                            collection.update_one({"_id": x["_id"] },
                                                   {'$set': {"similarityPercentage": str(similarityPercentage)}})
        except Exception as e:
            print(e)
            self.logger.info("====== Preprocessing the data set ========", e)

    def calculateWord(self,text):
        langData = {}
        response = ''
        try:
                response = requests.get(text, headers=headers, verify=False)
                soup = BeautifulSoup(response.text, 'html.parser')
                soupText = (soup.text)
                langdetect = detect(str(soupText))
                langData['country'] = langdetect
                langData['url'] = text
                if langdetect != 'en':
                    soupText = translator.translate((soupText))

                stop_words = set(stopwords.words('english'))
                text = re.sub(r'[^\w\s]', '', soupText)
                token_words = word_tokenize(text)
                stem_sentence = []
                for word in token_words:
                    stem_sentence.append(porter.stem(word))
                stemSentence = []
                for stemsen in stem_sentence:
                    if stemsen not in stop_words:
                        # Find total words in the document
                        lemmatizer = WordNetLemmatizer()
                        text = lemmatizer.lemmatize((stemsen))
                        stemSentence.append(text)
                langData['data'] = stemSentence

        except Exception  as e:
             self.logger.info("====== Inside cont function ========",e)
        return langData

    def compareWords(self,impersonatedDomaindict,originalDomaindict):
        countdict = {}
        commWordList = []
        for kim, vim in impersonatedDomaindict.items():
            for words in vim:
                for k, v in originalDomaindict.items():
                    for word in v:
                        if word.lower() == words.lower():
                            matchingWord = word.lower()
                            commWordList.append((matchingWord))
        countdict['comwords'] = set(commWordList)
        countdict['matchingCount'] =len(countdict['comwords'])
        return countdict




if __name__ == "__main__":
    logging.basicConfig(filename="BrandImpersonationDb.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    headers = {
        'Referer': 'https://itunes.apple.com',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'
    }
    translator = google_translator()
    brandImpersonationDbObj = BrandImpersonationDb()
    val = 1
    host = 'localhost'
    port = 27017
    try:
        connection = MongoClient(host, port)
        #connection = MongoClient('10.97.158.161', 27017)
        #db = connection.core_stag
        db = connection.core
        collection = db.client_incident
        brandImpersonationDbObj.dbConnection()
    except Exception as e:
        print(e)
        logging.exception("Exception in main():")
    finally:
        connection.close()
        logging.info("=====Closing the db Connection")


