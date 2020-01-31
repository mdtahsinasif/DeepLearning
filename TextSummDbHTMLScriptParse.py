# Implementation from https://dev.to/davidisrawi/build-a-quick-summarizer-with-python-and-nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import datetime
import json

from pymongo import MongoClient
#conda install -c anaconda beautifulsoup4
 



#text_str = open(r"C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/TextSummarization/text.txt","r")
from pathlib import Path
text_str1 = Path('C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/TextSummarization/text.txt').read_text(encoding='utf-8')


import urllib.request
# open a connection to a URL using urllib
#webUrl  = urllib.request.urlopen('https://ciso.economictimes.indiatimes.com/news/user-data-not-affected-from-new-mp4-file-bug-whatsapp/72106154?utm_source=RSS&utm_medium=ETRSS')

#get the result code and print it
#print ("result code: " + str(webUrl.getcode()))

####################

####################
# read the data from the URL and print it
#text_str = webUrl.read()
#print ('text_str2----------->',text_str)

#####################
#url = 'https://ciso.economictimes.indiatimes.com/news/user-data-not-affected-from-new-mp4-file-bug-whatsapp/72106154?utm_source=RSS&utm_medium=ETRSS'

#headers = {}
#headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'

#request = urllib.request.Request(url, headers=headers)
#text_str3 = urllib.request.urlopen(request)

#print(text_str3.read())
#####################

from pathlib import Path
text_str1 = Path('C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/TextSummarization/text.txt').read_text(encoding='utf-8')


text_str1 = '''
'''


url = 'https://ciso.economictimes.indiatimes.com/news/user-data-not-affected-from-new-mp4-file-bug-whatsapp/72106154?utm_source=RSS&utm_medium=ETRSS'

headers = {}
headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'

request = urllib.request.Request(url, headers=headers)
text_str2 = urllib.request.urlopen(request)

text_str = text_str2.read()


def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    # adding beautiful soap logic to get clean text from html 
    
    words = word_tokenize(review_text)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:
    """
    score a sentence by its words
    Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

        '''
        Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
        To solve this, we're dividing every sentence score by the number of words in the sentence.
        
        Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
        the dictionary.
        '''

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary




def run_summarization(text):
    
    
   # print('Inside Review Text')
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''

    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores,.5 * threshold)

    return summary

def run_dbConnection(localhost,port):
    connection = MongoClient()
    connection = MongoClient('localhost', 27017)
    db = connection.core
    collection = db.rss_feed_entry
    mydoc = collection.find({},{ "_id": 0,"description": 1 ,
                        "feedUrl":1,
                        "link":1,
                        "date": datetime.datetime.utcnow()
                        }).limit(3)
    for x in mydoc:
        jsonToStr = json.dumps(x)
        count = len(jsonToStr)
        list_key = x["link"]
        list_val = x["description"]
        lenList = len(list_key)
      #  print('Success-------->',lenList)
        if lenList < 60:
           # print(review_text)
            result = run_summarization(review_text)
            #print('condition true---',result)
            
     


if __name__ == '__main__':
    
    text_str1 = ''
    host ='localhost'
    port = 27017
    run_dbConnection(host,port)
    #print('review_text------->',text_str)
    #1.Remove HTML
    soup = BeautifulSoup(text_str, "html5lib")
    [x.extract() for x in soup.findAll(['script', 'style'])]
   # review_text = BeautifulSoup(soup).get_text()
   # print('review_text------->',soup.prettify())
   # soup.find('div',class_='msg-yellow').decompose()
    
    result = run_summarization(soup.get_text())
    print(result)
