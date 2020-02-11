# Implementation from https://dev.to/davidisrawi/build-a-quick-summarizer-with-python-and-nltk
from bson import ObjectId
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
import pandas as pd
import os
import datetime
import json
import urllib.request


# text_str = open(r"C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/TextSummarization/text.txt","r")
from pathlib import Path

text_str1 = Path(
    'C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/TextSummarization/text.txt').read_text(
    encoding='utf-8')

headers = {}
headers[
    'User-Agent'] = 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'

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

    words = word_tokenize(text_string)
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
    global average
    sumValues = 0

    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    try:
        # Average value of a sentence from original text
        average = (sumValues / len(sentenceValue))

    except ZeroDivisionError:
        print("Description is Empty:::Phising Webpage!")

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
    # print('Inside Review Text',text)
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)
    # print('Inside freq_table------------',)
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
    summary = _generate_summary(sentences, sentence_scores, 0.8 * threshold)
    return summary


def run_dbConnection(localhost, port):
   # connection = MongoClient()
    try:
         connection = MongoClient('localhost', 27017)
         db = connection.core
         collection = db.rss_feed_entry
         mydoc = collection.find({'summary': {'$exists': False}}).limit(48)
        # print("Connected successfully!!!")
    except:
            print("Could not connect to MongoDB")


    d = {}
    sno =0
    for x in mydoc:
        # inserting object id as key and description as value in Mongo
        sno = sno + 1
        #print("sno=========>", sno, '------------>', list_val)
        try:
            list_key = x['_id']
            list_val = x['link']
            d[list_key] = list_val
            print(list_key,'=======',list_val)
        except:
            print("sno=========>", list_val)




    for k,v in d.items():
        request = urllib.request.Request(
              v, headers=headers)

        text_str2 = urllib.request.urlopen(request)
        text_str = text_str2.read()
        # print('review_text------->',text_str)
        # 1.Remove HTML
        soup = BeautifulSoup(text_str, "html.parser")
        # 1.Remove Tags
        [x.extract() for x in soup.findAll(['script', 'style', 'nonscript'])]
        [y.decompose() for y in soup.findAll(['span', 'li', 'noscript', 'footer',
                                              'title', 'a', 'h3'])]

        for div in soup.findAll("div.cookie_accpt"):
            div.decompose()

        for div in soup.find_all("div", {'class': 'tags'}):
            div.decompose()

        for div in soup.find_all("div", {'class': 'cookie_stng hide'}):
            div.decompose()

        for div in soup.find_all("div", {'class': 'glob_nav'}):
            div.decompose()

        for div in soup.find_all("div", {'class': 'subscribe hideit widget'}):
            div.decompose()

        for hidden in soup.find_all(style='display:none'):
            hidden.decompose()

        review_text = soup.get_text(strip=True)
        result = run_summarization(review_text)
        #print(k,'-------------->',result)

        myquery = {"_id":k}
        newvalues = {"$set": {"summary": result}}
        collection.update_many(myquery, newvalues)
        #collection.insert({"test":'', "summary": result});


        #x = mycol.update_many(myquery, newvalues)





if __name__ == '__main__':
    text_str1 = ''
    host = 'localhost'
    port = 27017
    run_dbConnection(host, port)


