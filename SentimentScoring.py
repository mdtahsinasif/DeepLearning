# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:34:56 2020

@author: tahsin.asif
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

new_words = {
    'drupal': -10.0,
    'bypass':-5.0
}
sia = SentimentIntensityAnalyzer()

sia.lexicon.update(new_words)

passage = '''The Login Security module 6.x-1.x before 6.x-1.3 and 7.x-1.x before 7.x-1.3 for Drupal allows attackers to bypass intended restrictions via a crafted username.'''
print("Sentiment Score:",sia.polarity_scores(passage))
