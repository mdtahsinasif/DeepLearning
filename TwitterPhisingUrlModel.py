# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:29:15 2020

@author: tahsin.asif
"""

#pip install tweepy markovify


import json
import tweepy

CONSUMER_API_KEY = ""
CONSUMER_API_SECRET_KEY = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""

auth = tweepy.OAuthHandler(CONSUMER_API_KEY, CONSUMER_API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(
    auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True
)

user_id = "elonmusk"
count = 200
user_tweets = api.user_timeline(screen_name=user_id, count=count, tweet_mode="extended")

tweet_corpus = []
for tweet in user_tweets:
    tweet_corpus.append(tweet.full_text)
tweets_text = ". ".join(tweet_corpus)

tweets_text

import re


def replace_URLs(string, new_URL):
    """Replaces all URLs in a string with a custom URL."""
    modified_string = re.sub(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " " + new_URL + " ",
        string,
    )
    return modified_string

phishing_link = "https://urlzs.com/u8ZB"

processed_tweets_text = replace_URLs(tweets_text, phishing_link)

processed_tweets_text

import markovify

markov_model = markovify.Text(processed_tweets_text)

num_phishing_tweets_desired = 5
num_phishing_tweets_so_far = 0
generated_tweets = []
while num_phishing_tweets_so_far < num_phishing_tweets_desired:
    tweet = markov_model.make_short_sentence(140)
    if phishing_link in tweet and tweet not in generated_tweets:
        generated_tweets.append(tweet)
        num_phishing_tweets_so_far += 1
        
for tweet in generated_tweets:
    print(tweet)        
    
    
    

user = api.get_user(user_id)
for friend in user.friends():
    print(friend.screen_name)    
