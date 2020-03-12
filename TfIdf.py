# Importing libraries
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

# Input the file
txt1 = []
with open('C:\\Users\\tahsin.asif\\OneDrive - CYFIRMA INDIA PRIVATE LIMITED\\AI\\CveScorePrediction\\test1.txt') as file:
    txt1 = file.readlines()


# Preprocessing
def remove_string_special_characters(s):
    # removes special characters with ' '
    stripped = re.sub('[^a-zA-z\s]', '', s)
    stripped = re.sub('_', '', stripped)

    # Change any white space to one space
    stripped = re.sub('\s+', ' ', stripped)

    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped != '':
        return stripped.lower()

    # Stopword removal


stop_words = set(stopwords.words('english'))
your_list = ['skills', 'ability', 'job', 'description']
for i, line in enumerate(txt1):
    txt1[i] = ' '.join([x for
                        x in nltk.word_tokenize(line) if
                        (x not in stop_words) and (x not in your_list)])

# Getting trigrams
vectorizer = CountVectorizer(ngram_range=(1, 1))
X1 = vectorizer.fit_transform(txt1)
features = (vectorizer.get_feature_names())
print("\n\nFeatures : \n", features)
print("\n\nX1 : \n", X1.toarray())

# Applying TFIDF
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
X2 = vectorizer.fit_transform(txt1)
scores = (X2.toarray())
print("\n\nScores : \n", scores)

# Getting top ranking features
sums = X2.sum(axis=0)
data1 = []
for col, term in enumerate(features):
    data1.append((term, sums[0, col]))
ranking = pd.DataFrame(data1, columns=['term', 'rank'])
words = (ranking.sort_values('rank', ascending=False))
print("\n\nWords head : \n", words.head(7))
