# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:33:49 2021

@author: TahsinAsif
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings 
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)
import plotly.graph_objs as go
#pip install chart_studio
#import plotly.plotly as py
from chart_studio import plotly as py
#pip install cufflinks
import cufflinks
pd.options.display.max_columns = 30
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
#pip install bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
output_notebook()
from collections import Counter
import scattertext as st
import spacy
import string
import re
from string import digits
from pprint import pprint
#python -m spacy download en_core_web_sm
import en_core_web_sm
# Import stopwords with nltk.
from nltk.corpus import stopwords
stop = stopwords.words('english')

nlp = spacy.load('C:\\Users\\TahsinAsif\\Anaconda3\\envs\\keras-tf\\Lib\\site-packages\\en_core_web_md\\en_core_web_md-2.0.0')

df = pd.read_csv('C:\\Backup\\PycharmProjects\\PycharmProjects\\DarkWebAnalysis\\DarkWebAnalysisChina.csv',encoding= 'unicode_escape')
df
df.head()
df.info()
df.describe()
df.isnull().sum()

#filling the missing values since date is of an object type, let’s calculate the mode of that column and impute those values to the missing values.

# Find the mode of month in data
#date_mode = df.Date of Issued.mode()[0]

# Fill the missing values with mode value of month in data.
#data.month.fillna(date_mode, inplace = True)

# Let's see the null values in the month column.
# Find the mode of month in data
month_mode = df.DateofIssued.mode()[0]
month_mode
# Fill the missing values with mode value of month in data.
df.DateofIssued.fillna(month_mode, inplace = True)

# Let's see the null values in the month column.
df.DateofIssued.isnull().sum()

# Fill the missing values with mode value of unknown for motives in data.
df.HackerMotive.fillna('Unknown', inplace = True)
df.Country.fillna('Unknown', inplace = True)
df.Campaign.fillna('Unknown', inplace = True)
df.Why.fillna('Unknown', inplace = True)

# not working with categorical data
df.hist(bins=50, figsize=(20,15))
plt.show()
plt.xticks(rotation=90)

#calculate the percentage of each education category.
hackerMotiveVal = df.HackerMotive.value_counts(normalize=True)
print(hackerMotiveVal)
plt.show()

#plot the pie chart of education categories
df.Campaign.value_counts(normalize=True).plot.pie()
plt.show()


TargetedAccountsVal = df.TargetedAccounts.value_counts()
print(TargetedAccountsVal)
plt.show()



#plot the pie chart of education categories
#data.education.value_counts(normalize=True).plot.pie()
#plt.show()

# implementing scatter matrix

from pandas.plotting import scatter_matrix
#attributes =['TargetedAccounts','GroupName','IndustryGroup','Country','Campaign']
#attributes =['TOSHIBA','NHK','MOL','TOPPAN Printing','Mitsui & Co']
attributes= ['Nippon Steel','Tokyo Gas','KONICA MINOLTA','AEON Credit','SUNTORY','First Retailing',
             'Mitsubishi Corporation','Singapore Airline','FUJITSU','Sumitomo Rubber']
#scatter_matrix(df[attributes])
attributes = ['Cipla']

#data visualization for Targeted Accounts and Group Name
sns.set(style="darkgrid")
sns.scatterplot(x='TargetedAccounts',y='GroupName',data=df)
plt.xticks(rotation=90)

for indAccount in attributes : 
            
    High_Alert = df[(df['TargetedAccounts'] == indAccount)  ]        
    #print((indAccount))
    #sns.factorplot(x='TargetedAccounts',hue='Campaign', data=High_Alert,kind="count", size=6)
    #plt.xticks(rotation=90)
    
        
   
    #data visualization for Targeted Accounts and Country
    sns.set(style="darkgrid")      
    sns.scatterplot(x='TargetedAccounts',y='GroupName',data=High_Alert)
    plt.xticks(rotation=90)
   
    
    
     #data visualization for Targeted Accounts and Country
    sns.set(style="darkgrid")    
    sns.scatterplot(x='TargetedAccounts',y='Country',data=High_Alert)
    plt.xticks(rotation=90)
    
    
    
     #data visualization for Targeted Accounts and Campaign
    sns.set(style="darkgrid") 
    sns.scatterplot(x='TargetedAccounts',y='Campaign',data=High_Alert)
    plt.xticks(rotation=90)
        
    
    
    
    #data visualization for Targeted Accounts and Industry
    sns.scatterplot(x='TargetedAccounts',y='IndustryGroup',data=High_Alert)
    plt.xticks(rotation=90)
    plt.grid(True)
    
    
    #data visualization for Targeted Accounts and Motive
    sns.set(style="darkgrid") 
    sns.scatterplot(x='TargetedAccounts',y='HackerMotive',data=High_Alert)
    plt.xticks(rotation=90)
    
    
    #data visualization for Targeted Accounts and date of issuesd
    sns.set(style="darkgrid") 
    sns.scatterplot(x='TargetedAccounts',y='DateofIssued',data=High_Alert)
    plt.xticks(rotation=90)
    
    
    
    
    
   

        

        
        #data visualization for Targeted Accounts and Motive
    sns.scatterplot(x='TargetedAccounts',y='Why',data=High_Alert)
    plt.xticks(rotation=90)
        
        #data visualization for Targeted Accounts and Motive
    sns.scatterplot(x='TargetedAccounts',y='HackerMotive',data=High_Alert)
    plt.xticks(rotation=90)
    
    sns.scatterplot(x='TargetedAccounts',y='IndustryGroup',data=High_Alert)
    plt.xticks(rotation=90)
    plt.grid(True)


        

        
        
        
        
        #data visualization for Targeted Accounts and Industry Group
        
        
sns.scatterplot(x='TargetedAccounts',y='IndustryGroup',data=df)
plt.xticks(rotation=90)
plt.grid(True)
        
        

        #data visualization for Targeted Accounts and Country
sns.scatterplot(x='TargetedAccounts',y='Country',data=df)
plt.xticks(rotation=90)
        
        #data visualization for Targeted Accounts and Campaign
sns.scatterplot(x='TargetedAccounts',y='Campaign',data=df)
plt.xticks(rotation=90)
plt.grid(True)
        
        #data visualization for Targeted Accounts and Campaign
sns.scatterplot(x='TargetedAccounts',y='DateofIssued',data=df)
plt.xticks(rotation=90)
plt.grid(True)
        
        
        #data visualization for Targeted Accounts and Motive
sns.scatterplot(x='TargetedAccounts',y='Why',data=df)
plt.xticks(rotation=90)
        
        #data visualization for Targeted Accounts and Motive
sns.scatterplot(x='TargetedAccounts',y='HackerMotive',data=df)
plt.xticks(rotation=90)


        
test = df['HackerMotive'].value_counts()



# adding some additional parameters
sns.scatterplot(x='TargetedAccounts',y='GroupName',hue='IndustryGroup',data=df)
plt.xticks(rotation=90)

sns.scatterplot(x='TargetedAccounts',y='IndustryGroup',data=df)
plt.xticks(rotation=90)



sns.scatterplot(x='DateofIssued',y='TargetedAccounts',hue='IndustryGroup',data=df)
plt.xticks(rotation=90)

plt.polar(x='DateofIssued',y='TargetedAccounts')


#################################################

#show first five rows of record


#create count plot for room types
#%matplotlib inline
sns.set(style="darkgrid")
ax = sns.countplot(x='IndustryGroup',  data=df)
plt.show()
plt.xticks(rotation=90)


sns.set(style="darkgrid")
ax = sns.countplot(x='TargetedAccounts',  data=df)
plt.show()
plt.xticks(rotation=90)


sns.set(style="dark")
ax = sns.countplot(x='DateofIssued',  data=df)
plt.show()
plt.xticks(rotation=90)

sns.set(style="dark")
ax = sns.countplot(x='Campaign',  data=df)
plt.show()
plt.xticks(rotation=90)

sns.set(style="dark")
ax = sns.countplot(x='Country',  data=df)
plt.show()
plt.xticks(rotation=90)


sns.set(style="dark")
ax = sns.countplot(x='HackerMotive',  data=df)
plt.show()
plt.xticks(rotation=90)

sns.set(style="dark")
ax = sns.countplot(x='GroupName',  data=df)
plt.show()
plt.xticks(rotation=90)




# count plot on two categorical variable
sns.countplot(x ='TargetedAccounts', hue = "DateofIssued", data = df)
 
# Show the plot
plt.show()
plt.xticks(rotation=90)
############################################
# radar graph


fig = plt.figure()
ax = fig.add_subplot(111, projection="polar")

# theta has 5 different angles, and the first one repeated
theta = np.arange(len(df) + 1) / float(len(df)) * 2 * np.pi
# values has the 5 values from 'Col B', with the first element repeated
values = df['IndustryGroup'].values
values = np.append(values, values[0])

# draw the polygon and the mark the points for each angle/value combination
l1, = ax.plot(theta, values, color="C2", marker="o", label="Name of Col B")
plt.xticks(theta[:-1], df['Country'], color='grey', size=12)
ax.tick_params(pad=10) # to increase the distance of the labels to the plot
# fill the area of the polygon with green and some transparency
ax.fill(theta, values, 'green', alpha=0.1)

# plt.legend() # shows the legend, using the label of the line plot (useful when there is more than 1 polygon)
plt.title("Title")
plt.show()

#######################################













########################################


def preprocess(summary):
    summary = summary.str.replace("(<br/>)", "")
    summary = summary.str.replace('(<a).*(>).*(</a>)', '')
    summary = summary.str.replace('(&amp)', '')
    summary = summary.str.replace('(&gt)', '')
    summary = summary.str.replace('(&lt)', '')
    summary = summary.str.replace('(\xa0)', ' ') 
    summary = summary.str.replace('(</strong>)', ' ')
    summary = summary.str.replace('(<strong>)', ' ')
    summary = summary.str.replace('(<br />)', ' ')
    summary = summary.str.replace('(<u>)', ' ')
    summary = summary.str.replace('(</u>)', ' ')
    
    return summary

df['Hacker Motive'] = preprocess(df['Hacker Motive'])

#New column for sentiment polarity. Two new columns for lengths of the review and word count.

df['polarity'] = df['Hacker Motive'].map(lambda text: TextBlob(str(text)).sentiment.polarity)
print (df['polarity']) 
df['summary_len'] = df['Hacker Motive'].astype(str).apply(len)
df['word_count'] = df['Hacker Motive'].apply(lambda x: len(str(x).split()))

#cl = df.loc[df.polarity == 1, ['summary']].sample(5).values
#for c in cl:
 #   print(c[0])
    
    
print('5 random reviews with the most neutral sentiment(zero) polarity: \n')
cl = df.loc[df.polarity == 0, ['Hacker Motive']].sample(5).values
for c in cl:
    print(c[0])    
    
    
df.polarity.min()    

df.loc[df.polarity == -0.22000000000000003]
#after 38


def barplot(words, words_counts, title):
    fig = plt.figure(figsize=(18, 6))
    bar_plot = sns.barplot(x=words, y=words_counts)
    for item in bar_plot.get_xticklabels():
        item.set_rotation(90)
    plt.title(title)
    plt.show()


barplot(words=df['polarity'], words_counts=df['word_count'], title='Most Frequent Words used in motives plots')
plt.xticks(rotation=90)
#continuous features: visual EDA
#sns.boxplot(x='Fare',data=titanic_train)
sns.distplot(df['polarity'])
sns.distplot(df['polarity'], kde=False)
sns.distplot(df['polarity'], bins=20, rug=True, kde=False)
sns.distplot(df['polarity'], bins=100, kde=False)



df['polarity'].iplot(
    kind='hist',
    bins=50,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')


#categorical columns: numerical EDA
pd.crosstab(index=df['Group Name'], columns="count")


#categorical columns: visual EDA
sns.countplot(x=df['Group Name'])
plt.xticks(rotation=90)


df['Date of Issued'].plot(
    kind='hist',
    x='Issue date',
    y='count',
    title='Name Distribution')

sns.distplot(df['summary_len'])
sns.distplot(df['summary_len'], kde=False)
sns.distplot(df['summary_len'], bins=20, rug=True, kde=False)
sns.distplot(df['summary_len'], bins=100, kde=False)

df['summary_len'].plot(
    kind='hist',
    bins=100,
    x='summary length',
    y='count',
    title='Review Text Length Distribution')

#Top unigrams before removing stop words

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

df = df[~df['Hacker Motive'].isnull()]    
common_words = get_top_n_words(df['Hacker Motive'], 20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['Hacker Motive' , 'count'])

sns.distplot(df1.groupby('Hacker Motive').sum()['count'].sort_values(ascending=False), kde=False)


        
#Top unigrams after removing stop words


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['Hacker Motive'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['word' , 'count'])

#sns.distplot(df1.groupby(df['summary']).sum()['count'].sort_values(ascending=False), kde=False)
sns.countplot(df2.groupby('count').sum()['word'].sort_values(ascending=False))
sns.countplot(df2['word'])
plt.xticks(rotation=90)
plt.plot(df2['word'],df2['count'])
plt.show()
plt.xticks(rotation=90)

#Top bigrams before removing stop words

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['Hacker Motive'], 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['word' , 'count'])

sns.countplot(df3.groupby('count').sum()['word'].sort_values(ascending=False))
sns.countplot(df3['word'])
plt.xticks(rotation=90)

plt.plot(df3['word'],df3['count'])
plt.show()
plt.xticks(rotation=90)


#Top bigrams after removing stop words
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['Hacker Motive'], 20)
for word, freq in common_words:
    print(word, freq)
df4 = pd.DataFrame(common_words, columns = ['word' , 'count'])
#sns.countplot(df3['word'])
#plt.xticks(rotation=90)


plt.plot(df4['word'],df4['count'])
plt.show()
plt.xticks(rotation=90)
ax = sns.scatterplot(x=df4['word'], y=df4['count'])
plt.xticks(rotation=90)


#Top trigrams before removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(df['Hacker Motive'], 20)
for word, freq in common_words:
    print(word, freq)
df5 = pd.DataFrame(common_words, columns = ['word' , 'count'])


plt.plot(df5['word'],df5['count'])
plt.show()
plt.xticks(rotation=90)
ax = sns.scatterplot(x=df5['word'], y=df5['count'])
plt.xticks(rotation=90)

#Top trigrams after removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_trigram(df['Hacker Motive'], 20)
for word, freq in common_words:
    print(word, freq)
df6 = pd.DataFrame(common_words, columns = ['word' , 'count'])

plt.plot(df6['word'],df6['count'])
plt.show()
plt.xticks(rotation=90)
ax = sns.scatterplot(x=df6['word'], y=df6['count'])
plt.xticks(rotation=90)

#Top 20 part-of-speech tagging of review corpus
#python -m textblob.download_corpora
blob = TextBlob(str(df['Hacker Motive']))
pos_df = pd.DataFrame(blob.tags, columns = ['word' , 'pos'])
pos_df = pos_df.pos.value_counts()[:20]
plt.plot(pos_df)
plt.show()
plt.xticks(rotation=90)
pos_df.iplot(
    kind='bar',
    xTitle='POS',
    yTitle='count', 
    title='Top 20 Part-of-speech tagging for review corpus')


#Topic Modeling with LSA
reindexed_data = df['Hacker Motive']
tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
reindexed_data = reindexed_data.values
document_term_matrix = tfidf_vectorizer.fit_transform(reindexed_data)

n_topics = 6
lsa_model = TruncatedSVD(n_components=n_topics)
lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix)


def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic 
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)


lsa_keys = get_keys(lsa_topic_matrix)
lsa_categories, lsa_counts = keys_to_counts(lsa_keys)

def get_top_n_words(n, keys, document_term_matrix, tfidf_vectorizer):
    '''
    returns a list of n_topic strings, where each string contains the n most common 
    words in a predicted category, in order
    '''
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
        top_word_indices.append(top_n_word_indices)   
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:,index] = 1
            the_word = tfidf_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))         
    return top_words

top_n_words_lsa = get_top_n_words(3, lsa_keys, document_term_matrix, tfidf_vectorizer)

for i in range(len(top_n_words_lsa)):
    print("Topic {}: ".format(i+1), top_n_words_lsa[i])
    
    
top_3_words = get_top_n_words(2, lsa_keys, document_term_matrix, tfidf_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lsa_categories]

fig, ax = plt.subplots(figsize=(16,8))
ax.bar(lsa_categories, lsa_counts);
ax.set_xticks(lsa_categories);
ax.set_xticklabels(labels);
ax.set_ylabel('Number of review text');
ax.set_title('LSA topic counts');
plt.show();    

tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 
                        n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_topic_matrix)

def get_mean_topic_vectors(keys, two_dim_vectors):
    '''
    returns a list of centroid vectors from each predicted topic category
    '''
    mean_topic_vectors = []
    for t in range(n_topics):
        reviews_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                reviews_in_that_topic.append(two_dim_vectors[i])    
        
        reviews_in_that_topic = np.vstack(reviews_in_that_topic)
        mean_review_in_that_topic = np.mean(reviews_in_that_topic, axis=0)
        mean_topic_vectors.append(mean_review_in_that_topic)
    return mean_topic_vectors

colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5" ])
colormap = colormap[:n_topics]

top_3_words_lsa = get_top_n_words(3, lsa_keys, document_term_matrix, tfidf_vectorizer)
lsa_mean_topic_vectors = get_mean_topic_vectors(lsa_keys, tsne_lsa_vectors)

plot = figure(title="t-SNE Clustering of {} LSA Topics".format(n_topics), plot_width=700, plot_height=700)
plot.scatter(x=tsne_lsa_vectors[:,0], y=tsne_lsa_vectors[:,1], color=colormap[lsa_keys])

for t in range(n_topics):
    label = Label(x=lsa_mean_topic_vectors[t][0], y=lsa_mean_topic_vectors[t][1], 
                  text=top_3_words_lsa[t], text_color=colormap[t])
    plot.add_layout(label)
    
show(plot)

# differentiator words
corpus = st.CorpusFromPandas(df, category_col='Group Name', text_col='Hacker Motive', nlp=nlp).build()
print(list(corpus.get_scaled_f_scores_vs_background().index[:10]))

#['cyfirma', 'exfiltration', 'ndash', 'ransomware', 'iocs', 'heartbleed', 'exfiltrate', 'colluding', 'reputational', 'cti']

#associated terms of each company name
term_freq_df = corpus.get_term_freq_df()
term_freq_df['Hacker Motive'] = corpus.get_scaled_f_scores('GOTHIC PANDA / APT3')
pprint(list(term_freq_df.index[:8]))

term_freq_df['CTI Score'] = corpus.get_scaled_f_scores('Bluenoroff')
pprint(list(term_freq_df.sort_values(by='CTI Score', ascending=False).index[:10]))


norm_data = pd.DataFrame(np.random.normal(size=100000))

skewed_data = pd.DataFrame(np.random.exponential(size=100000))

skewed_data.plot(kind="density",figsize=(10,10),xlim=(-1,5))
