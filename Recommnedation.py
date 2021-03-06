# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:53:58 2020

@author: tahsin.asif
"""

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
#pip install gensim
from gensim.models import Word2Vec 
import matplotlib.pyplot as plt
#%matplotlib inline

import warnings;
warnings.filterwarnings('ignore')

#pip install xlrd

df = pd.read_excel('C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/Recommendation/Online Retail.xlsx')
df.head()

df.shape

df.isnull().sum()

df.dropna(inplace = True)
df.isnull().sum()

df['StockCode']= df['StockCode'].astype(str)

customers = df["CustomerID"].unique().tolist()
len(customers)


# shuffle customer ID's
random.shuffle(customers)

# extract 90% of customer ID's
customers_train = [customers[i] for i in range(round(0.9*len(customers)))]

# split data into train and validation set
train_df = df[df['CustomerID'].isin(customers_train)]
validation_df = df[~df['CustomerID'].isin(customers_train)]

# list to capture purchase history of the customers
purchases_train = []

# populate the list with the product codes
for i in tqdm(customers_train):
    temp = train_df[train_df["CustomerID"] == i]["StockCode"].tolist()
    purchases_train.append(temp)
    
# list to capture purchase history of the customers
purchases_val = []

# populate the list with the product codes
for i in tqdm(validation_df['CustomerID'].unique()):
    temp = validation_df[validation_df["CustomerID"] == i]["StockCode"].tolist()
    purchases_val.append(temp)  
    
    
# train word2vec model
model = Word2Vec(window = 10, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model.build_vocab(purchases_train, progress_per=200)

model.train(purchases_train, total_examples = model.corpus_count, 
            epochs=10, report_delay=1)

model.init_sims(replace=True)    
print(model)

# extract all vectors
X = model[model.wv.vocab]

X.shape

#pip install umap-learn
import umap

cluster_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,
                              n_components=2, random_state=42).fit_transform(X)

plt.figure(figsize=(10,9))
plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1], s=3, cmap='Spectral')

products = train_df[["StockCode", "Description"]]

# remove duplicates
products.drop_duplicates(inplace=True, subset='StockCode', keep="last")

# create product-ID and product-description dictionary
products_dict = products.groupby('StockCode')['Description'].apply(list).to_dict()

# test the dictionary
products_dict['84029E']

def similar_products(v, n = 6):
    
    # extract most similar products for the input vector
    ms = model.similar_by_vector(v, topn= n+1)[1:]
    
    # extract name and similarity score of the similar products
    new_ms = []
    for j in ms:
        pair = (products_dict[j[0]][0], j[1])
        new_ms.append(pair)
        
    return new_ms  

similar_products(model['90019A'])


def aggregate_vectors(products):
    product_vec = []
    for i in products:
        try:
            product_vec.append(model[i])
        except KeyError:
            continue
        
    return np.mean(product_vec, axis=0)


len(purchases_val[0])

aggregate_vectors(purchases_val[0]).shape

similar_products(aggregate_vectors(purchases_val[0]))

similar_products(aggregate_vectors(purchases_val[0][-10:]))

similar_products(aggregate_vectors(purchases_val[0][-10:]))
