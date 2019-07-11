# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:20:30 2019

@author: 鱼红叶
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.metrics import precision_score

data_path = 'E:/nlp/competition/ceil2019/tfidf_0523/data/'
df_train = pd.read_csv(data_path+"train_method1.csv", names=["text1","text2","label"])

array_train = df_train.values
X1 = array_train[:,0].tolist()
X2 = array_train[:,1].tolist()
y = array_train[:,2].astype('int8').tolist()

X1_train, X1_valid, X2_train, X2_valid, y_train, y_valid = train_test_split(
    X1, X2, y, test_size=0.33, random_state=666)
#print(len(X1_train), len(X2_train), len(y_train)) #670

with open("./saved_model/tokenizer_0523.pickle","rb") as handle:
    tokenizer = pickle.load(handle)
word_index = tokenizer.word_index

for w,i in word_index.items():
    word_index[w] = i-1

l1 = len(X1_train)
l2 = len(X1)
l3 = len(X1) + len(X2_train)

vectorizer = TfidfVectorizer(vocabulary=word_index)
X = X1_train + X1_valid + X2_train + X2_valid
#print(len(X))
X = vectorizer.fit_transform(X)
#print(X.shape)

#print(vectorizer.get_feature_names())
X1_train = X[:l1].toarray()
X1_valid = X[l1:l2].toarray()
X2_train = X[l2:l3].toarray()
X2_valid = X[l3:].toarray()

X_train = []
for i in range(l1):
    X_train.append(list(X1_train[i]) + list(X2_train[i]))
X_train = sparse.csr_matrix(X_train)

X_valid = []
for i in range(len(X1_valid)):
    X_valid.append(list(X1_valid[i]) + list(X2_valid[i]))
X_valid = sparse.csr_matrix(X_valid)


model_SGD = SGD(alpha=0.0008,random_state = 2, shuffle = True, loss = 'log', max_iter=1e4)                      
model_SGD.fit(X_train, y_train) # Fit the model.
print("precision score: {:<8.5f}".format(precision_score(y_valid, model_SGD.predict(X_valid))))
