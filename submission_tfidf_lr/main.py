# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:28:30 2019

@author: 鱼红叶
"""


import re
import json
import jieba
import pickle
import pandas as pd


#####################################################################################
# txt to dataframe
#####################################################################################

print('txt to dataframe')
input_path = "/input/input.txt"

data = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        dict_ = json.loads(line)
        data.append(dict_)

text1, text2, text3 = [], [], []
for i,d in enumerate(data):
    text1.append(d['A'])
    text2.append(d['B'])
    text3.append(d['C'])

test_data = pd.DataFrame()
test_data['text1'] = text1
test_data['text2'] = text2
test_data['text3'] = text3

print(len(text1), len(text2), len(text3))
print('txt to dataframe finished')


#####################################################################################
# data preprocess
#####################################################################################

print('data preprocess')
def remove_date(sentence):
    # sentence = re.sub(u"([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日", "", sentence)
    sentence = re.sub(u"([0-9]{4})年", "", sentence)
    sentence = re.sub(u"([0-9]{1,2})月", "", sentence)
    sentence = re.sub(u"([0-9]{1,2})日", "", sentence)
    
    return sentence


def remove_num_afterMou(sentence):
    sentence = re.sub(u"某([0-9]){1}", "某", sentence)
    return sentence


def remove_num_infrontofDunhao(sentence):
    sentence = re.sub("([0-9]){1}、", "", sentence)
    return sentence


def sentence_cut(sentence):
    return " ".join(jieba.cut(str(sentence)))

    
print("begin process sentence1")
test_data["text1"] = test_data["text1"].apply(remove_date)
test_data["text1"] = test_data["text1"].apply(remove_num_afterMou)
test_data["text1"] = test_data["text1"].apply(remove_num_infrontofDunhao)
test_data["text1"] = test_data["text1"].apply(sentence_cut)

print("begin process sentence2")
test_data["text2"] = test_data["text2"].apply(remove_date)
test_data["text2"] = test_data["text2"].apply(remove_num_afterMou)
test_data["text2"] = test_data["text2"].apply(remove_num_infrontofDunhao)
test_data["text2"] = test_data["text2"].apply(sentence_cut)

print("begin process sentence3")
test_data["text3"] = test_data["text3"].apply(remove_date)
test_data["text3"] = test_data["text3"].apply(remove_num_afterMou)
test_data["text3"] = test_data["text3"].apply(remove_num_infrontofDunhao)
test_data["text3"] = test_data["text3"].apply(sentence_cut)

print('data preprocess finished!')


#####################################################################################
# get tokenizer
#####################################################################################

#print('get tokenizer')
#from keras.preprocessing.text import Tokenizer
#
#X1 = test_data["text1"].tolist()
#X2 = test_data["text2"].tolist()
#X3 = test_data["text3"].tolist()
#
#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(X1+X2+X3)
##sequence1 = tokenizer.texts_to_sequences(X1)
##sequence2 = tokenizer.texts_to_sequences(X2)
##sequence3 = tokenizer.texts_to_sequences(X3)
#word_index = tokenizer.word_index
#
#print(len(word_index))
#print('get tokenizer finiashed')


#####################################################################################
# predict
#####################################################################################

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
#from sklearn.linear_model import SGDClassifier as SGD
#from sklearn.metrics import precision_score


array_test = test_data.values
X1 = array_test[:, 0].tolist()
X2 = array_test[:, 1].tolist()
X3 = array_test[:, 2].tolist()

with open("tokenizer_version1.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
word_index = tokenizer.word_index

for w, i in word_index.items():
    word_index[w] = i-1

vectorizer = TfidfVectorizer(vocabulary=word_index)
X = X1 + X2 + X3
# print(len(X))
X = vectorizer.fit_transform(X)
# print(X.shape)

X1 = X[:len(X1)].toarray()
X2 = X[len(X1):(len(X1) + len(X2))].toarray()
X3 = X[(len(X1) + len(X2)):].toarray()

AB = []
for i in range(len(X1)):
    AB.append(list(X1[i]) + list(X2[i]))
AB = sparse.csr_matrix(AB)

AC = []
for i in range(len(X1)):
    AC.append(list(X1[i]) + list(X3[i]))
AC = sparse.csr_matrix(AC)

#读取Model
with open('lr_01.pickle', 'rb') as f:
    model_SGD = pickle.load(f)

output_path = "/output/output.txt"

with open(output_path, 'w') as f:
    for i in range(len(X1)):
#        print(model_SGD.predict_proba(AB[i]))
#        break
        if model_SGD.predict_proba(AB[i])[0][1] > model_SGD.predict_proba(AC[i])[0][1]:
            out = 'B'
        else:
            out = 'C'
        f.write(out)
        f.write('\n')
