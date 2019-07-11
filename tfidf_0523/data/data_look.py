# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:50:49 2019

@author: 鱼红叶
Levenshtein向量之间距离计算

第一想法，由于sim(text1, text2) > sim(text1, text3),
直接设置sim(text1, text2) = 1, sim(text1, text3) = 0
但是误差会很大，有可能text1和text2之间很相似，text1和text3之间也很相似
train_sample1.csv

第二想法，在模型上考虑
model(text1, text2, text3) = 1
model(text1, text3, text2) = 0
可以改变一半序列的顺序，标签设置为0
train_sample2.csv

"""
import json
import pandas as pd

data = []
with open('input.txt', 'r', encoding='utf-8') as f:
    for line in f:
        dict_ = json.loads(line)
        data.append(dict_)

print(len(data[0]))
#print('----------------------------------------------')
#print(data[0]['A'])
#print('----------------------------------------------')
#print(data[0]['B'])
#print('----------------------------------------------')
#print(data[0]['C'])

#第一想法
text1, text2, labels = [], [], []
for i,d in enumerate(data):
    text1.append(d['A'])
    text1.append(d['A'])
    text2.append(d['B'])
    text2.append(d['C'])
    labels.append(1)
    labels.append(0)
print(len(text1), len(text2), len(labels))

df = pd.DataFrame()
df['text1'] = text1
df['text2'] = text2
df['labels'] = labels
df.to_csv('train_rawtext_method1.csv', index=False)


#第二想法
text1, text2, text3, labels = [], [], [], []
for i,d in enumerate(data):
    if (i+1) <= 250:
        text1.append(d['A'])
        text2.append(d['B'])
        text3.append(d['C'])
        labels.append(1)
    else:
        text1.append(d['A'])
        text2.append(d['C'])
        text3.append(d['B'])
        labels.append(0)
print(len(text1), len(text2), len(text3), len(labels))

df = pd.DataFrame()
df['text1'] = text1
df['text2'] = text2
df['text3'] = text3
df['labels'] = labels
df.to_csv('train_rawtext_method2.csv', index=False)