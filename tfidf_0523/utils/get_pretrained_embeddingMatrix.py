# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:48:26 2019

@author: 鱼红叶
"""


import pickle
import numpy as np
from gensim.models import KeyedVectors

with open("../saved_model/tokenizer_0523.pickle","rb") as handle:
    tokenizer = pickle.load(handle)
word_index = tokenizer.word_index

embedding_path = 'E:/python_code/ATEC/data/'
embed_size = 300
embed = KeyedVectors.load_word2vec_format(embedding_path+'sgns.baidubaike.bigram-char')
emb_mean,emb_std = 0.0072130375, 0.3210329

num_vocab = len(word_index)
embedding_matrix = np.random.normal(emb_mean, emb_std, (num_vocab + 1, embed_size))

words = list(embed.vocab)
word_vec = {w:embed[w] for w in words}

print('文本vocab大小: ', num_vocab)
print('百度百科w2v.vocab大小: ', len(words))

for word,i in word_index.items():
    embedding_vector = word_vec.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector
#print(word_vec.get('1'))
#print(word_vec.get(','))
    
with open("../saved_model/embedding_matrix_0523.pickle",'wb') as file:
    pickle.dump(embedding_matrix, file)
    