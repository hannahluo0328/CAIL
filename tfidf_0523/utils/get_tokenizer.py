# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:05:26 2019

@author: 鱼红叶
"""

import pandas as pd
from keras.preprocessing.text import Tokenizer
import pickle

data_path = 'E:/python_code/CAIL2019_similarityMatch/data/'

train_data=pd.read_csv(data_path+"train_method1.csv", names=["text1","text2","label"])
X1 = train_data["text1"].tolist()
X2 = train_data["text2"].tolist()
Y = train_data["label"].tolist()


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X1+X2)
sequence1 = tokenizer.texts_to_sequences(X1)
sequence2 = tokenizer.texts_to_sequences(X2)
word_index = tokenizer.word_index
print(len(word_index))
with open("../saved_model/tokenizer_0523.pickle","wb") as handle:
    pickle.dump(tokenizer, handle, protocol=2)