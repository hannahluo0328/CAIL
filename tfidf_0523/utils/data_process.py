# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:24:59 2019

@author: fishredleaf

数据集的预处理
1 数字如何处理？包括序号，日期，金额
分析：
(1)序号和日期对于相似度判断没有意义，仅保留金额
(2)一般都会有请求法院依法判决，后面跟上几个小点，有序号标注

[1] 匹配标准格式的xx年xx月xx日，删除
[2] 匹配汉字“某”后面的数字，删除
[3] 匹配前后都是符号的单个数字，删除(见分析(2))

2 所有的符号都删除？

"""

import jieba
import pkuseg
import re
import pandas as pd

def remove_date(sentence):
#    sentence = re.sub(u"([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日", "", sentence)
    sentence = re.sub(u"([0-9]{4})年", "", sentence)
    sentence = re.sub(u"([0-9]{1,2})月", "", sentence)
    sentence = re.sub(u"([0-9]{1,2})日", "", sentence)
    
    return sentence

def remove_num_afterMOU(sentence):
    sentence = re.sub(u"某([0-9]){1}", "某", sentence)
    return sentence

def remove_num_infrontofDUNHAO(sentence):
    sentence = re.sub("([0-9]){1}、", "", sentence)
    return sentence


def sentence_cut(sentence, mode, seg):
    if mode == "jieba":
        return " ".join(jieba.cut(str(sentence)))
    else:
        return " ".join(seg.cut(str(sentence)))

if __name__=="__main__":
    
    data_path = 'E:/python_code/CAIL2019_similarityMatch/data/'
    wordCut_mode = "jieba" #"pkuseg"
    
    seg = pkuseg.pkuseg()

    print('load origin data')
    train_data = pd.read_csv(data_path+"train_rawtext_method1.csv")
    
    print("begin process sentence1")
    train_data["text1"] = train_data["text1"].apply(remove_date)
    train_data["text1"] = train_data["text1"].apply(remove_num_afterMOU)
    train_data["text1"] = train_data["text1"].apply(remove_num_infrontofDUNHAO)
    train_data["text1"] = train_data["text1"].apply(sentence_cut, mode=wordCut_mode, seg=seg)
    
    print("begin process sentence2")
    train_data["text2"] = train_data["text2"].apply(remove_date)
    train_data["text2"] = train_data["text2"].apply(remove_num_afterMOU)
    train_data["text2"] = train_data["text2"].apply(remove_num_infrontofDUNHAO)
    train_data["text2"] = train_data["text2"].apply(sentence_cut, mode=wordCut_mode, seg=seg)
    
    train_data.to_csv(data_path+"train_method1.csv", index=False, header=None)
    
    
