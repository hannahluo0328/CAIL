# -*- coding: utf-8 -*-
"""
Created on Tue May 23 08:47:30 2019

@author: 鱼红叶
"""

import jieba
import pkuseg

print('using jieba')
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))
print(", ".join(jieba.cut("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")))

print()
print('using pkuseg')

seg = pkuseg.pkuseg()
seg_list = seg.cut("我来到北京清华大学")
print('pkuseg: ', seg_list)

print()
print('make contrast')
text = "你用不用蚂蚁花呗啊"
jieba.add_word("蚂蚁花呗")
print(','.join(jieba.cut(text)))
seg = pkuseg.pkuseg(user_dict=["蚂蚁花呗"])
print(seg.cut(text))
