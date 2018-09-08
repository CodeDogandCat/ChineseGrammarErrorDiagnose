# -*- coding: utf-8 -*-
import re
# from pyltp import Segmentor
import jieba.posseg as pseg
import jieba
import os
import sys
import json
import math
# import kenlm
import nltk
from collections import Counter

fin_word_count = open('./word_count/TNewsSegafter1_word_count.json', encoding='utf-8')
fin_word_pos_count = open('./word_pos_count/TNewsSegafter1_word_pos_count.json', encoding='utf-8')
fout = open('./word_pos_count_ratio/TNewsSegafter1_word_pos_count_ratio.json', encoding='utf-8', mode='w+')
print("word count dict load")
word_count_dict = json.load(fin_word_count)
print("word pos count dict load")
word_pos_count_dict = json.load(fin_word_pos_count)
word_pos_ratio = {}
print("calc word pos count ration")
for k, v in word_pos_count_dict.items():
    key = str(k).split("@@")[0]
    if key in word_count_dict.keys():
        ratio = round((float(v) / float(word_count_dict[key])), 4)
        word_pos_ratio[k] = ratio
        print(k + " " + str(ratio))
print("output word pos ratio dict")
json.dump(word_pos_ratio, fout, ensure_ascii=False)
fout.write('\n')
fin_word_pos_count.close()
fin_word_count.close()
fout.close()
