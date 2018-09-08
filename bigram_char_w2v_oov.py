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
import multiprocessing
import gensim
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences


# bigram
model = gensim.models.KeyedVectors.load_word2vec_format("./word2vecModel/bigram_char_embedding.bin", binary=True)

fin_word_count = open('./bigram_char_count/TNewsSegafter2_bigram_char_count.json', encoding='utf-8')
fout = open('./bigram_char_count/less_freq', encoding='utf-8', mode='w+')
print("bigram_char_count dict load")
word_count_dict = json.load(fin_word_count)
less_seq_list = []
sum = [0.0 for i in range(200)]
average = []
count = 0
for k, v in word_count_dict.items():
    if v == 5 and k in model.wv.vocab:
        count += 1
        # print(str(k) + "::" + str(v))
        less_seq_list.append(model[k])
        for i, num in enumerate(model[k]):
            sum[i] += num
        fout.write(str(k) + '\n')
        if count >= 100:
            break
print("print sum")
print(sum)
fin_word_count.close()
fout.close()
print('calc average')
for item in sum:
    average.append(float(item) / count)
print('average result')
print(average)