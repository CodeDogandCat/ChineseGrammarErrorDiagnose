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
from multiprocessing import Pool
import math
import nltk
import sys
import os


def runn(path, index):
    print(str(os.getpid()) + '--' + str(os.getppid()) + '--' + str(index))
    (filepath, tempfilename) = os.path.split(path)
    (filename, extension) = os.path.splitext(tempfilename)
    out_path = os.path.join(filepath, filename + ".word_count.json")
    print(out_path)
    f0 = open(out_path, "w+", encoding='UTF-8')
    fin = open(path, encoding='UTF-8')
    lines = fin.readlines()

    word_count_dict = {}
    for txt in lines:
        if len(txt.strip()) > 0:
            word_counts = Counter(txt.strip(' \n').split(' '))
            for w, f in iter(word_counts.items()):
                if w in word_count_dict.keys():
                    word_count_dict[str(w)] += int(f)
                else:
                    word_count_dict[str(w)] = int(f)
    json.dump(word_count_dict, f0, ensure_ascii=False)
    f0.write('\n')
    fin.close()
    f0.close()
    return word_count_dict


def mergeDict(objs):
    _total = objs[0]
    for i in range(1, len(objs)):
        for k, v in objs[i].items():
            if k in _total.keys():
                _total[k] += v
            else:
                _total[k] = v
    return _total


processor = 32
# path = './word/TNewsSegafter1_'
word_count_dicts = []
# for i in range(processor):
    # p = Pool(processor)
    # for i in range(processor):
    #     tmp = p.apply_async(runn, args=(path + str(i) + ".txt", i), )
    #     word_count_dicts.append(tmp)
    #     print("process " + str(i) + " is going on ~~")
    # p.close()
    # p.join()
    # print("process ok~~")

path = './word_count/TNewsSegafter1_'
for i in range(processor):
    fin_word_count = open(path+ str(i) + '.word_count.json', encoding='utf-8')
    word_count_dict = json.load(fin_word_count)
    fin_word_count.close()
    word_count_dicts.append(word_count_dict)

print('merge word count dict')
res = mergeDict(word_count_dicts)
print("output final word count dict")
fout = open('./word_count/TNewsSegafter1_word_count.json', encoding='utf-8', mode='w+')
json.dump(res, fout, ensure_ascii=False)
fout.write('\n')
fout.close()