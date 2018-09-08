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


def dataSplit(inputpath, count):
    (filepath, tempfilename) = os.path.split(inputpath)
    (filename, extension) = os.path.splitext(tempfilename)
    outputlist = []
    for i in range(count):
        outputpath = os.path.join('./word/', filename + "_" + str(i) + extension)
        print(outputpath)
        outputlist.append(outputpath)
    outputfiles = []
    for path in outputlist:
        output = open(path, encoding='utf-8', mode='w+')
        outputfiles.append(output)

    print('open input')
    fin = open(inputpath, encoding='utf-8')
    print('read input')
    lines = fin.readlines()  # 调用文件的 readline()方法
    print('calculate lines')
    total = len(lines)
    sclice = math.floor(total / count)
    i = 0
    while i < count - 1:
        print("write file " + str(i))
        outputfiles[i].writelines(lines[i * sclice:(i + 1) * sclice])
        outputfiles[i].close()
        print("write file " + str(i) + " is ok~~ ")
        i += 1
    print("write file " + str(i))
    outputfiles[i].writelines(lines[i * sclice:])
    outputfiles[i].close()
    print("write file " + str(i) + " is ok~~ ")
    print("all is ok~~")


# dataSplit('TNewsSegafter2.txt', 32)
dataSplit('TNewsSegafter1.txt', 32)
