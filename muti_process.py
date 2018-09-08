# -*- coding: utf-8 -*-
from multiprocessing import Pool
import math
import nltk
import sys
import os


def runn(data, index, size, path):
    print("jjjjj"+str(index))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = sys.argv[1]
        f = open(path, "r", encoding='UTF-8')
        print(path)
        bline = f.readlines()
        processor = 20
        p = Pool(processor)
        for i in range(processor):
            p.apply_async(runn, args=(bline, i, processor, path), )
            print("process "+str(i)+" is going on ~~")
        p.close()
        p.join()
        f.close()
        print("process ok~~")
    else:
        print("argv count error")
