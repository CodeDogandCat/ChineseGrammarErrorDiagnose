# -*- coding:utf-8 -*-
import re
# from pyltp import Segmentor
import jieba
import os
import sys
import pandas as pd

def genChar(path):
    fin = open('data/me.' + path + '.raw.txt', encoding='utf-8')
    fout = open('data/me.' + path + '.char.txt', encoding='utf-8', mode='w+')
    txt = fin.read()
    # print(len(txt))
    for item in txt:
        # string = re.sub("[\s+\.\!\/_,$%^*(+\“\”\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", item)
        string = item.strip()
        if len(string) > 0:
            fout.write(string + " ")
    fin.close()
    fout.close()


def segment(name):
    fin = open('data/me.' + name + '.raw.txt', encoding='utf-8')
    fout = open('data/me.' + name + '.word.txt', encoding='utf-8', mode='w+')
    txt = fin.read()
    # print(len(txt))

    # HIT-LTP
    # segmentor = Segmentor()
    # segmentor.load("/path/to/your/cws/model")
    # words = segmentor.segment("元芳你怎么看")
    # print("|".join(words))
    # segmentor.release()

    # jieba
    seglist = jieba.cut(txt, cut_all=False)
    for item in seglist:
        # string = re.sub("[\s+\.\!\/_,$%^*(+\“\”\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", item)
        string = item.strip()
        if len(string) > 0:
            fout.write(string + " ")
    fin.close()
    fout.close()

def dataSplit(path):

    (filepath, tempfilename) = os.path.split(path)
    (filename, extension) = os.path.splitext(tempfilename)
    train_path=os.path.join(filepath,filename+".train"+extension)
    valid_path=os.path.join(filepath,filename+".valid"+extension)
    test_path=os.path.join(filepath,filename+".test"+extension)
    if not os.path.exists(train_path):
        os.system(r"touch {}".format(train_path))  # 调用系统命令行来创建文件
    if not os.path.exists(valid_path):
        os.system(r"touch {}".format(valid_path))  # 调用系统命令行来创建文件
    if not os.path.exists(test_path):
        os.system(r"touch {}".format(test_path))  # 调用系统命令行来创建文件

    fout_train = open(train_path, encoding='utf-8', mode='w+')
    fout_valid = open(valid_path, encoding='utf-8', mode='w+')
    fout_test = open(test_path, encoding='utf-8', mode='w+')

    print('open input')
    fin = open(path, encoding='utf-8')
    print('read input')
    lines = fin.readlines()  # 调用文件的 readline()方法
    print('calculate lines')
    count=len(lines)
    l1=int(count*0.8)
    l2=int(l1*0.8)
    print('write train data')
    fout_train.writelines(lines[0:l2])
    print('write valid data')
    fout_valid.writelines(lines[l2:l1])
    print('write test data')
    fout_test.writelines(lines[l1:count])

    # print(len(txt))
    # txt_clean=[]
    # for item in txt:
    #     # string = re.sub("[\s+\.\!\/_,$%^*(+\“\”\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", item)
    #     string = item.strip()
    #     if len(string) > 0:
    #         txt_clean.append(string)
    # l=len(txt)
    # train_all_data=txt[0:int(l*0.8)]
    # test_data=txt[int(l*0.8):]
    # l2=len(train_all_data)
    # train_data=train_all_data[0:int(l2*0.8)]
    # valid_data=train_all_data[int(l2*0.8):]
    #
    # fout_train.write(" ".join(train_data))
    # fout_valid.write(" ".join(valid_data))
    # fout_test.write(" ".join(test_data))

    fin.close()
    fout_train.close()
    fout_valid.close()
    fout_test.close()
    print("output files are as following: ")
    print(train_path)
    print(valid_path)
    print(test_path)

# genChar('train')
# segment('train')
# genChar('valid')
# segment('valid')
# genChar('test')
# segment('test')

if __name__ == '__main__':
    if len(sys.argv)==2:
        dataSplit(sys.argv[1])
        print("split ok")
    else:
        print("argv count error")
