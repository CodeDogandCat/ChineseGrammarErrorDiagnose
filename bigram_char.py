# -*- coding: utf-8 -*-
from multiprocessing import Pool
import math
import nltk
import sys
import os


def runn(path, index):
    print(str(os.getpid()) + '--' + str(os.getppid()) + '--' + str(index))
    (filepath, tempfilename) = os.path.split(path)
    (filename, extension) = os.path.splitext(tempfilename)
    out_path = os.path.join(filepath, filename + ".bigram_char_" + extension)
    print(out_path)
    f0 = open(out_path, "w+", encoding='UTF-8')
    fin = open(path, encoding='UTF-8')
    lines = fin.readlines()
    for txt in lines:
        if len(txt.strip()) > 0:
            mystr = ['<start>']
            content = [j for j in txt.strip("\n") if j != " "]
            mystr += content
            mystr.append('<end>')
            bigrm = list(nltk.bigrams(mystr))
            out = ""
            for item in bigrm:
                out += item[0] + item[1] + " "
            f0.write(out.strip() + "\n")
    fin.close()
    f0.close()

processor = 32
path = 'TNewsSegafter2_'
for i in range(processor):
    p = Pool(processor)
    for i in range(processor):
        p.apply_async(runn, args=(path + str(i) + ".txt", i), )
        print("process " + str(i) + " is going on ~~")
    p.close()
    p.join()
    print("process ok~~")
#
# if __name__ == '__main__':
#     if len(sys.argv) == 2:
#         path = sys.argv[1]
#         f = open(path, "r", encoding='UTF-8')
#         print(path)
#         bline = f.readlines()
#         processor = 2
#         p = Pool(processor)
#         for i in range(processor):
#             p.apply_async(runn, args=(bline, i, processor, path), )
#             print("process "+str(i)+" is going on ~~")
#         p.close()
#         p.join()
#         f.close()
#         print("process ok~~")
#     else:
#         print("argv count error")
