# -*- coding: utf-8 -*-
import multiprocessing
import gensim
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

# model = word2vec.Word2Vec.load("./word2vecModel/bigram_char_embedding.model")
model = gensim.models.KeyedVectors.load_word2vec_format("./word2vecModel/bigram_char_embedding.bin", binary=True)
print(model['未知'])
# print("加载模型成功")
# model.wv.save_word2vec_format("./word2vecModel/bigram_char_embedding.bin", binary=True)
# print("保存成功")
# # 计算两个词的相似度/相关程度
# y1 = model.similarity("糖葫芦", "好")
# print("【糖葫芦】和【好】的相似度为：", y1)
# print("--------\n")
#
# # 寻找对应关系
# print("书-不错，质量-")
# y3 = model.most_similar(['质量', '不错'], ['书'], topn=3)
# for item in y3:
#     print(item[0], item[1])
# print("--------\n")
#
# # 寻找不合群的词
# y4 = model.doesnt_match("书 书籍 教材 很".split())
# print("不合群的词：", y4)
# print("--------\n")
