# -*- coding: utf-8 -*-
# 文本文件必须是utf-8无bom格式
from gensim.models.word2vec import Word2Vec
import gensim
model = Word2Vec.load('./word2vecModel/word_embedding.model')
#model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\lanxum\\Desktop\\mymodel.model')  # 3个文件放在一起：Word60.model   Word60.model.syn0.npy   Word60.model.syn1neg.npy
print("read model successful")

word_list = ['教育',
            '不存在的词',
            '的',
            '我',
            '你',
            '他',
            '个',
            '1',
            '完成',
            '吃',
            '苹果',
            '香蕉',
            '词汇',
            '物理',
            '地球',
            '黑死病',
            '瘟疫',
            '', ]
for word in word_list:
    if word in model.wv.index2word:
        vec = model[word]
        print(type(vec))
        print(word,vec)
    else:
        print(word + '\t\t\t——不在词汇表里' + '\n\n')
