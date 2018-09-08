# -*- coding: utf-8 -*-
# import multiprocessing
#
# from gensim.models import Word2Vec
# from gensim.models.word2vec import PathLineSentences
#
# input_dir = './word/'
# model = Word2Vec(PathLineSentences(input_dir),
#                  size=200, window=5, min_count=5,
#                  workers=multiprocessing.cpu_count() * 2, iter=20, sg=1)
#
# model.save('./word2vecModel/word_embedding.model')
# model.wv.save_word2vec_format("./word2vecModel/word_embedding.bin", binary=True)


# -*- coding: utf-8 -*-
import multiprocessing
import gensim
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

model = word2vec.Word2Vec.load("./word2vecModel/word_embedding.model")
# model = gensim.models.KeyedVectors.load_word2vec_format("./word2vecModel/bigram_char_embedding.bin", binary=True)
print("加载模型成功")
model.wv.save_word2vec_format("./word2vecModel/word_embedding.bin", binary=True)
print("保存成功")