# -*- coding: utf-8 -*-
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

input_dir = './bigram_char/'
model = Word2Vec(PathLineSentences(input_dir),
                 size=200, window=5, min_count=5,
                 workers=multiprocessing.cpu_count() * 2, iter=20, sg=1)

model.save('./word2vecModel/bigram_char_embedding.model')
model.save_word2vec_format("./word2vecModel/bigram_char_embedding.bin",binary=True)
