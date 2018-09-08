from gensim.models import word2vec
sentences = word2vec.LineSentence('./TNewsSegafter1.txt')
model =word2vec.Word2Vec(sentences, sg=1, size=200,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=40)
model.save('./mymodel.model')
model.wv.save_word2vec_format('./mymodel.bin', binary=True)
