# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import jieba
import collections
import torch.tensor
import numpy as np
import os
import sys
import jieba.posseg as pseg
import nltk
from pyltp import Postagger
import json
import math
import kenlm
import gensim
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
import os
from pyltp import Parser
from pyltp import Postagger
import jieba
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

torch.manual_seed(1)


def read_data(path):
    '''
    读数据
    :param path:
    :return:
    '''
    print('read_data')
    f0 = open(path, "r", encoding='UTF-8')
    x = f0.readlines()
    return x


def seg(data):
    '''
    结巴分词
    :param data:
    :return:
    '''
    print('seg')
    res = []
    for txt in data:
        words = jieba.cut(txt)
        words = list(words)
        for i in range(len(words)):
            words[i] = words[i].strip()
        res.append(" ".join(words))
    return res


def construct_char_embedding(input):
    '''
    构建 char embedding
    :param input:
    :return:
    '''
    print('construct_char_embedding')
    char_embedding = {}
    data = []
    for line in input:
        data.append("".join(line))
    data = "".join(data)
    word_to_ix = _build_vocab(data)
    # print(word_to_ix)
    embeds = nn.Embedding(len(word_to_ix), 100)  # 单词数, 100 维嵌入
    lookup_tensor = torch.LongTensor(list(word_to_ix.values()))
    # embed = embeds(autograd.Variable(lookup_tensor)).detach().numpy()
    embed = embeds(autograd.Variable(lookup_tensor)).data.numpy()
    i = 0
    for key in word_to_ix.keys():
        char_embedding[key] = embed[i, :]
        i += 1
    return char_embedding


def get_char_embedding(data, char_embedding):
    '''
    获取char embedding
    :param data:
    :param char_embedding:
    :return:
    '''
    print('get_char_embedding')
    res = []
    tmp = []
    for line in data:
        char_list = ' '.join(line).strip().split()
        # char_embedding_output = ""
        char_embedding_output = []
        for char in char_list:
            if char in char_embedding.keys():
                # char_embedding_output += char + "@@" + char_embedding[char] + " "
                # print('$$$$$$$$$$$$')
                # print(char)
                # print(char_embedding[char])
                # print('~~~~~~~~')
                tmp = char_embedding[char]
                char_embedding_output.append(char_embedding[char])
            else:
                print("char embedding OOV! " + str(char))
                char_embedding_output.append(tmp)
                # char_embedding_output += char + "@@" + '0000000000' + " "
                # char_embedding_output.append(char_embedding[char])
        # res.append(char_embedding_output.strip())
        res.append(char_embedding_output)
    return res


def get_char_bigram(data):
    '''
    生成char bigram
    :param data:
    :return:
    '''
    print('get_char_bigram')
    res = []
    for txt in data:
        if len(txt.strip()) > 0:
            mystr = ['<start>']
            content = [j.strip() for j in txt.strip(" \n") if j != " "]
            mystr += content
            mystr.append('<end>')
            bigrm = list(nltk.bigrams(mystr))
            # print(bigrm)
            out = ""
            i = 0
            for item in bigrm:
                if i != 0:
                    out += str(str(item[0]).replace(' ', '') + str(item[1]).replace(' ', '')).replace(" ", "") + " "
                else:
                    i = 1
            # print(out.strip())
            res.append(out.strip())
    return res


def get_char_bigram_embedding(data):
    '''
    获取char bigram 的 embedding
    :param data:
    :return:
    '''
    print('get_char_bigram_embedding')
    print('load w2v model')
    model = gensim.models.KeyedVectors.load_word2vec_format("./word2vecModel/bigram_char_embedding.bin", binary=True)
    print('load w2v model ok')
    res = []
    for txt in data:
        line_res = []
        if len(txt.strip()) > 0:
            for bigram in txt.strip(' \n').split():
                if bigram in model.wv.vocab:
                    # print('#########')
                    # print(bigram)
                    # print("***********")
                    line_res.append(model[bigram])
                    # print(model[bigram])
                else:
                    print('char bigram embedding OOV! ' + str(bigram))
                    line_res.append(model['未知'])
        res.append(line_res)
    return res


def _build_vocab(data):
    '''
    字符级别的构建字典
    :param data:
    :return:
    '''
    print("_build_vocab")
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_ix = dict(zip(words, range(len(words))))

    return word_to_ix


def _build_vocab2(data):
    '''
    列表元素的构建字典
    :param data:
    :return:
    '''
    print("_build_vocab")
    counter = my_counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_ix = dict(zip(words, range(len(words))))

    return word_to_ix


def my_counter(data):
    '''
    字典级别构建字典过程中用到的计数器
    :param data:
    :return:
    '''
    print('my_counter')
    res = {}
    for item in data:
        if item not in res.keys():
            res[item] = 0
        else:
            res[item] += 1
    return res


def get_word_and_char_pos(data):
    '''
    获得word 、char pos
    :param data:
    :return:
    '''
    print('get_word_and_char_pos')
    postagger = Postagger()  # 初始化实例
    LTP_DATA_DIR = './ltp_data_v3.4.0'
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
    postagger.load(pos_model_path)  # 加载模型
    res_word_pos = []
    res_char_pos = []
    for sent in data:
        if len(sent) > 0:
            words = sent.strip(' \n').split()
            postags = postagger.postag(words)  # 词性标注
            # print('words len ' + str(len(words)))
            # print('postags len ' + str(len(postags)))
            word_pos_out = ""
            char_pos_out = ""
            for j in range(len(words)):
                word = words[j].strip(" \n")
                word_pos = str(postags[j]).strip(" \n")
                word_pos_out += word + "@@" + word_pos + " "
                if len(word) > 0:
                    i = 0
                    char = word[i]
                    char_pos_out += char + "@@"
                    char_pos_out += "B-" + str(word_pos) + " "
                    i += 1
                    while i < len(word):
                        char_pos_out += word[i] + '@@'
                        char_pos_out += 'I-' + str(word_pos) + " "
                        i += 1

            res_word_pos.append(word_pos_out.strip())
            res_char_pos.append(char_pos_out.strip())

    postagger.release()  # 释放模型
    return res_word_pos, res_char_pos


def get_char_pos_embedding(data):
    '''
    构建并获得 char pos embedding
    :param data:
    :return:
    '''
    print('get_char_pos_embedding')
    pos_raw_type = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                    'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'x']
    pos_type = []
    for pos in pos_raw_type:
        pos_type.append("B-" + pos)
        pos_type.append(("I-" + pos))

    pos_embedding = {}
    pos_to_ix = _build_vocab2(pos_type)
    # print(pos_to_ix)
    embeds = nn.Embedding(len(pos_to_ix), 50)  # 单词数, 50 维嵌入
    lookup_tensor = torch.LongTensor(list(pos_to_ix.values()))
    # embed = embeds(autograd.Variable(lookup_tensor)).detach().numpy()
    embed = embeds(autograd.Variable(lookup_tensor)).data.numpy()
    i = 0
    for key in pos_to_ix.keys():
        pos_embedding[key] = embed[i, :]
        i += 1
    res = []
    tmp_v = []
    for line in data:
        line_res = []
        char_pos_list = line.strip(" \n").split()
        # char_pos_embedding_output = ""
        for char_pos in char_pos_list:
            tmp = char_pos.split('@@')
            char, pos = tmp[0], tmp[1]
            if pos in pos_embedding.keys():
                # print('######')
                # print(str(char) + "::" + str(pos))
                # print("~~~~~~~~~")
                # char_pos_embedding_output += char + "@@" + pos_embedding[pos] + " "
                tmp_v = pos_embedding[pos]
                line_res.append(pos_embedding[pos])
            else:
                print("char pos embedding OOV! " + str(char) + "::" + str(pos))
                line_res.append(tmp_v)
                # char_pos_embedding_output += char + "@@" + '0000000000' + " "
        # res.append(char_pos_embedding_output.strip())
        res.append(line_res)
    return res


def get_char_pos_score(data):
    '''
    获取char pos score
    :param data:
    :return:
    '''
    print('get_char_pos_score')
    fin = open('./word_pos_count_ratio/TNewsSegafter1_word_pos_count_ratio.json', encoding='utf-8', mode='r')
    print("word pos ratio dict load")
    word_pos_count_ratio_dict = json.load(fin)
    print("calc word pos count ratio")
    res = []
    for row in data:
        words = str(row).strip(' \n').split(' ')
        length = len(words)
        j = 0
        pos_score_out = ""
        while j < length:
            query_key = words[j].strip()
            word = words[j].strip().split("@@")[0]
            pos = words[j].strip().split("@@")[1]
            score = 0
            if query_key in word_pos_count_ratio_dict.keys():
                score = word_pos_count_ratio_dict[query_key]
            else:
                print('get pos score ' + str(word) + " OOV ! ")
            if len(word) > 0:
                # 拆到 char level
                i = 0
                char = word[i]
                pos_score_out += char + "@@"
                pos_score_out += "B-" + str(score) + " "
                i += 1
                while i < len(word):
                    pos_score_out += word[i] + '@@'
                    pos_score_out += 'I-' + str(score) + " "
                    i += 1
            j += 1
        pos_score_out = pos_score_out.strip()
        res.append(pos_score_out)
    return res


def bin(data, bin_num=10, min_v=None, max_v=None):
    """
    连续数据分箱
    :param data:
    :param bin_num:
    :return:
    """
    print('bin')
    res = []
    if min_v is None:
        min_v = min(data)
    if max_v is None:
        max_v = max(data)
    diff = max_v - min_v
    deta = diff / bin_num
    tmp = []
    for i in range(1, bin_num):
        tmp.append(min_v + i * deta)
    tmp.append(max_v)
    for item in data:
        for index, v in enumerate(tmp):
            if float(item) <= float(v):
                res.append(index)
                break
    return res


def get_char_pos_score_embedding(data):
    '''
    获取 char pos score embedding
    :param data:
    :return:
    '''
    print('get_char_pos_score_embedding')
    res = []
    # 获取一维数组 score
    score_raw = []
    for row in data:
        char_pos_score_list = str(row).strip(' \n').split(' ')
        for char_pos_score in char_pos_score_list:
            pos_score = float(char_pos_score.strip().split("@@")[1].split('-')[1])
            score_raw.append(pos_score)
    bin_num = 20
    score_binned = bin(score_raw, bin_num, 0.0, 1.0)
    index_of_score_binned = 0

    # 所有可能的符号
    pos_score_type = []
    for i in range(bin_num):
        pos_score_type.append('B-' + str(i))
        pos_score_type.append('I-' + str(i))
    # 构建词典
    pos_score_embedding = {}
    pos_to_ix = _build_vocab2(pos_score_type)
    # print('pos to ix')
    # print(pos_to_ix)
    # embedding
    embeds = nn.Embedding(len(pos_to_ix), 50)  # 单词数, 50 维嵌入
    lookup_tensor = torch.LongTensor(list(pos_to_ix.values()))
    # embed = embeds(autograd.Variable(lookup_tensor)).detach().numpy()
    embed = embeds(autograd.Variable(lookup_tensor))
    embed = embed.data.numpy()
    i = 0
    for key in pos_to_ix.keys():
        pos_score_embedding[key] = embed[i, :]
        i += 1

    # 查找embedding
    tmp = []
    for line in data:
        line_res = []
        char_pos_score_list = line.strip(' \n').split()
        for char_pos_score in char_pos_score_list:
            full_pos_score = char_pos_score.strip().split("@@")[1]
            prefix = full_pos_score.split('-')[0]
            score = score_binned[index_of_score_binned]
            query_key = prefix + '-' + str(score)
            index_of_score_binned += 1
            if query_key in pos_score_embedding.keys():
                # print("##########")
                # print(query_key)
                # print('~~~~~~~~~~~~~~~')
                tmp = pos_score_embedding[query_key]
                line_res.append(pos_score_embedding[query_key])
            else:
                print("get_char_pos_score_embedding OOV! " + str(query_key))
                line_res.append(tmp)
        res.append(line_res)
    return res


def get_proba(model, sentence):
    """
    计算离散概率（多个则计算联合概率）
    :param model:
    :param sentence:
    :return:
    """
    return model.score(sentence, bos=False, eos=False)


def pmi(p_union, p1, p2):
    """
    计算pmi
    :param p_union:
    :param p1:
    :param p2:
    :return:
    """
    return p_union - p1 * p2


def get_char_pmi(data):
    """
    获取 pmi
    :param data:
    :return:
    """
    print('get_char_pmi')
    model = kenlm.LanguageModel('../software/kenlm/test.bin')
    res = []
    for line in data:
        words = line.strip().split()
        length = len(words)
        words.append('\n')
        i = 0
        pmi_out = ""
        while i < length:
            p_union = get_proba(model, words[i] + " " + words[i + 1])
            p1 = get_proba(model, words[i])
            p2 = get_proba(model, words[i + 1])
            p = pmi(p_union, p1, p2)
            # 拆到 char level
            word = words[i]
            if len(word) > 0:
                # 拆到 char level
                j = 0
                char = word[j]
                pmi_out += char + "@@"
                pmi_out += "B#" + str(p) + " "
                j += 1
                while j < len(word):
                    pmi_out += word[j] + '@@'
                    pmi_out += 'I#' + str(p) + " "
                    j += 1
            i += 1
        # last_char = words[i]
        # p_union = get_proba(model, last_char + " \n")
        # p1 = get_proba(model, last_char)
        # p2 = get_proba(model, '\n')
        # p = pmi(p_union, p1, p2)
        # pmi_out += last_char + "@@" + 'B#' + str(p)
        res.append(pmi_out.strip())
    return res


def get_char_pmi_embedding(data):
    """
    获取 chat pmi embedding
    :param char_pmi:
    :return:
    """
    print('get_char_pmi_embedding')
    res = []
    # 获取一维数组 score
    score_raw = []
    for row in data:
        char_pos_score_list = str(row).strip(' \n').split(' ')
        for char_pos_score in char_pos_score_list:
            # print('**************')
            # print(char_pos_score.strip())
            # print(char_pos_score.strip().split("@@"))
            # print(char_pos_score.strip().split("@@")[1])
            # print(char_pos_score.strip().split("@@")[1].split('-'))
            # print(char_pos_score.strip().split("@@")[1].split('-')[1])
            tmp = char_pos_score.strip().split("@@")[1].split('#')[1]
            # print('~~~~~~~~~~')
            pos_score = float(tmp)
            score_raw.append(pos_score)
    bin_num = 20
    score_binned = bin(score_raw, bin_num)
    index_of_score_binned = 0

    # 所有可能的符号
    pos_score_type = []
    for i in range(bin_num):
        pos_score_type.append('B#' + str(i))
        pos_score_type.append('I#' + str(i))
    # 构建词典
    pos_score_embedding = {}
    pos_to_ix = _build_vocab2(pos_score_type)
    # embedding
    embeds = nn.Embedding(len(pos_to_ix), 50)  # 单词数, 50 维嵌入
    lookup_tensor = torch.LongTensor(list(pos_to_ix.values()))
    # embed = embeds(autograd.Variable(lookup_tensor)).detach().numpy()
    embed = embeds(autograd.Variable(lookup_tensor)).data.numpy()
    i = 0
    for key in pos_to_ix.keys():
        pos_score_embedding[key] = embed[i, :]
        i += 1
    tmp = []
    # 查找embedding
    for line in data:
        line_res = []
        char_pos_score_list = line.strip(' \n').split()
        for char_pos_score in char_pos_score_list:
            # print("###########")
            # print(char_pos_score)
            # print("~~~~~~~~~~~")
            full_pos_score = char_pos_score.strip().split("@@")[1]
            prefix = full_pos_score.split('#')[0]
            score = score_binned[index_of_score_binned]
            query_key = prefix + '#' + str(score)
            index_of_score_binned += 1
            if query_key in pos_score_embedding.keys():
                tmp = pos_score_embedding[query_key]
                line_res.append(pos_score_embedding[query_key])
            else:
                print("get_char_pmi_embedding OOV! " + str(query_key))
                line_res.append(tmp)
        res.append(line_res)
    return res


def get_word_dependence(data):
    """
    获取 词 的依赖关系
    :param data: 
    :return: 
    """
    print('get_word_dependence')
    print('load ltp model')
    LTP_DATA_DIR = './ltp_data_v3.4.0/'  # ltp模型目录的路径
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型
    print('read file')
    res = []
    for txt in data:
        word_pos_list = txt.strip(' \n').split()
        res_line = []
        words = []
        postags = []
        for word_pos in word_pos_list:
            words.append(word_pos.split('@@')[0])
            postags.append(word_pos.split('@@')[1])
        arcs = parser.parse(words, postags)  # 句法分析
        arcs = list(arcs)
        for i in range(len(words)):
            res_word = []
            res_word.append(str(words[i]))
            res_word.append(str(words[int(arcs[i].head) - 1]))
            res_word.append(str(arcs[i].relation))
            res_line.append(res_word)
        res.append(res_line)
    print('release model')
    parser.release()  # 释放模型
    return res


def get_one_hot_dependence_type():
    """
    获取 dependence的one-hot编码
    :return:
    """
    depend_dict = {}
    data = ['SBV', 'VOB', 'IOB', 'FOB', 'DBL', 'ATT', 'ADV', 'CMP', 'COO', 'POB', 'LAD', 'RAD', 'IS', 'HED', 'UNK']
    values = array(data)
    # print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)
    i = 0
    while i < len(onehot_encoded):
        depend_dict[data[i]] = onehot_encoded[i]
        i += 1
    # invert first example
    # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    # print(inverted)
    # print(depend_dict)
    return depend_dict


def get_dependence_matrix_embedding(dependences_matrix):
    """
    把dependence matrix 里面的每个小项embedding
    :param dependences_matrix:
    :return:
    """
    # 加载word2vec模型
    print('get_dependence_matrix_embedding')
    print('load w2v model')
    model = gensim.models.KeyedVectors.load_word2vec_format("./word2vecModel/word_embedding.bin", binary=True)
    print('load w2v model ok')
    depend_type_dict = get_one_hot_dependence_type()
    res = []
    for line in dependences_matrix:
        res_line = []
        for word_dependence in line:
            word = word_dependence[0]
            word_depend = word_dependence[1]
            depend_type = word_dependence[2]
            if word in model.wv.vocab:
                word_embedding = model[word]
            else:
                word_embedding = model['未知']
                print("get_dependence_matrix_embedding word oov")
            if word_depend in model.wv.vocab:
                word_depend_embedding = model[word_depend]
            else:
                word_depend_embedding = model['未知']
                print("get_dependence_matrix_embedding word oov")
            if depend_type in depend_type_dict.keys():
                depend_type_embedding = depend_type_dict[depend_type]
            else:
                print("get_dependence_matrix_embedding depend type oov")
                print(depend_type)
                depend_type_embedding = depend_type_dict['UNK']
            for char in word:
                res_char = []
                res_char.extend(word_embedding)
                res_char.extend(word_depend_embedding)
                res_char.extend(depend_type_embedding)
                res_line.append(res_char)
        res.append(res_line)
    return res


def embedding_by_subnet(dependences_matrix):
    """
    获取词语依赖特征的embedding
    :param dependences_matrix:
    :return:
    """
    print('embedding_by_subnet')
    res = []
    return res


def feats_combination(training_x_char_embedding, char_bigram_embedding, char_pos_embedding, char_pos_score_embedding,
                      char_pmi_embedding, dependences_embedding):
    """
    特征的embedding合并
    :param training_x_char_embedding:[[[char_embedding]]]
    :param char_bigram_embedding:[[[char_bigram_embedding]]]
    :param char_pos_embedding:[[[char_pos_embedding]]]
    :param char_pos_score_embedding:[[[char_pos_score_embedding]]]
    :param char_pmi_embedding:[[[char_pos_score_embedding]]]
    :param dependences_embedding:
    :return:
    """
    print('feats_combination')
    res = []
    index_0 = 0
    len_0 = len(training_x_char_embedding)
    print('len_0 ' + str(len_0))
    print("########")
    for line in training_x_char_embedding:
        print(len(line))
    print("##########")
    while index_0 < len_0:
        print('line ' + str(index_0))
        res_line = []
        len_1 = len(training_x_char_embedding[index_0])
        print('!!!!!!!!!!!!!!!!')
        print(len(training_x_char_embedding[index_0]))
        print(len(char_bigram_embedding[index_0]))
        print(len(char_pos_embedding[index_0]))
        print(len(char_pos_score_embedding[index_0]))
        print(len(char_pmi_embedding[index_0]))
        print('!!!!!!!!!!!!!!!!')
        index_1 = 0
        while index_1 < len_1:
            res_char = []
            res_char.extend(training_x_char_embedding[index_0][index_1])
            res_char.extend(char_bigram_embedding[index_0][index_1])
            res_char.extend(char_pos_embedding[index_0][index_1])
            res_char.extend(char_pos_score_embedding[index_0][index_1])
            res_char.extend(char_pmi_embedding[index_0][index_1])
            print(str(index_1) + "个字的特征维度" + str(len(res_char)))
            res_line.append(res_char)
            index_1 += 1
        res.append(res_line)
        index_0 += 1
    return res


def main():
    # feat engineering begin
    # 1. 读取训练数据
    train_x = read_data('training/debug.txt')
    print('training/debug.txt data')
    print(np.array(train_x).shape)
    train_y = read_data('training/regular.txt')

    # 2. feat1: char embedding(random,100dim)
    print('feat1: char embedding(random)')
    char_embedding = construct_char_embedding(train_x)
    print(len(char_embedding))
    training_x_char_embedding = get_char_embedding(train_x, char_embedding)
    print(np.array(training_x_char_embedding).shape)

    # # 3. feat2: char bigram embedding(w2v,200dim)
    print('feat2: char bigram embedding(w2v）')
    char_bigram = get_char_bigram(train_x)
    print(np.array(char_bigram).shape)
    char_bigram_embedding = get_char_bigram_embedding(char_bigram)
    print(np.array(char_bigram_embedding).shape)

    # 4. feat3: char pos embedding(random,50dim)
    print('feat3: char pos embedding(random)')
    train_x_seg = seg(train_x)
    print(np.array(train_x_seg).shape)
    word_pos, char_pos = get_word_and_char_pos(train_x_seg)
    print(np.array(char_pos).shape)
    char_pos_embedding = get_char_pos_embedding(char_pos)
    print(np.array(char_pos_embedding).shape)

    # 5. feat4: char pos score embedding(50)
    print('feat4: char pos score embedding')
    char_pos_score = get_char_pos_score(word_pos)
    print(np.array(char_pos_score).shape)
    char_pos_score_embedding = get_char_pos_score_embedding(char_pos_score)
    print(np.array(char_pos_score_embedding).shape)

    # 6. feat5: pmi embedding(50)
    print('feat5: pmi embedding')
    char_pmi = get_char_pmi(train_x_seg)
    print(np.array(char_pmi).shape)
    char_pmi_embedding = get_char_pmi_embedding(char_pmi)
    print(np.array(char_pmi_embedding).shape)

    # # 7. feat6: dependence
    print('feat6: dependence')
    dependences_matrix = get_word_dependence(word_pos)
    print(np.array(dependences_matrix).shape)
    # print(dependences_matrix)
    dependences_matrix_embedding = get_dependence_matrix_embedding(dependences_matrix)
    print(np.array(dependences_matrix_embedding).shape)
    # print(dependences_matrix_embedding)
    dependences_final_embedding = embedding_by_subnet(dependences_matrix)
    #
    # feats combination
    feats = feats_combination(training_x_char_embedding, char_bigram_embedding, char_pos_embedding,
                              char_pos_score_embedding, char_pmi_embedding, dependences_final_embedding)
    # # feat engineering end
    print(np.array(feats).shape)
    # print(feats)

    ## dump to pickle
    # import pickle
    # print("save feat to pickle")
    # fout_1 = open('feat_part1_without_dependence.pickle', mode='wb')
    # pickle.dump(feats, fout_1)
    # fout_2 = open('feat_part2_dependence.pickle', mode='wb')
    # pickle.dump(dependences_matrix_embedding, fout_2)
    # fout_1.close()
    # fout_2.close()
    # print('save feat ok~~')


if __name__ == "__main__":
    main()
