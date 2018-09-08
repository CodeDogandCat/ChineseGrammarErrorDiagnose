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
word_w2v_oov = [0.029771322715096176, -0.1797641052212566, -0.15540868396521546, 0.016750613988842814,
                0.13139006277080625,
                0.18927307306905278, -0.08124262181110681, -0.11126805551582947, -0.0699414604960475,
                -0.03340416052145884,
                0.09550570552702993, 0.22970463457284496, 0.027260677864833268, -0.04238693268503994,
                -0.06484487481415271,
                -0.1105280127376318, 0.23480660470901057, 0.13355112165212632, -0.05781659089261666,
                0.061979648979613555,
                0.11935871035209857, 0.13244090287014842, 0.05829930666368455, -0.23015625100815668,
                -0.10686272406484931,
                -0.08166696057451191, -0.1000116947304923, -0.15575152582954616, 0.2990026765363291,
                0.13900414601899683,
                0.07931797427823767, -0.118478541104123, -0.05718246125383303, -0.1984299713576911,
                -0.026754255649866537,
                -0.06595066790934652, 0.0701980098709464, -0.02021617696620524, -0.02016193582327105,
                -0.059079236197285355,
                0.0716109878453426, -0.08493055985192768, -0.1679619058163371, 0.21693602593615652,
                0.006383166288433131,
                -0.1926433660648763, -0.07516566651727771, 0.026055407213862056, -0.022615917877992615,
                -0.06840781694743782,
                0.15320199897629208, 0.23695636728429237, 0.04463470644375775, -0.08514232685847674, 0.2198224404733628,
                0.026490205188747496, 0.1871540654078126, -0.12004976535798051, 0.19885064010974018,
                0.06408853961387649,
                -0.16148298711050302, 0.017060692647937685, 0.01778742492897436, -0.15750804571434857,
                -0.006176323270192369,
                -0.18479602967971004, 0.0423849461642385, -0.05876891918596812, -0.2518880943860859,
                0.02102667710161768,
                0.11943946411483922, -0.22649141522211722, -0.007660277767572552, 0.03801495554522262,
                -0.06877650342823471,
                0.10171551928648113, -0.2247908383863978, -0.016697161078918724, -0.06680803403316532,
                -0.1342469186190283,
                0.09360605071065947, -0.04069377420222736, 0.029106231809128077, -0.20650600037071853,
                -0.09095133155118675,
                0.08879572547972202, -0.14368580755311997, -0.1794797536265105, 0.032536996271810496,
                -0.2160530481953174,
                0.0483464745618403, -0.16178062450606376, -0.037870418124366555, 0.05067566099343821,
                -0.09106704042933415,
                -0.02212720193900168, 0.012446866587270051, 0.04129601973108947, -0.014168246542540145,
                -0.013927340838126839, 0.001756250481121242, 0.15318492190213873, -0.1464926715742331,
                0.08147144995164127,
                -0.0004667425580555573, -0.03500249912205618, -0.02921883626258932, 0.07437000475008972,
                -0.08972592541307677, -0.278478710136842, 0.17771107251755894, 0.13963137297658249,
                -0.09806235486234073,
                0.2250216169701889, 0.004012143041472882, -0.09557053294032812, -0.07825231843627989,
                -0.016884643267840147,
                -0.07446355580352247, -0.2593223954923451, 0.3167566105537116, -0.1457909119618125,
                -0.19490755932056344,
                -0.027154209732543677, -0.052847609906457364, -0.04406707959016785, -0.12769893463118934,
                0.03398580197710544, -0.0649230542758596, 0.0631581172265578, -0.25887035638093947,
                -0.013910979232750833,
                0.02969641001021955, 0.048437727801501754, -0.014546974645927548, 0.07877976108633447,
                0.05024458586703986,
                0.11588065101765097, 0.20645466508547541, 0.21020315198227763, -0.0055749696283601225,
                -0.3162370246462524,
                -0.04343375195283443, 0.16891142137465068, -0.13932878642925062, -0.11708662887103856,
                0.005635970403673128,
                0.14869307507411578, 0.041947758985916156, -0.04424683906836435, -0.0404190760711208,
                -0.10970234530745074,
                -0.11172063216450624, -0.058651520577259364, 0.19431356606073677, 0.1269311320083216,
                -0.08463340376736596,
                -0.03388741540722549, 0.15248170197010041, 0.027585286480607464, -0.16349083660636096,
                0.14774509760551155,
                -0.32782237785169854, 0.10565179306198842, 0.13771146617771593, -0.12816797761712223,
                -0.018764491938054562,
                0.03492144131625537, -0.11515498196706175, 0.0596710146847181, -0.039062334955669936,
                0.21859635678119957,
                0.11628108082804829, -0.10700459591811523, 0.033867743264709135, 0.17128340468741954,
                0.08141454989672639,
                -0.001826310115866363, 0.09521348401671276, -0.015111713451333345, -0.0887389195105061,
                -0.15221852611051873,
                0.08309945185203105, 0.08956746823649155, 0.11707902904367075, 0.024592045371682615,
                -0.037700746519840324,
                0.06767876122350572, -0.020213128487812357, -0.03641606221906841, 0.003265181382303126,
                0.07741490107262507,
                -0.1375643080496229, 0.04282059690449387, -0.03293223386630416, 0.004753618957329309,
                -0.05660321190676768,
                -0.07076106734770292, 0.03658419543644413, 0.2137074063459295]

bigram_char_w2v_oov = [0.0005831120477523654, 0.05769355658077984, 0.12002986708888784, 0.02792617691215128,
                       -0.0165245641436195, -0.09695760914706625, 0.05609601116273552, 0.08652350618503987,
                       -0.028972933550248853, -0.0836353358766064, 0.0849736112030223, -0.02666875595226884,
                       -0.020405250646872444, -0.21165981879225002, -0.12105198444536654, -0.08862852128222584,
                       -0.03746848228154704, 0.0014151945023331792, -0.1748117186035961, 0.16272600308875554,
                       0.06458960736636073, -0.04625230273464695, -0.06489812372019514, -0.09944242114259395,
                       -0.07131767384562408, -0.06197674044407904, 0.13522486154339275, 0.0010741005279123782,
                       -0.19159874787088482, -0.07682107161730528, -0.013088669899152592, -0.058603720496175814,
                       0.05777258842950687, 0.12523305448819883, 0.05759075023815967, 0.1413981898035854,
                       0.008187892674468457, 0.1993807079643011, -0.07562924352125265, 0.04866087635746226,
                       -0.05535431342432275, 0.07364608417032287, 0.013401636760681868, -0.16837963529513217,
                       -0.001584289837628603, 0.04429339017398888, -0.0425003757054219, 0.09941836113779573,
                       -0.0676285585644655, 0.05643448509043083, 0.1634789887536317, 0.1444323143424117,
                       0.0025518684869166464, 0.158257306413725, 0.35787533338647337, -0.013363099077250808,
                       0.054245654002297666, 0.14205771978362464, 0.007907110687810928, -0.12557187173981219,
                       0.1005135894753039, 0.1016167526517529, 0.25137840980663895, -0.10658334084262605,
                       -0.10403030975256115, 0.06830983675085008, 0.0644283254200127, 0.09659427983220667,
                       -0.012899707892793232, -0.20812497273087502, -0.04112312449840829, -0.16671951009397162,
                       0.03114026532683056, -0.06676869763061405, -0.12480341850779951, 0.060505917337141,
                       0.008584432429634034, 0.10437187836039812, -0.05145611778367311, -0.009951586164534091,
                       0.08693315162439831, 0.01378415804298129, 0.21809956297744065, 0.047395511487557086,
                       -0.013894369804766028, -0.08947949332767166, -0.08390223082154989, -0.06129725483246148,
                       -0.16656324918963947, -0.11042080660350621, -0.07477013414143585, -0.03512691631156486,
                       -0.011686076866462827, 0.03905259353923611, -0.0714030680549331, -0.2036379600391956,
                       0.056049841886851935, 0.15164176262449472, -0.12168411927763373, 0.2966868049930781,
                       0.037680698513868266, 0.06907511651050299, 0.11436443277169019, 0.16279801382275763,
                       0.06743543540593237, -0.03545702121919021, 0.015150566475931555, 0.15195377516094596,
                       -0.04128599758027121, -0.14153385942685417, -0.18168750708056905, 0.29744140363065524,
                       -0.04710268787108362, 0.2608131282031536, 0.004762616329826414, -0.09313854874111711,
                       0.2036620754469186, -0.15466788918245583, 0.022796724267336685, 0.004960063034668565,
                       -0.09669597027939744, 0.07317261587828398, 0.2018115581292659, 0.04438425726257265,
                       -0.03930774960899726, 0.004069343716837466, -0.163145291734254, -0.1539107006182894,
                       0.03278163569513708, -0.23219201508851256, 0.18850403533317148, 0.10309564968803897,
                       -0.017139279863622504, -0.03679624608717859, 0.07257863374252338, 0.03814029755551019,
                       -0.06647380839189282, 0.2722448652656749, -0.2459639698639512, -0.226106969662942,
                       -0.05599095429759473, -0.0006434149469714611, -0.12418424426694401, 0.099498569983989,
                       -0.11322341598337517, 0.04001857514085714, -0.11126881028292701, -0.021179578201845288,
                       0.021721539106220007, 0.05091574957827106, -0.04940707235131413, -0.024806712353019976,
                       0.1675801670999499, -0.07362946415320039, -0.06828219747520052, 0.018201832827180624,
                       0.2906926402822137, -0.11191651111934334, 0.1718908471800387, -0.038370648656200504,
                       0.1657574639422819, -0.11632950928062201, -0.1340338543918915, -0.09386566654080525,
                       0.06651024695253, -0.06860605745110661, -0.1619687252584845, 0.0964965711091645,
                       0.20186959565035067, -0.030679129541967994, -0.16937910285894758, 0.04485365980421193,
                       0.07872250892978627, -0.20419704088912113, 0.05977127591148019, -0.15425279846560444,
                       0.23060828770743683, 0.0347466272697784, -0.009706325181759894, 0.03832375263562426,
                       -0.014624082962982357, -0.1570566437393427, 0.017129206252284347, 0.207728965782444,
                       -0.16490250316099264, 0.09864169714041054, -0.13120224852114915, -0.08690581420669333,
                       0.027403228039620444, -0.06854337667755317, -0.01744082013145089, -0.17409826907562093,
                       -0.1458344029565342, -0.11723822960833787, 0.014643618064583279, 0.24176822017412633,
                       0.048013266291236506, 0.1376235930575058, 0.008741217153146862, 0.192767362780869]


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
                    line_res.append(bigram_char_w2v_oov)
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
    pos_type.append('unk')
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
                line_res.append(pos_embedding[pos])
            else:
                print("char pos embedding OOV! " + str(char) + "::" + str(pos))
                line_res.append(pos_embedding['unk'])
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
    pos_score_type.append('unk')
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
                line_res.append(pos_score_embedding[query_key])
            else:
                print("get_char_pos_score_embedding OOV! " + str(query_key))
                line_res.append(pos_score_embedding['unk'])
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
    pos_score_type.append('unk')
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
                line_res.append(pos_score_embedding[query_key])
            else:
                print("get_char_pmi_embedding OOV! " + str(query_key))
                line_res.append(pos_score_embedding['unk'])
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
    data = ['SBV', 'VOB', 'IOB', 'FOB', 'DBL', 'ATT', 'ADV', 'CMP', 'COO', 'POB', 'LAD', 'RAD', 'IS', 'HED', 'unk']
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
                word_embedding = word_w2v_oov
                print("get_dependence_matrix_embedding word oov")
            if word_depend in model.wv.vocab:
                word_depend_embedding = model[word_depend]
            else:
                word_depend_embedding = word_w2v_oov
                print("get_dependence_matrix_embedding word oov")
            if depend_type in depend_type_dict.keys():
                depend_type_embedding = depend_type_dict[depend_type]
            else:
                print("get_dependence_matrix_embedding depend type oov")
                print(depend_type)
                depend_type_embedding = depend_type_dict['unk']
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
            res_line.append(res_char)
            index_1 += 1
        res.append(res_line)
        index_0 += 1
    return res


def main():
    # feat engineering begin
    # 1. 读取训练数据
    # train_x = read_data('training/new.txt')
    # print('training/new.txt data')
    # print(np.array(train_x).shape)
    # train_y = read_data('training/regular.txt')

    # 测试数据
    train_x = read_data('training/18_old.txt')
    print('training/18_old.txt data')
    print(np.array(train_x).shape)
    train_y = read_data('training/18_regular.txt')

    # 2. feat1: char embedding(random,100dim)
    print('feat1: char embedding(random)')
    char_embedding = construct_char_embedding(train_x)
    print(len(char_embedding))
    training_x_char_embedding = get_char_embedding(train_x, char_embedding)
    print(np.array(training_x_char_embedding).shape)
    # print(training_x_char_embedding)

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

    # ## dump to pickle train
    # import pickle
    # print("save feat to pickle")
    # fout_1 = open('feat_part1_without_dependence.pickle', mode='wb')
    # pickle.dump(feats, fout_1)
    # fout_2 = open('feat_part2_dependence.pickle', mode='wb')
    # pickle.dump(dependences_matrix_embedding, fout_2)
    # fout_1.close()
    # fout_2.close()
    # print('save feat ok~~')

    ## dump to pickle test
    import pickle
    print("save feat to pickle")
    fout_1 = open('/raid/xxx/18_feat_part1_without_dependence.pickle', mode='wb')
    pickle.dump(feats, fout_1)
    fout_2 = open('/raid/xxx/18_feat_part2_dependence.pickle', mode='wb')
    pickle.dump(dependences_matrix_embedding, fout_2)
    fout_1.close()
    fout_2.close()
    print('save feat ok~~')

if __name__ == "__main__":
    main()
