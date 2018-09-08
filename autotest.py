import kenlm
import jieba
import numpy as np
import json
from flask import Flask, request
app = Flask(__name__)  # 创建一个服务，赋值给APP
model = kenlm.Model('../software/kenlm/test.bin')
def get_ngram_score(chars, mode=model):
    """
    取n元文法得分
    :param chars: list, 以词或字切分
    :param mode:
    :return:
    """
    return mode.score(' '.join(chars), bos=False, eos=False)
def jiancuo(sentence):
    tokens_word = list(jieba.cut(sentence))
    defen = 1
    error=[]
    ngram_avg_scores = []
    for n in [2, 3]:
        scores = []
        for i in range(len(tokens_word) - n + 1):
                word = tokens_word[i:i + n]
                score = get_ngram_score(word, mode=model)
                scores.append(score)
            # 移动窗口补全得分
            for _ in range(n - 1):
                scores.insert(0, scores[0])
                scores.append(scores[-1])
            avg_scores = [sum(scores[i:i + n]) / len(scores[i:i + n]) for i in range(len(tokens_word))]
            ngram_avg_scores.append(avg_scores)
        # 取拼接后的ngram平均得分
        sent_scores = list(np.average(np.array(ngram_avg_scores), axis=0))
        for i in range(len(sent_scores)):
            if sent_scores[i]<-9:
                error.append(tokens_word[i])
                defen=0
    return defen,error
def guanjianci1(keywords1,keywords2,keywords3,keywords4,centense):
    keywords1_list=keywords1.split('，')
    keywords2_list = keywords2.split('，')
    keywords3_list = keywords3.split('，')
    keywords4_list=keywords4.split('，')
    anwserlist=[]
    defen=0
    flag1=False
    flag2=False
    for k in keywords1_list:
        if k!='' and k in centense:
            anwserlist.append(k)
            flag1=True
    if flag1==True:
        defen=defen+0.5
    else:
        for k in keywords2_list:
            if k!='' and k in centense:
                anwserlist.append(k)
                flag1=True
        if flag1==True:
            defen=defen+0.3
def shiyishi(centense):
    payload = {'test': centense}
    r = requests.get("http://10.2.101.162:8802/detect", params=payload)
    data2 = json.loads(r.text)
    return data2
def baoyibao(keywords1,keywords2,keywords3,keywords4,centense):
    inputkeywords = []
    keywords = jieba.analyse.extract_tags(centense, topK=4, withWeight=False)
    listkeyword=keywords1+keywords2+keywords3+keywords4
    for i in keywords:
        if i not in listkeyword:
            inputkeywords.append(i)
    return inputkeywords
def haoyihao(centense):
    payload = {'test': centense}
    r = requests.get("http://10.5.190.22:8801/getppl", params=payload)
    data2 = json.loads(r.text)
    return data2
def zuoyizuo(a,b):
    aa=int(a)
    bb=int(b)
    return aa+bb
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == "POST":
        parms = request.form.to_dict()
        sentence = parms.get('sentence')
        ansnum = parms.get('ansnum')
        keylist = parms.get('keylist')
        sublist = parms.get('sublist')
    else:
        sentence = request.args.get('test')
        ansnum = request.args.get('ansnum')
        keylist = request.args.get('keylist')
        sublist = request.args.get('sublist')
    wordscore = 0
    sentscore = 0
    sumscore = 0
    answerlist = []
    studans = []
    perror = []
    worderror = []
    data = {}
    data['wordscore'] = wordscore
                flag = True
        if flag==True:
            wordscore = wordscore + 1.0/sumnum
        else:
            for i in range(len(subabclist)):
                if subabclist[i] in sentence:
                    answerlist.append(subabclist[i])
                    flag = True
            if flag==True:
                wordscore = wordscore + 0.5/sumnum
    keywords = jieba.analyse.extract_tags(centense, topK=sumnum, withWeight=False)
    for i in keywords:
        if i not in answerlist:
            studans.append(i)
    sentscore,perror = jiancuo(sentence)
    sumscore = sentscore + wordscore
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8802,debug=True)
import kenlm
import jieba
import numpy as np
import json
from flask import Flask, request
app = Flask(__name__)  # 创建一个服务，赋值给APP
model = kenlm.Model('../software/kenlm/test.bin')
def get_ngram_score(chars, mode=model):
    """
    取n元文法得分
    :param chars: list, 以词或字切分
    :param mode:
    :return:
    """
    return mode.score(' '.join(chars), bos=False, eos=False)
