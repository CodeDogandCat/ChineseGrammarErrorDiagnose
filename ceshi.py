import kenlm
import jieba
import numpy as np
model = kenlm.Model('../software/kenlm/test.bin')
#model2 = kenlm.Model('../software/kenlm/testchar5.bin')
def get_ngram_score(chars, mode=model):
    """
    取n元文法得分
    :param chars: list, 以词或字切分
    :param mode:
    :return:
    """
    return mode.score(' '.join(chars), bos=False, eos=False)
def detect(sentence):
    # 切词
    tokens_word = list(jieba.cut(sentence))
    defen = 1
    error=[]
    ngram_avg_scores = []
    for n in [3]:
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
        print(avg_scores)
        ngram_avg_scores.append(avg_scores)
    # 取拼接后的ngram平均得分
    sent_scores = list(np.average(np.array(ngram_avg_scores), axis=0))
    print(sent_scores)
    for i in range(len(sent_scores)):
        if sent_scores[i]<-9:
            error.append(tokens_word[i])
            defen = 0
    return defen, error
cent1='他们高兴。'
cent2='我世界级的时间。'
print(detect(cent1))
print(detect(cent2))
