# -*- coding: utf-8 -*-
import re
import os
import sys
import json
import math
import kenlm
import nltk
from collections import Counter
import os


def print_score(sentence):
    words = sentence.split()
    for i, (prob, length, oov) in enumerate(model.full_scores(sentence, bos=False, eos=False)):
        print('{0} {1} {2}'.format(prob, length, ' '.join(words[i + 2 - length:i + 2])))
        if oov:
            print('\t"{0}" is an OOV'.format(words[i + 1]))


def print_score_condition(sentence):
    print(model.score(sentence, bos=False, eos=False))


model = kenlm.LanguageModel('../software/kenlm/test.bin')
print('{0}-gram model'.format(model.order))

print_score('我')
print("~~~~~")
print_score('是')
print("~~~~~")
print_score("我 是 谁")
print_score_condition("我 是 谁")
print("~~~~~")
print_score("我 在 哪")
print_score_condition("我 在 哪")
