#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from word_dict import wordDict
from gensim import models
from numpy.random import uniform
from paths import word2vec_path, check_uptodate
from poems import Poems
from singleton import Singleton
from utils import WORD_VEC_DIM
import numpy as np
import os

#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from word_dict import wordDict
from gensim import models
from numpy.random import uniform
from paths import word2vec_path, check_uptodate
from poems import Poems
from singleton import Singleton
from utils import WORD_VEC_DIM
import numpy as np
import os


def _gen_word2vec():
    print("Generating word2vec model ...")
    word_dict = wordDict()
    poems = Poems()
    poems=[poem[0]+poem[1]+poem[2]+poem[3] for poem in poems]
    print(poems[1])
    model = models.Word2Vec(poems, size = WORD_VEC_DIM, min_count = 1) # 低频词比较多
    embedding = uniform(-1.0, 1.0, [len(word_dict), WORD_VEC_DIM])
    for i, ch in enumerate(word_dict):
        if ch in model:
            embedding[i, :] = model[ch]
    np.save(word2vec_path, embedding)

class word2Vec(Singleton):

    def __init__(self):
        '''if not check_uptodate(word2vec_path):
            _gen_word2vec()'''
        self.embedding = np.load(word2vec_path)
        self.word_dict = wordDict()
    def print_(self, s):
        print(s)
    def similar_word_(self, s, num=1):
        if num <= 0:
            return []
        if len(s) == 2:
            return s
        tmp = self.embedding[self.word_dict.word2int(s)]
        t = 1000
        ret = []
        for i in range(len(self.word_dict)):
            sim = np.sum(np.square(tmp - self.embedding[i]))
            if len(ret) < num:
                ret.append([i, sim])
            else:
                max_cur = max([i[1] for i in ret])
                if max_cur > sim:
                    for j in range(len(ret)):
                        if ret[j][1] == max_cur:
                            ret[j] = [i, sim]
        return [self.word_dict.int2word(i[0]) for i in ret]
    def get_embedding(self):
        return self.embedding

    def get_vect(self, ch):
        return self.embedding[self.word_dict.word2int(ch)]

    def get_vects(self, text):
        return np.stack(map(self.get_vect, text)) if len(text) > 0 \
                else np.reshape(np.array([[]]), [0, WORD_VEC_DIM])


# For testing purpose.
if __name__ == '__main__':
    word2vec = word2Vec()
    print(word2vec.get_embedding().shape)
    #print(word2vec.get_vect('^'))
    print(word2vec.similar_word_('剑', 4))
    print(word2vec.similar_word_("樱", 4))

'''def _gen_word2vec():
    print("Generating word2vec model ...")
    word_dict = wordDict()
    poems = Poems()
    poems=[poem[0]+poem[1]+poem[2]+poem[3] for poem in poems]
    model = models.Word2Vec(poems, size = WORD_VEC_DIM, min_count = 1) # 低频词比较多
    embedding = uniform(-1.0, 1.0, [len(word_dict), WORD_VEC_DIM])
    for i, ch in enumerate(word_dict):
        if ch in model:
            embedding[i, :] = model[ch]
    np.save(word2vec_path, embedding)

class word2Vec(Singleton):

    def __init__(self):
        if not check_uptodate(word2vec_path):
            _gen_word2vec()
        self.embedding = np.load(word2vec_path)
        self.word_dict = wordDict()

    def get_embedding(self):
        return self.embedding

    def get_vect(self, ch):
        return self.embedding[self.word_dict.word2int(ch)]

    def get_vects(self, text):
        return np.stack(map(self.get_vect, text)) if len(text) > 0 \
                else np.reshape(np.array([[]]), [0, WORD_VEC_DIM])


# For testing purpose.
if __name__ == '__main__':
    word2vec = word2Vec()
    t = word2vec.get_embedding()
    for i in range(100):
        print(np.min(t[i]), np.max(t[i]))
    print(t)
'''
