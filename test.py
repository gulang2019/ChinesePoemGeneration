import pickle as pkl
'''
storefile=r'.\store.txt'
storefile2=r'raw\raw_dict.txt'
rawfile=r'C:\test\ChinesePoetryGeneration\raw\raw\corpus.txt'
words=dict()
with open(rawfile,'r',encoding='utf-8') as fin:
    lines=fin.readlines()
    size=len(lines)
    l=0
    for line in lines:
        sentences=line.split('|')
        for sentence in sentences:
            wordlist=sentence.split(' ')
            for word in wordlist:
                word=word.strip()
                words[word]=words.get(word,0)+1
        l+=1
        if l%100==0:
            print(l/size)
        if l<100:
            print(words)
pkl.dump(words,open(storefile2,'wb'))
'''
from word_vec2 import word2Vec
from word_dict import wordDict, end_of_sentence, start_of_sentence
from data_utils import batch_train_data
from paths import save_dir
from pron_dict import PronDict
from random import random
from singleton import Singleton
from utils import WORD_VEC_DIM, NUM_OF_SENTENCES
import numpy as np
import os
import sys
import tensorflow as tf
if __name__ == '__main__':
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[1, 2], [3, 4]])
    c = tf.concat([a, b], axis = -1)
    print(c)