#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from word_dict import end_of_sentence, start_of_sentence
from paths import gen_data_path, plan_data_path, check_uptodate
from poems import Poems
from rank_words import RankedWords
from utils import split_sentences
import re
import subprocess

train_data_path = {
    0: './data/pair',
    1: './data/sentence1',
    2: './data/sentence2',
    3: './data/sentence3',
    4: './data/sentence4'
}
train_data_name = {
    0: ['pairs.txt', 'pair.txt'],
    1: ['keyword1.txt'],
    2: ['keyword2.txt', 'pair.txt'],
    3: ['keyword3.txt','nokeyword3.txt' ],
    4: ['keyword4.txt','nokeyword4.txt' , 'pair.txt']
}

def gen_train_data():
    print("Generating training data ...")
    '''with open(r'raw/corpus.txt', 'r',encoding='utf-8') as fin:
        for line in fin.readlines()[0 : 6]:
            for sentence in split_sentences(line):
                print(' '.join(sentence))'''
    poems = Poems()
    poems.shuffle()
    ranked_words = RankedWords()
    plan_data = []
    gen_data = [[],[],[],[]]
    gen_data_for_pair_train = []
    sentence2_without_keyword = []
    sentence4_without_keyword = []
    for poem in poems:
        if len(poem) != 4:
            continue # Only consider quatrains.
        valid = True
        context = start_of_sentence()
        gen_lines = []
        keywords = []
        for i, sentence in enumerate(poem):

            if len(sentence) != 5: # 选5言诗
                valid = False
                break
            temp = ''.join(sentence)
            tempList = [temp[:2], temp[1:3], temp[2:4], temp[3:5]] + sentence
            words = list(filter(lambda seg: seg in ranked_words, tempList))
            if len(words) == 0:
                valid = False
                break
            keyword = words[0]
            for word in words[1 : ]:
                if ranked_words.get_rank(word) < ranked_words.get_rank(keyword):
                    keyword = word
            if len(keyword) == 2:
                keyword = keyword[0] + ' ' + keyword[1]
            gen_line = ' '.join(sentence) + ' ' + end_of_sentence() + '\t' + start_of_sentence() + ' '+ ' '.join(sentence) + \
                    '\t' + keyword + '\t' + context + '\n'
            if i == 2:
                sentence2_without_keyword.append(' '.join(sentence) + ' ' + end_of_sentence() + '\t' + start_of_sentence() + ' ' + ' '.join(sentence) +\
                    '\t' + '^' + '\t' + context + '\n')
            if i == 3:
                sentence4_without_keyword.append(' '.join(sentence) + ' ' + end_of_sentence() + '\t' + start_of_sentence() + ' ' + ' '.join(sentence) +\
                    '\t' + '^' + '\t' + context + '\n')
            if i == 1 or i == 3:
                gen_line_ = ' '.join(sentence) + ' ' + end_of_sentence() + '\t' + start_of_sentence() + ' ' + ' '.join(sentence) +\
                    '\t' + '^' + '\t' + '^' + ' ' +  ' '.join(last_sentence) + '\n'
                gen_data_for_pair_train.append(gen_line_)
            gen_lines.append(gen_line)
            keywords.append(keyword)
            context += ' '+ ' '.join(sentence) + ' ' + end_of_sentence()
            last_sentence = sentence
        if valid:
            plan_data.append('\t'.join(keywords) + '\n')
            for i, line in enumerate(gen_lines):
                gen_data[i].append(line)
    with open(plan_data_path, 'w') as fout:
        for line in plan_data:
            fout.write(line)
    '''with open(gen_data_path, 'w') as fout:
        pass'''
    for i in range(1,5):
        with open(train_data_path[i] + '/' + f'keyword{i}' + '.txt',  'w') as f:
            for line in gen_data[i-1]:
                f.write(line)
    with open(train_data_path[0] + '/pair.txt', 'w') as f:
        for line in gen_data_for_pair_train:
            f.write(line)
    with open(train_data_path[2] + '/pair.txt', 'w') as f:
        for line in gen_data_for_pair_train:
            f.write(line)
    with open(train_data_path[4] + '/pair.txt', 'w') as f:
        for line in gen_data_for_pair_train:
            f.write(line)
    with open(train_data_path[3] + '/nokeyword3.txt','w') as f:
        for line in sentence2_without_keyword:
            f.write(line)
    with open(train_data_path[4] + '/nokeyword4.txt', 'w') as f:
        for line in sentence4_without_keyword:
            f.write(line)


def batch_train_data(batch_size, mode, i = 0):
    """ Training data generator for the poem generator."""
     # Shuffle data order and cool down CPU
    # .
    if mode != 0 or i != 0:
        gen_train_data()
    filepath = train_data_path[mode] + '/' + train_data_name[mode][i]
    keywords = []
    ground_truth = []
    contexts = []
    sentences = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            toks = line.strip().split('\t')
            sentences.append(toks[0])
            ground_truth.append(toks[1])
            keywords.append(toks[2])
            contexts.append(toks[3])

            if len(keywords) == batch_size:
                yield keywords, ground_truth, contexts, sentences
                keywords.clear()
                ground_truth.clear()
                contexts.clear()
                sentences.clear()
    '''for i in range(4):
        keywords = []
        ground_truth = []
        contexts = []
        sentences = []
        with open(f'./data/gen_data_{i}.txt', 'r') as fin:
            for line in fin.readlines():
                toks = line.strip().split('\t')
                sentences.append(toks[0])
                ground_truth.append(toks[1])
                keywords.append(toks[2])
                contexts.append(toks[3])

                if len(keywords) == batch_size:
                    yield keywords, ground_truth, contexts, sentences
                    keywords.clear()
                    ground_truth.clear()
                    contexts.clear()
                    sentences.clear()'''

        # For simplicity, only return full batches for now.


if __name__ == '__main__':
    #gen_train_data()
    if not check_uptodate(plan_data_path) or \
            not check_uptodate(gen_data_path):
        gen_train_data()
    for k in range(5):
        j = 0
        for i in batch_train_data(10, k):
            j += 1
            if j == 10:
                break
            print(i)
