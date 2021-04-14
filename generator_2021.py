# -*- coding:utf-8 -*-

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
import tensorflow.compat.v1 as tf
import time
import matplotlib.pyplot as plt
BATCH_SIZE = 128
NUM_UNITS = 128
LEN_PER_SENTENCE = 5
_model_path = os.path.join(save_dir, 'model')
WORD_DICT_SIZE = wordDict().__len__()
model_load_path = {
    0:'./models/pair',
    1: './models/pair',
    2: './models/pair',
    3: './models/pair',
    4: './models/pair',
}
model_save_path = {
    0:'./models/pair',
    1:'./models/sentence1',
    2:'./models/sentence2',
    3:'./models/sentence3',
    4:'./models/sentence4',
}
result_save_path = {
    0:'./result/pair',
    1:'./result/sentence1',
    2:'./result/sentence2',
    3:'./result/sentence3',
    4:'./result/sentence4',
}
class Generator(Singleton):
    def bidirectionRnn(self, name, id, length, dtype):
        with tf.variable_scope(name,reuse=False):
            embedding = tf.nn.embedding_lookup(self.embedding, id)
            gru_fw_cell = tf.nn.rnn_cell.GRUCell(NUM_UNITS // 2, dtype=dtype)
            gru_bw_cell = tf.nn.rnn_cell.GRUCell(NUM_UNITS // 2, dtype=dtype)
            outputs, outputstates = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell,
                                                                    cell_bw=gru_bw_cell,
                                                                    inputs=embedding,
                                                                    sequence_length=length,
                                                                    initial_state_fw=gru_fw_cell.zero_state(BATCH_SIZE, dtype),
                                                                    initial_state_bw=gru_bw_cell.zero_state(BATCH_SIZE, dtype),
                                                                    dtype=dtype,)
            return outputs, outputstates
    def _build_keyword_encoder(self):
        with tf.name_scope('keyword'):
            self.keyword_id = tf.placeholder(shape=(BATCH_SIZE, None), dtype = tf.int32)
            self.keyword_length = tf.placeholder(shape = (BATCH_SIZE), dtype=tf.int32)
            _, outputstates = self.bidirectionRnn('keyword', self.keyword_id, self.keyword_length, tf.float64)
            self.keyword_encoded = tf.concat(outputstates, -1)
            print('keyword_encoded', self.keyword_encoded)
            # keyword_encoded : batch * numOfunits
    def _build_context_encoder(self):
        with tf.name_scope('context'):
            self.context_id = tf.placeholder(shape=(BATCH_SIZE, None), dtype=tf.int32)
            self.context_length = tf.placeholder(shape=(BATCH_SIZE), dtype=tf.int32)
            outputs, outputstates = self.bidirectionRnn('context', self.context_id, self.context_length, tf.float64)
            self.context_encoded = tf.concat(outputs, 2)
            self.context_encoded_state = tf.concat(outputstates, -1)
            # context_encoded : Batch * maxTime * numOfUnits
    '''def _build_decoder(self):
        with tf.name_scope('decoder'):
            decoder_gru_cell = tf.nn.rnn_cell.GRUCell(num_units=NUM_UNITS, dtype=tf.float64)
            context = tf.concat([tf.expand_dims(self.keyword_encoded, 1), self.context_encoded], 1)
            # context batch * maxtime * numOfUnits
            h = self.keyword_encoded
            # h batch * numOfUnits
            Ey = tf.nn.embedding_lookup(self.embedding, np.array([self.char_dict.word2int('^')] * BATCH_SIZE, dtype=np.int32))
            # Ey batchSize * numOfUnits
            self.w = tf.Variable(tf.truncated_normal([NUM_UNITS, WORD_DICT_SIZE],stddev=0.1, dtype=tf.float64), dtype=tf.float64, name = 'w')
            self.b = tf.Variable(tf.constant(0, shape=[WORD_DICT_SIZE], dtype=tf.float64), dtype=tf.float64, name='b')
            self.w1 = tf.Variable(tf.truncated_normal([NUM_UNITS, NUM_UNITS],stddev=0.1, dtype=tf.float64), dtype=tf.float64, name = 'w1')
            self.w2 = tf.Variable(tf.truncated_normal([NUM_UNITS, NUM_UNITS],stddev=0.1, dtype=tf.float64), dtype=tf.float64, name = 'w2')
            self.v = tf.Variable(tf.truncated_normal([NUM_UNITS], stddev=0.1, dtype=tf.float64), dtype = tf.float64, name='v')
            for i in range(LEN_PER_SENTENCE + 1):

                # get context
                h_ = tf.expand_dims(h, 1)
                score = (tf.tanh(tf.matmul(h_, self.w1) + tf.matmul(context, self.w2))) * self.v
                # score: batch maxtime numOfUnits
                score = tf.reduce_sum(score, -1)
                # batch maxtime
                score = tf.expand_dims(score, -1)
                # batch maxtime 1
                context_vec = score * context
                # batch maxtime numOfUnits
                context_vec = tf.reduce_sum(context_vec, 1)
                # batch numOfUnits

                input = tf.concat([Ey, context_vec], axis=-1)
                # batch 2 * numOfUnits
                h, _ = decoder_gru_cell.__call__(inputs=input, state=h)

                # get Eyprev
                logits = tf.matmul(h, self.w) + self.b
                # batch * wordDictSize
                prob = tf.math.softmax(logits, -1)
                # batch * wordDictSize
                y_pre = tf.argmax(prob, axis=-1)
                # batch * 1
                Ey = tf.nn.embedding_lookup(self.embedding, y_pre)
                # Ey: batch, numOfUnits

                self.y_prob = tf.expand_dims(prob, 1) if i == 0 else tf.concat([self.y_prob, tf.expand_dims(prob, 1)], 1)
                self.y_id = tf.expand_dims(y_pre, 1) if i == 0 else tf.concat([self.y_id, tf.expand_dims(y_pre, 1)], -1)
                # batch * (i + 1)'''
    def get_attention_vec(self, h, context):
        # h B * numOfUnits; context: B * maxtime * numOfUnits
        #print('h, context', h, context)
        score = self.v * tf.tanh(tf.expand_dims(h @ self.w1, 1) + context @ self.w2)
        #print('score1', score)
        score = tf.reduce_sum(score, axis=-1)
        #print('score2', score)
        # B * maxtime
        score = tf.nn.softmax(score, axis=-1) # the model must guarantee the maxtime are the same!!!
        # B * maxtime
        #print('score3', score)
        context = context * tf.expand_dims(score, axis=-1)
        context = tf.reduce_sum(context, axis=-2)
        # context, Batchsize * numOfUnits
        concat = tf.concat([context, h], axis=-1)
        # BatchSize * 2numOfUnits
        a = tf.tanh(tf.matmul(concat, self.wa))

        return a, score
    def _build_decoder_with_attention(self):
        with tf.name_scope('decoder_attention'):
            decoder_gru_cell = tf.nn.rnn_cell.GRUCell(num_units=NUM_UNITS, dtype=tf.float64)
            context = tf.concat([tf.expand_dims(self.keyword_encoded, 1), self.context_encoded], 1)
            # context : [batch, len(context) + 1, numOfUnits]
            hidden_state = self.context_encoded_state # refinable
            hidden_state_inference = self.context_encoded_state
            # decoder_init_state : [batchSize, numOfUnits]
            self.decoder_input_id = tf.placeholder(shape=(BATCH_SIZE, None), dtype=tf.int32)
            decoder_input = tf.nn.embedding_lookup(self.embedding, self.decoder_input_id)
            # decoder_input : [batchSize, len(sentence)+1, numOfUnits]

            # attention paras
            self.w1 = tf.Variable(tf.truncated_normal([NUM_UNITS, NUM_UNITS], stddev=0.1, dtype=tf.float64),
                                  dtype=tf.float64, name='w1')
            self.w2 = tf.Variable(tf.truncated_normal([NUM_UNITS, NUM_UNITS], stddev=0.1, dtype=tf.float64),
                                  dtype=tf.float64, name='w2')
            self.v = tf.Variable(tf.truncated_normal([NUM_UNITS], stddev=0.1, dtype=tf.float64), dtype=tf.float64,
                                 name='v')
            self.wa = tf.Variable(tf.truncated_normal([2*NUM_UNITS, NUM_UNITS], stddev=0.1, dtype=tf.float64), dtype=tf.float64,
                                 name='wa')
            attention_vec, _ = self.get_attention_vec(hidden_state, context)
            attention_vec_inference = attention_vec
            # B * U
            self.logits = tf.expand_dims(attention_vec, axis=1) # hold a pos
            # B *1* U
            for i in range(LEN_PER_SENTENCE + 1):
                input = decoder_input[:,i,:]
                # input : B * U
                input = tf.concat([input, attention_vec], axis=-1)
                # input: B * 2U
                hidden_state, _ = decoder_gru_cell(inputs=input, state=hidden_state)
                attention_vec, _ = self.get_attention_vec(hidden_state, context)
                self.logits = tf.concat([self.logits, tf.expand_dims(attention_vec, axis=1)], axis=1)
            self.logits = self.logits[:,1:,:]
            # self.logits BatchSize * maxtime * numOfUnits
            self.w = tf.Variable(tf.truncated_normal([NUM_UNITS, WORD_DICT_SIZE], stddev=0.1, dtype=tf.float64),
                                  dtype=tf.float64, name='w')
            self.b = tf.Variable(tf.zeros(shape=(WORD_DICT_SIZE), dtype=tf.float64),
                                  dtype=tf.float64, name='b')
            self.prob = tf.nn.softmax((tf.matmul(self.logits, self.w) + self.b), axis=-1)
            # B * maxtime * U



            # doing inference
            input_inference = tf.nn.embedding_lookup(self.embedding, [self.char_dict.word2int('^')] * BATCH_SIZE)
            input_inference = tf.concat([input_inference, attention_vec_inference], axis=-1)
            for i in range(LEN_PER_SENTENCE + 1):
                hidden_state_inference, _ = decoder_gru_cell(inputs = input_inference, state=hidden_state_inference)
                attention_vec_inference, s = self.get_attention_vec(hidden_state_inference, context)
                self.prob_inference = tf.nn.softmax(tf.matmul(attention_vec_inference, self.w) + self.b, axis=-1)
                self.id_inference = tf.argmax(self.prob_inference, axis=-1)
                embedding_inference = tf.nn.embedding_lookup(self.embedding, self.id_inference)
                input_inference = tf.concat([embedding_inference, attention_vec_inference], axis=-1)
                if i == 0:
                    self.id_inference_in_sum = tf.expand_dims(self.id_inference, 1)
                    self.prob_inference_in_sum = tf.expand_dims(self.prob_inference,1)
                    self.score = tf.expand_dims(s[0], 0)
                else:
                    self.id_inference_in_sum = tf.concat([self.id_inference_in_sum, tf.expand_dims(self.id_inference, 1)], axis=1)
                    self.prob_inference_in_sum = tf.concat([self.prob_inference_in_sum, tf.expand_dims(self.prob_inference, 1)],axis=1)
                    self.score = tf.concat([self.score, tf.expand_dims(s[0], 0)], axis=0)
                    #

    def _build_soft_max(self):
        self.label_id = tf.placeholder(shape=(BATCH_SIZE, None), dtype=tf.int32)
        one_hot = tf.one_hot(indices=self.label_id, depth=WORD_DICT_SIZE, axis=-1, dtype=tf.float64)
        # one_hot batch maxtime WordDictSize
        self.loss = tf.reduce_mean(tf.reduce_sum(-(tf.log(self.prob) * one_hot), axis=-1))  # 这里也依赖于maxtime一样，但对于古诗没啥问题
    def _build_optimizer(self):
        lr = tf.Variable(initial_value=tf.constant((1e-3), dtype=tf.float64), dtype=tf.float64, name='lr')
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def _build_layers(self):
        self._build_keyword_encoder()
        self._build_context_encoder()
        #self._build_decoder()
        self._build_decoder_with_attention()
        self._build_soft_max()
        self._build_optimizer()

    def __init__(self):
        self.char_dict = wordDict() # 词典
        self.char2vec = word2Vec() # 词向量 gensim
        self.embedding = self.char2vec.get_embedding()
        self._build_layers()
        self.model_path = [None, './models/new',  './models/thirdSentence/', './models/new']
        self.loss_path = [f'./result/loss/{i}_sentence' for i in range(4)]
        self.attention_img_path = [f'./result/attention/']

    def get_id_length(self, sentences):
        id = [[self.char_dict.word2int(word) for word in sentence.strip().split()] for sentence in sentences]
        length = [len(sentence) for sentence in id]
        max_length = max(length)
        id = [i + [0] * (max_length - len(i)) for i in id]
        return np.array(id, dtype=np.int32), np.array(length, dtype=np.int32)

    def train(self, epoch, mode, phase, data_mode = -1, data_phase = -1, seq=-1):
        if data_mode == -1:
            data_mode = mode
            data_phase = phase
        # keyword_id, keyword_length, context_id, context_length, label_id
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_path_l = model_load_path[mode]
            model_path_s = model_save_path[mode]
            saver = tf.train.Saver(max_to_keep = 5, sharded=False)
            try:
                saver.restore(sess, tf.train.latest_checkpoint(model_path_s))
            except:
                print('mode not found')
                model_load_path[mode] = model_path_s
            for i in range(epoch):
                # time.sleep(10)
                print(f'begin epoch {i}')
                for keywords, ground_truth, contexts, sentences in batch_train_data(BATCH_SIZE, data_mode, data_phase):
                    keyword_id, keyword_length = self.get_id_length(keywords)
                    ground_truth_id, _ = self.get_id_length(ground_truth)
                    context_id, context_length = self.get_id_length(contexts)
                    label_id, _ = self.get_id_length(sentences)
                    _, loss = sess.run((self.train_op, self.loss), feed_dict={
                        self.keyword_id: keyword_id,
                        self.keyword_length:keyword_length,
                        self.decoder_input_id:ground_truth_id,
                        self.context_id: context_id,
                        self.label_id: label_id,
                        self.context_length: context_length,

                    })
                if (i + 1) % 5 == 0:
                    time.sleep(30)
                    with open(result_save_path[mode] + '/loss.txt', 'a') as f:
                        f.write(f'{i}\t{loss}\n')
                    print(loss)
                    saver.save(sess, model_save_path[mode] + f'/{seq}_{i}', global_step=i + 1)
                    poem = self.generate(['春', '秋', '冬', '夏'], i, result_save_path[mode], model_save_path[mode], f'{seq}_{i}')
                    print(poem)
                    with open(result_save_path[mode] + '/poem.txt', 'a') as f:
                        f.write(poem + '\n')
    def draw(self, M, id, fig):
        # M = (len(sentence), len(context))
        ax = fig.add_subplot(id)
        ax.imshow(M, cmap=plt.cm.gray)
    def generate(self, keywords, epoch=0, img_save_path = './result', model_path = '', name = ''):
        pron_dict = PronDict()
        fig = plt.figure()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            context = '^'
            i = 0
            for keyword in keywords:
                i += 1
                keyword_id, keyword_length = self.get_id_length([keyword] * BATCH_SIZE)
                context_id, context_length = self.get_id_length([context] * BATCH_SIZE)
                y_id, y_prob, score = sess.run([self.id_inference_in_sum, self.prob_inference_in_sum, self.score], feed_dict={
                    self.keyword_id: keyword_id,
                    self.keyword_length: keyword_length,
                    self.context_id: context_id,
                    self.context_length: context_length
                })
                self.draw(score, 220 + i, fig)
                if i == 2:
                    memo_word = self.char_dict.int2word(y_id[0][4])
                if i == 4:
                    temp = y_prob[4]
                    argmax = -1
                    tempMax = 0
                    for j in range(WORD_DICT_SIZE):
                        word = self.char_dict.int2word(j)
                        if word == '^' or word == '$':
                            continue
                        if pron_dict.co_rhyme(word, memo_word) and y_prob[0][4][j] > tempMax:
                            argmax = j
                            tempMax = y_prob[0][4][j]
                    y_id[0][4] = argmax

                context += ' ' + ' '.join([self.char_dict.int2word(i) for i in list(y_id[0])])
            plt.savefig(img_save_path + f'/attention_at_epoch{epoch}' + name + '.png')
            return context
    def generate_by_multiple_models(self, keywords, epoch=0, img_save_path = './result', model_path = '', name = ''):
        pron_dict = PronDict()
        fig = plt.figure()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            context = '^'
            i = 0
            for j, keyword in enumerate(keywords):
                model_path = model_save_path[j+1]
                ckpt_state = tf.train.get_checkpoint_state(model_save_path[1])
                module_file = tf.train.latest_checkpoint(model_path)
                '''path = ckpt_state.model_checkpoint_path
                path = path.replace('\\','/')'''
                saver = tf.train.Saver()
                saver.restore(sess, module_file)
                i += 1
                keyword_id, keyword_length = self.get_id_length([keyword] * BATCH_SIZE)
                context_id, context_length = self.get_id_length([context] * BATCH_SIZE)
                y_id, y_prob, score = sess.run([self.id_inference_in_sum, self.prob_inference_in_sum, self.score], feed_dict={
                    self.keyword_id: keyword_id,
                    self.keyword_length: keyword_length,
                    self.context_id: context_id,
                    self.context_length: context_length
                })
                self.draw(score, 220 + i, fig)
                if i == 2:
                    memo_word = self.char_dict.int2word(y_id[0][4])
                if i == 4:
                    temp = y_prob[4]
                    argmax = -1
                    tempMax = 0
                    for j in range(WORD_DICT_SIZE):
                        word = self.char_dict.int2word(j)
                        if word == '^' or word == '$':
                            continue
                        if pron_dict.co_rhyme(word, memo_word) and y_prob[0][4][j] > tempMax:
                            argmax = j
                            tempMax = y_prob[0][4][j]
                    y_id[0][4] = argmax

                context += ' ' + ' '.join([self.char_dict.int2word(i) for i in list(y_id[0])])
            plt.savefig(img_save_path + f'/attention_at_epoch{epoch}' + name + '.png')
            return context
if __name__ == '__main__':
    modes = [[[0,20], [1,20]], [[0,30]],[[0,20]],[[0,20],[1,20]],[[0,30],[1,20]]]
    generator = Generator()
    # generator.generate_by_multiple_models(['春','桃','秋','菊'])
    '''for i, _ in enumerate(modes):
        if i < 4:
            continue
        for j in _:
            if i == 3 and j[0] == 0:
                continue
            t1 = time.time()
            generator.train(j[1], i, j[0])
            print(f'mode{i}phase{j[0]} finishes at {time.time() - t1}')
    trained [30,1,0], [30,3,0],[30,3,1],[10,4,2],[10,4,0],[10,4,1],[20, 2, 1],[20,2,0],[10,2,1],[10,2,0]
    '''
    '''
    0: ['pairs.txt', 'pair.txt'],
    1: ['keyword1.txt'],
    2: ['keyword2.txt', 'pair.txt'],
    3: ['keyword3.txt','nokeyword3.txt' ],
    4: ['keyword4.txt','nokeyword4.txt' , 'pair.txt']
    [30, 0, 0, 0, 0],[30,0,0,0,1], [40, 1, 0, 1, 0], [40,2,0,2,0], [30,3,0,3,1], [30,3,0,3,0], [30,4,0,4,1], [30,4,0,4,0]
    '''
    epoch_mode_phase = [[10,1,0,2,0],[10,1,0,1,0],[10,2,0,1,0],[10,2,0,2,0],[10,3,0,1,0],[10,3,0,2,0],[10,3,0,3,0],[10,4,0,1,0],[10,4,0,2,0],[10,4,0,3,0],[10,4,0,4,0]]
    '''for i in range(1,4):
        for j in range(1):
            epoch_mode_phase.append([1,i,0,j,0])'''
    i = 16
    for epoch, mode, phase, data_mode, data_phase in epoch_mode_phase:
        generator.train(epoch=epoch,mode=mode,phase=phase, data_mode=data_mode, data_phase=data_phase, seq=i)
        i+=1
    # generator.train(100)
    poem = generator.generate_by_multiple_models(['春','桃','夏','菊'])
    print(poem)
    #generator.train(100)

