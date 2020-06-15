# coding=utf-8

"""
a lstm + dssm implementation for short text match using tensroflow library

"""

import random
import tensorflow as tf
from tensorflow.contrib import rnn


class Dssm(object):
    def __init__(self, num_lstm_units, batch_size, negtive_size, SOFTMAX_R, learning_rate, vocab_size,
                 embedding_size=100, use_same_cell=False):
        """Constructor for Dssm

        Args:
          num_lstm_units: int, The number of units in the LSTM cell.
          batch_size: int, The number of examples in each batch
          negtive_size: int, The number of negative example.
          SOFTMAX_R: float, A regulatory factor for cosine similarity
          learning_rate: float, learning rate
          vocab_size: int, The number of vocabulary
          embedding_size: int the size of vocab embedding
          use_same_cell: (optional) bool whether to use same cell for fw, bw lstm, default is false
        """
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")  # [batch_size, seq_len]
        self.length_x = tf.placeholder(tf.int32, [None, ], name="length_x")  # [batch_size, ]
        self.input_y = tf.placeholder(tf.int32, [None, None], name="input_y")  # [batch_size, seq_len]
        self.length_y = tf.placeholder(tf.int32, [None, ], name="length_y")  # [batch_size, ]
        self.lstm_fw_cell = rnn.BasicLSTMCell(num_lstm_units)
        if use_same_cell:
            self.lstm_bw_cell = self.lstm_fw_cell
        else:
            self.lstm_bw_cell = rnn.BasicLSTMCell(num_lstm_units)
        with tf.name_scope("keep_prob"):
            self.lstm_fw_cell = rnn.DropoutWrapper(self.lstm_fw_cell, input_keep_prob=self.keep_prob,
                                                   output_keep_prob=self.keep_prob)
            self.lstm_bw_cell = rnn.DropoutWrapper(self.lstm_bw_cell, input_keep_prob=self.keep_prob,
                                                   output_keep_prob=self.keep_prob)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # one_gram
            self.vocab = tf.get_variable('w', [vocab_size, embedding_size])
            self.lstm_input_embedding_x = tf.nn.embedding_lookup(self.vocab,
                                                                 self.input_x)  # [batch_size, seq_len, embedding_size]
            self.lstm_input_embedding_y = tf.nn.embedding_lookup(self.vocab,
                                                                 self.input_y)  # [batch_size, seq_len, embedding_size]

        with tf.name_scope('representation'):
            self.states_x = \
            tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, self.lstm_bw_cell, self.lstm_input_embedding_x,
                                            self.length_x,
                                            dtype=tf.float32)[1]
            self.output_x = tf.concat([self.states_x[0][1], self.states_x[1][1]], 1)  # [batch_size, 2*num_lstm_units]
            self.states_y = \
            tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, self.lstm_bw_cell, self.lstm_input_embedding_y,
                                            self.length_y,
                                            dtype=tf.float32)[1]
            self.output_y = tf.concat([self.states_y[0][1], self.states_y[1][1]], 1)  # [batch_size, 2*num_lstm_units]
            self.q_y_raw = tf.nn.relu(self.output_x, name="q_y_raw")  # [batch_size, num_lstm_units*2]
            print("self.q_y_raw: " + str(self.q_y_raw))
            self.qs_y_raw = tf.nn.relu(self.output_y, name="qs_y_raw")  # [batch_size, num_lstm_units*2]
            print("self.qs_y_raw: " + str(self.qs_y_raw))

        with tf.name_scope('rotate'):
            temp = tf.tile(self.qs_y_raw, [1, 1])  # [batch_size, num_lstm_units*2]
            self.qs_y = tf.tile(self.qs_y_raw, [1, 1])  # [batch_size, num_lstm_units*2]
            for i in range(negtive_size):
                rand = int((random.random() + i) * batch_size / negtive_size)
                if rand == 0:
                    rand = rand + 1
                rand_qs_y1 = tf.slice(temp, [rand, 0], [batch_size - rand, -1])  # [batch_size - rand, num_lstm_units*2]
                rand_qs_y2 = tf.slice(temp, [0, 0], [rand, -1])  # [rand, num_lstm_units*2]
                self.qs_y = tf.concat(axis=0, values=[self.qs_y, rand_qs_y1,
                                                      rand_qs_y2])  # [batch_size*(negtive_size+1), num_lstm_units*2]

        with tf.name_scope('sim'):
            # cosine similarity
            q_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.q_y_raw), 1, True)),
                             [negtive_size + 1, 1])  # [(negtive_size + 1) * batch_size, 1]
            qs_norm = tf.sqrt(tf.reduce_sum(tf.square(self.qs_y), 1, True))  # [batch_size*(negtive_size+1), 1]
            prod = tf.reduce_sum(tf.multiply(tf.tile(self.q_y_raw, [negtive_size + 1, 1]), self.qs_y), 1,
                                 True)  # [batch_size*(negtive_size + 1), 1]
            norm_prod = tf.multiply(q_norm, qs_norm)  # [batch_size*(negtive_size + 1), 1]
            sim_raw = tf.truediv(prod, norm_prod)  # [batch_size*(negtive_size + 1), 1]
            self.cos_sim = tf.transpose(tf.reshape(tf.transpose(sim_raw), [negtive_size + 1,
                                                                           batch_size])) * SOFTMAX_R  # [batch_size, negtive_size + 1]

        with tf.name_scope('loss'):
            # train Loss
            self.prob = tf.nn.softmax(self.cos_sim)  # [batch_size, negtive_size + 1]
            self.hit_prob = tf.slice(self.prob, [0, 0], [-1, 1])  # [batch_size, 1]  #positive
            raw_loss = -tf.reduce_sum(tf.log(self.hit_prob)) / batch_size
            self.loss = raw_loss

        with tf.name_scope('training'):
            # optimizer
            self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 1000, 0.96, staircase=True)
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                  global_step=self.global_step)

        # acc for test data
        with tf.name_scope('cosine_similarity_pre'):
            # Cosine similarity
            self.q_norm_pre = tf.sqrt(tf.reduce_sum(tf.square(self.q_y_raw), 1, True))  # b*1
            self.qs_norm_pre = tf.transpose(tf.sqrt(tf.reduce_sum(tf.square(self.qs_y_raw), 1, True)))  # 1*sb
            self.prod_nu_pre = tf.matmul(self.q_y_raw, tf.transpose(self.qs_y_raw))  # b*sb
            self.norm_prod_de = tf.matmul(self.q_norm_pre, self.qs_norm_pre)  # b*sb
            self.cos_sim_pre = tf.truediv(self.prod_nu_pre, self.norm_prod_de) * SOFTMAX_R  # b*sb

        with tf.name_scope('prob_pre'):
            self.prob_pre = tf.nn.softmax(self.cos_sim_pre)  # b*sb
            # self.hit_prob_pre = tf.slice(self.prob_pre, [0, 0], [-1, 1])  # [batch_size, 1]  #positive
            # self.test_loss = -tf.reduce_sum(tf.log(self.hit_prob_pre)) / batch_size
