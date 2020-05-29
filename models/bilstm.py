# coding=utf-8

"""
a bi-lstm implementation for short text classification using tensroflow library

"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn


class BiLSTM(object):

    def __init__(self, FLAGS):
        """Constructor for BiLSTM

                Args:
                  FLAGS: tf.app.flags, you can see the FLAGS of run_bi_lstm.py
        """
        self.input_x = tf.placeholder(tf.int64, [None, FLAGS.seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None, ], name="input_y")
        self.x_len = tf.placeholder(tf.int64, [None, ], name="x_len")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("embedding", initializer=tf.orthogonal_initializer()):
            with tf.device('/cpu:0'):
                # word embedding table
                self.vocab = tf.get_variable('w', [FLAGS.vocab_size, FLAGS.embedding_size])
                embedded = tf.nn.embedding_lookup(self.vocab, self.input_x)  # [batch_size, seq_length, embedding_size]
                inputs = tf.split(embedded, FLAGS.seq_length,
                                  1)  # [[batch_size, 1, embedding_size], [batch_size, 1, embedding_size], number is seq_length]
                inputs = [tf.squeeze(input_, [1]) for input_ in
                          inputs]  # [[batch_size, embedding_size], [batch_size, embedding_size], number is seq_length]

        with tf.variable_scope("encoder", initializer=tf.orthogonal_initializer()):
            lstm_fw_cell = rnn.BasicLSTMCell(FLAGS.num_units)
            lstm_bw_cell = rnn.BasicLSTMCell(FLAGS.num_units)
            lstm_fw_cell_stack = rnn.MultiRNNCell([lstm_fw_cell] * FLAGS.lstm_layers, state_is_tuple=True)
            lstm_bw_cell_stack = rnn.MultiRNNCell([lstm_bw_cell] * FLAGS.lstm_layers, state_is_tuple=True)
            lstm_fw_cell_stack = rnn.DropoutWrapper(lstm_fw_cell_stack, input_keep_prob=self.dropout_keep_prob,
                                                    output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell_stack = rnn.DropoutWrapper(lstm_bw_cell_stack, input_keep_prob=self.dropout_keep_prob,
                                                    output_keep_prob=self.dropout_keep_prob)
            self.outputs, self.fw_st, self.bw_st = rnn.static_bidirectional_rnn(lstm_fw_cell_stack, lstm_bw_cell_stack,
                                                                                inputs, sequence_length=self.x_len,
                                                                                dtype=tf.float32)  # multi-layer
            # only use the last layer
            last_layer_no = FLAGS.lstm_layers - 1
            self.states = tf.concat([self.fw_st[last_layer_no].h, self.bw_st[last_layer_no].h],
                                    1)  # [batchsize, (num_units * 2)]

        attention_size = 2 * FLAGS.num_units
        with tf.variable_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([2 * FLAGS.num_units, attention_size], stddev=0.1),
                                      name='attention_w')  # [num_units * 2, num_units * 2]
            attention_b = tf.get_variable("attention_b", initializer=tf.zeros([attention_size]))  # [num_units * 2]
            u_list = []
            for index in range(FLAGS.seq_length):
                u_t = tf.tanh(tf.matmul(self.outputs[index], attention_w) + attention_b)  # [batchsize, num_units * 2]
                u_list.append(u_t)  # seq_length * [batchsize, num_units * 2]
            u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1),
                              name='attention_uw')  # [num_units * 2, 1]
            attn_z = []
            for index in range(FLAGS.seq_length):
                z_t = tf.matmul(u_list[index], u_w)
                attn_z.append(z_t)  # seq_length * [batchsize, 1]
            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)  # [batchsize, seq_length]
            alpha = tf.nn.softmax(attn_zconcat)  # [batchsize, seq_length]
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            alpha_trans = tf.reshape(tf.transpose(alpha, [1, 0]),
                                     [FLAGS.seq_length, -1, 1])  # [seq_length, batchsize, 1]
            self.final_output = tf.reduce_sum(self.outputs * alpha_trans, 0)  # [batchsize, num_units * 2]

        with tf.variable_scope("output_layer"):
            weights = tf.get_variable("weights", [2 * FLAGS.num_units, FLAGS.label_size])
            biases = tf.get_variable("biases", initializer=tf.zeros([FLAGS.label_size]))

        with tf.variable_scope("acc"):
            # use attention
            self.logits = tf.matmul(self.final_output, weights) + biases  # [batchsize, label_size]
            # not use attention
            # self.logits = tf.matmul(self.states, weights) + biases
            self.prediction = tf.nn.softmax(self.logits, name="prediction_softmax")  # [batchsize, label_size]
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))
            self.global_step = tf.train.get_or_create_global_step()
            self.correct = tf.equal(tf.argmax(self.prediction, 1), self.input_y)
            self.acc = tf.reduce_mean(tf.cast(self.correct, tf.float32))
            _, self.arg_index = tf.nn.top_k(self.prediction, k=FLAGS.label_size)  # [batch_size, label_size]

        with tf.variable_scope('training'):
            # optimizer
            self.learning_rate = tf.train.exponential_decay(FLAGS.lr, self.global_step, 200, 0.96, staircase=True)
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                  global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    def export_model(self, export_path, sess):
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        tensor_info_x = tf.saved_model.utils.build_tensor_info(self.input_x)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(self.prediction)
        tensor_info_len = tf.saved_model.utils.build_tensor_info(self.x_len)
        tensor_dropout_keep_prob = tf.saved_model.utils.build_tensor_info(self.dropout_keep_prob)  # 1.0 for inference
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input': tensor_info_x, 'sen_len': tensor_info_len,
                        'dropout_keep_prob': tensor_dropout_keep_prob},
                outputs={'output': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        legacy_init_op = None
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={'prediction': prediction_signature, },
                                             legacy_init_op=legacy_init_op, clear_devices=True, saver=self.saver)
        builder.save()
