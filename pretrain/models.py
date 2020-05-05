# -*- coding: utf-8 -*-
"""
models implementation for runing pretrain and finetune language models using tensroflow library

"""
import os
import sys
import tensorflow as tf
import numpy as np
import collections
from collections import namedtuple
import re

class BiDirectionalLmModel(object):
    def __init__(self, input_arg, other_arg_dict):
        self.lstm_dim = input_arg.lstm_dim
        self.embedding_dim = input_arg.embedding_dim
        self.layer_num = input_arg.layer_num
        self.token_num = other_arg_dict['token_num']
        if 2 * self.lstm_dim != self.embedding_dim:
            print('please set the 2 * lstm_dim == embedding_dim')
            assert False

    def build(self, for_the_other_seq=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=True if for_the_other_seq else False):
            if for_the_other_seq:
                ph_suffix = "_1"
            else:
                ph_suffix = ""
            self.ph_tokens = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ph_tokens" + ph_suffix)  # [batch, len]
            self.ph_length = tf.placeholder(dtype=tf.int32, shape=[None], name="ph_length" + ph_suffix)  # [batch]
            self.ph_dropout_rate = tf.placeholder(dtype=tf.float32, shape=None, name="ph_dropout_rate" + ph_suffix)

            self.v_token_embedding = tf.get_variable(name='v_token_embedding', shape=[self.token_num, self.embedding_dim],
                                                     dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            seq_embedding = tf.nn.embedding_lookup(self.v_token_embedding, self.ph_tokens)  # [batch, len, embedding_dim]
            seq_embedding = tf.nn.dropout(seq_embedding, keep_prob=1 - self.ph_dropout_rate)

            last_output = seq_embedding
            cur_state = None
            for layer in range(1, self.layer_num + 1):
                fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_dim, name="fw_layer_" + str(layer))
                bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_dim, name="bw_layer_" + str(layer))
                cur_output, cur_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, last_output, self.ph_length, dtype=tf.float32)  # [batch, length, dim]
                cur_output = tf.concat(cur_output, -1)    # [batch, length, 2 * dim]
                cur_output = tf.nn.dropout(cur_output, keep_prob=1 - self.ph_dropout_rate)
                last_output = tf.contrib.layers.layer_norm(last_output + cur_output, begin_norm_axis=-1)  # add and norm

            output = tf.layers.dense(last_output, self.embedding_dim, activation=tf.tanh,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())  # [batch, length, 2 * dim]
            output = tf.nn.dropout(output, keep_prob=1 - self.ph_dropout_rate)

            # sequence output
            self.output = tf.contrib.layers.layer_norm(last_output + output, begin_norm_axis=-1)  # add and norm [batch, length, 2 * dim]

            # concat pool output
            self.concat_pool_output = tf.concat([cur_state[0][1], cur_state[1][1]], -1)  # [batch, 2 * dim]

            seq_len = tf.shape(self.ph_tokens)[1]

            # max pool output
            mask = tf.expand_dims(tf.cast(tf.sequence_mask(self.ph_length, maxlen=seq_len), tf.float32), axis=2)  # [batch, len, 1]
            mask = (1 - mask) * -1e5
            self.max_pool_output = tf.reduce_max(self.output + mask, axis=1, keepdims=False)  # [batch, 2 * dim]

            # mean pool output
            mask = tf.expand_dims(tf.cast(tf.sequence_mask(self.ph_length, maxlen=seq_len), tf.float32), axis=2)  # [batch, len, 1]
            self.mean_pool_output = tf.reduce_sum(self.output * mask, axis=1, keepdims=False) / \
                                    tf.expand_dims(tf.cast(self.ph_length, tf.float32), axis=1)  # [batch, 2 * dim]

def create_bidirectional_lm_training_op(input_arg, other_arg_dict):
    loss_op, model = create_bidirectional_lm_model(input_arg, other_arg_dict)
    train_op, learning_rate_op = create_optimizer(loss_op, input_arg.learning_rate, input_arg.train_step, input_arg.warmup_step, input_arg.clip_norm, input_arg.weight_decay)
    model.loss_op = loss_op
    model.train_op = train_op
    model.learning_rate_op = learning_rate_op
    return model

def create_bidirectional_lm_model(input_arg, other_arg_dict):
    model = BiDirectionalLmModel(input_arg, other_arg_dict)
    model.build()
    max_predictions_per_seq = input_arg.max_predictions_per_seq

    model.global_step = tf.train.get_or_create_global_step()
    model.ph_labels = tf.placeholder(dtype=tf.int32, shape=[None, max_predictions_per_seq], name="ph_labels")  # [batch, max_predictions_per_seq]
    model.ph_positions = tf.placeholder(dtype=tf.int32, shape=[None, max_predictions_per_seq], name="ph_positions")  # [batch, max_predictions_per_seq]
    model.ph_weights = tf.placeholder(dtype=tf.float32, shape=[None, max_predictions_per_seq], name="ph_weights")  # [batch, max_predictions_per_seq]

    real_output = gather_indexes(model.output, model.ph_positions)  # [batch * max_predictions_per_seq, embedding_dim]

    bias = tf.get_variable("bias", shape=[model.token_num], initializer=tf.zeros_initializer())
    logits = tf.matmul(real_output, model.v_token_embedding, transpose_b=True)  # tie weight  [None, token_num]
    logits = tf.nn.bias_add(logits, bias)

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(tf.reshape(model.ph_labels, [-1]), depth=model.token_num, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])  # [batch * max_predictions_per_seq, 1]
    weights = tf.reshape(model.ph_weights, [-1])  # [batch * max_predictions_per_seq, 1]
    loss = (tf.reduce_sum(weights * per_example_loss)) / (tf.reduce_sum(weights) + 1e-5)

    return loss, model

def create_optimizer(loss, init_lr=5e-5, num_train_steps=1000000, num_warmup_steps=20000, clip_nom=1.0, weight_decay=0.01):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(learning_rate, global_step, num_train_steps, end_learning_rate=0.0, power=1.0, cycle=False)  # linear warmup

    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
        warmup_learning_rate = init_lr * tf.cast(global_steps_int, tf.float32) / tf.cast(warmup_steps_int, tf.float32)
        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=weight_decay, learning_rate=learning_rate)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_nom)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    return train_op, learning_rate

def gather_indexes(seq_output, positions):
    batch_size = tf.shape(seq_output)[0]
    length = tf.shape(seq_output)[1]
    dim = tf.shape(seq_output)[2]

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * length, [-1, 1])
    output_tensor = tf.gather(tf.reshape(seq_output, [-1, dim]), tf.reshape(positions + flat_offsets, [-1]))
    return output_tensor  # [batch * max_predictions_per_seq, dim]

# finetune model utils
def create_finetune_classification_training_op(input_arg, other_arg_dict):
    model = create_finetune_classification_model(input_arg, other_arg_dict)
    repre = model.max_pool_output  # [batch, 2 * dim]

    model.ph_labels = tf.placeholder(dtype=tf.int32, shape=[None], name="ph_labels")  # [batch]
    logits = tf.layers.dense(repre, other_arg_dict['label_num'], kernel_initializer=tf.contrib.layers.xavier_initializer(), name='logits')
    model.softmax_op = tf.nn.softmax(logits, -1)
    model.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=model.ph_labels), -1)
    model.global_step_op = tf.train.get_or_create_global_step()

    print("learning_rate : ", input_arg.learning_rate)
    if input_arg.opt_type == "sgd":
        print("use sgd")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=input_arg.learning_rate)
    elif input_arg.opt_type == "adagrad":
        print("use adagrad")
        optimizer = tf.train.AdagradOptimizer(learning_rate=input_arg.learning_rate)
    elif input_arg.opt_type == "adam":
        print("use adam")
        optimizer = tf.train.AdamOptimizer(learning_rate=input_arg.learning_rate)
    else:
        assert False

    list_g_v_pair = optimizer.compute_gradients(model.loss_op)
    model.train_op = optimizer.apply_gradients(list_g_v_pair, global_step=model.global_step_op)

    return model
    
def create_finetune_classification_model(input_arg, other_arg_dict):
    model = BiDirectionalLmModel(input_arg, other_arg_dict)

    model.build()

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if input_arg.init_checkpoint:
        print("init from checkpoint!")
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, input_arg.init_checkpoint)
        tf.train.init_from_checkpoint(input_arg.init_checkpoint, assignment_map)

    print("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        print("name = %s, shape = %s%s" % (var.name, var.shape, init_string))

    return model

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    name_to_variable = collections.OrderedDict()  # trainable variables
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    assignment_map = collections.OrderedDict()
    initialized_variable_names = {}
    init_vars = tf.train.list_variables(init_checkpoint)  # variables in checkpoint
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names