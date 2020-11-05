# -*- coding: utf-8 -*-
"""
models implementation for runing pretrain and finetune language models using tensroflow library

"""
import tensorflow as tf
import collections
import re
import math


class BiDirectionalLmModel(object):
    """Constructor for BiDirectionalLmModel
            Args:
              lstm_dim: int, The number of units in the LSTM cell.
              embedding_dim: int, The size of vocab embedding
              layer_num: int, The number of LSTM layer.
              token_num: int, The number of Token
              input_arg: dict, Args of inputs
            """
    def __init__(self, input_arg, other_arg_dict):
        self.lstm_dim = input_arg.lstm_dim
        self.embedding_dim = input_arg.embedding_dim
        self.layer_num = input_arg.layer_num
        self.token_num = other_arg_dict["token_num"]
        self.input_arg = input_arg
        if 2 * self.lstm_dim != self.embedding_dim:
            tf.logging.info('please set the 2 * lstm_dim == embedding_dim')
            assert False

    #Build graph for SPTM
    def build(self):
        assert self.input_arg.representation_type in ["lstm", "transformer"]
        self.ph_tokens = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ph_tokens")  # [batch_size, seq_length]
        self.ph_length = tf.placeholder(dtype=tf.int32, shape=[None], name="ph_length")  # [batch_size]
        self.ph_dropout_rate = tf.placeholder(dtype=tf.float32, shape=None, name="ph_dropout_rate")
        self.ph_input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ph_input_mask")  #[batch_size, seq_length]

        self.v_token_embedding = tf.get_variable(name="v_token_embedding",
                                                 shape=[self.token_num, self.embedding_dim],
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())  #[token_num, embedding_dim]
        seq_embedding = tf.nn.embedding_lookup(self.v_token_embedding,
                                               self.ph_tokens)  # [batch_size, seq_length, embedding_dim]

        if self.input_arg.representation_type == "lstm":
            tf.logging.info("representation using lstm ...........................")
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                seq_embedding = tf.nn.dropout(seq_embedding, keep_prob=1 - self.ph_dropout_rate)
                last_output = seq_embedding
                cur_state = None
                for layer in range(1, self.layer_num + 1):
                    fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_dim, name="fw_layer_" + str(layer))
                    bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_dim, name="bw_layer_" + str(layer))
                    cur_output, cur_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, last_output, self.ph_length,
                                                                        dtype=tf.float32)  # [batch, length, dim]
                    cur_output = tf.concat(cur_output, -1)  # [batch, length, 2 * dim]
                    cur_output = tf.nn.dropout(cur_output, keep_prob=1 - self.ph_dropout_rate)
                    last_output = tf.contrib.layers.layer_norm(last_output + cur_output, begin_norm_axis=-1)  # add and norm

                output = tf.layers.dense(last_output, self.embedding_dim, activation=tf.tanh,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())  # [batch, length, 2 * dim]
                output = tf.nn.dropout(output, keep_prob=1 - self.ph_dropout_rate)

                # sequence output
                self.output = tf.contrib.layers.layer_norm(last_output + output,
                                                       begin_norm_axis=-1)  # add and norm [batch, length, 2 * dim]

                # max pool output
                seq_len = tf.shape(self.ph_tokens)[1]
                mask = tf.expand_dims(tf.cast(tf.sequence_mask(self.ph_length, maxlen=seq_len), tf.float32),
                                  axis=2)  # [batch, len, 1]
                mask = (1 - mask) * -1e5
                self.max_pool_output = tf.reduce_max(self.output + mask, axis=1, keepdims=False)  # [batch, 2 * dim]
        elif self.input_arg.representation_type == "transformer":
            tf.logging.info("representation using transformer ...........................")
            input_shape = self.get_shape_list(seq_embedding)  # [batch_size, seq_length, embedding_size]
            batch_size = input_shape[0]
            seq_length = input_shape[1]
            embedding_size = input_shape[2]

            with tf.variable_scope("pos_embeddings"):
                all_position_embeddings = tf.get_variable(
                    name="position_embeddings",
                    shape=[self.input_arg.max_position_embeddings, embedding_size],
                    initializer=tf.truncated_normal_initializer(stddev=self.input_arg.initializer_range))  # [max_position_embeddings, embedding_size]
                position_embeddings = tf.slice(all_position_embeddings, [0, 0], [seq_length, -1])  # [seq_length, embedding_size]
                position_embeddings = tf.reshape(position_embeddings, [1, seq_length, embedding_size])  # [1, seq_length, embedding_size]
                seq_embedding += position_embeddings  # [batch_size, seq_length, embedding_size]
                seq_embedding = tf.contrib.layers.layer_norm(seq_embedding, begin_norm_axis=-1, begin_params_axis=-1) #[batch_size, seq_length, embedding_size]

            with tf.variable_scope("encoder"):
                assert self.input_arg.hidden_size % self.input_arg.num_attention_heads == 0
                attention_head_size = self.input_arg.hidden_size // self.input_arg.num_attention_heads
                self.all_layer_outputs = []
                if embedding_size != self.input_arg.hidden_size:
                    hidden_output = self.dense_layer_2d(seq_embedding, self.input_arg.hidden_size, None, name="embedding_to_hidden")
                else:
                    hidden_output = seq_embedding  # [batch_size, seq_length, hidden_size]

                with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
                    for layer_id in range(self.input_arg.num_hidden_layers):
                        with tf.name_scope("layer_%d" % layer_id):
                            with tf.variable_scope("self_attention"):
                                q = self.dense_layer_3d(hidden_output, self.input_arg.num_attention_heads, attention_head_size, None, "query")  # [B, F, N, D]
                                k = self.dense_layer_3d(hidden_output, self.input_arg.num_attention_heads, attention_head_size, None, "key")  # [B, F, N, D]
                                v = self.dense_layer_3d(hidden_output, self.input_arg.num_attention_heads, attention_head_size, None, "value")  # [B, F, N, D]
                                q = tf.transpose(q, [0, 2, 1, 3])  # [B, N, F, D]
                                k = tf.transpose(k, [0, 2, 1, 3])  # [B, N, F, D]
                                v = tf.transpose(v, [0, 2, 1, 3])  # [B, N, F, D]
                                attention_mask = tf.reshape(self.ph_input_mask, [batch_size, 1, seq_length, 1])  # [B, 1, F, 1]
                                logits = tf.matmul(q, k, transpose_b=True)  # q*k => [B, N, F, F]
                                logits = tf.multiply(logits, 1.0 / math.sqrt(float(self.get_shape_list(q)[-1])))  # q*k/sqrt(Dk)  => [B, N, F, F]
                                from_shape = self.get_shape_list(q)  # [B, N, F, D]
                                broadcast_ones = tf.ones([from_shape[0], 1, from_shape[2], 1], tf.float32)  # [B, 1, F, 1]
                                attention_mask = tf.matmul(broadcast_ones, tf.cast(attention_mask, tf.float32), transpose_b=True)  # [B, 1, F, 1] * [B, 1, F, 1] => [B, 1, F, F]
                                adder = (1.0 - attention_mask) * -10000.0  # [B, 1, F, F]
                                logits += adder  # [B, N, F, F]
                                attention_probs = tf.nn.softmax(logits, name="attention_probs")  # softmax(q*k/sqrt(Dk)), [B, N, F, F]
                                attention_output = tf.matmul(attention_probs, v)  # softmax(q*k/sqrt(Dk))*v , [B, N, F, F] * [B, N, F, D] => [B, N, F, D]
                                attention_output = tf.transpose(attention_output, [0, 2, 1, 3])  #[B, F, N, D]
                                attention_output = self.dense_layer_3d_proj(attention_output, self.input_arg.hidden_size, attention_head_size, None, name="dense")  # [B, F, H]
                                attention_output = tf.contrib.layers.layer_norm(inputs=attention_output + hidden_output, begin_norm_axis=-1, begin_params_axis=-1)  # [B, F, H]

                            with tf.variable_scope("ffn"):
                                intermediate_output = self.dense_layer_2d(attention_output, self.input_arg.intermediate_size, tf.nn.relu, name="dense")  # [B, F, intermediate_size]
                                hidden_output = self.dense_layer_2d(intermediate_output, self.input_arg.hidden_size, None, name="output_dense")  # [B, F, hidden_size]
                                hidden_output = tf.contrib.layers.layer_norm(inputs=hidden_output + attention_output, begin_norm_axis=-1, begin_params_axis=-1)  # [B, F, H]
                                layer_output = self.dense_layer_2d(hidden_output, embedding_size, None, name="layer_output_dense")  # [B, F, embedding_size]
                            self.all_layer_outputs.append(layer_output)
                self.output = self.all_layer_outputs[-1]  # [B, F, embedding_size]
                # max pool output
                mask = tf.expand_dims(tf.cast(tf.sequence_mask(self.ph_length, maxlen=seq_length), tf.float32), axis=2)  # [B, F, 1]
                mask = (1 - mask) * -1e5
                self.max_pool_output = tf.reduce_max(self.output + mask, axis=1, keepdims=False)  # [B, embedding_size]

    #make Matrix from 4D to 3D
    def dense_layer_3d_proj(self, input_tensor, hidden_size, head_size, activation, name=None):
        input_shape = self.get_shape_list(input_tensor)  # [B,F,N,D]
        num_attention_heads = input_shape[2]
        with tf.variable_scope(name):
            w = tf.get_variable(name="kernel", shape=[num_attention_heads * head_size, hidden_size], initializer=tf.truncated_normal_initializer(stddev=self.input_arg.initializer_range))
            w = tf.reshape(w, [num_attention_heads, head_size, hidden_size])
            b = tf.get_variable(name="bias", shape=[hidden_size], initializer=tf.zeros_initializer)
            output = tf.einsum("BFND,NDH->BFH", input_tensor, w)  # [B, F, H]
            output += b
        if activation is not None:
            return activation(output)
        else:
            return output

    #make Matrix for 3D transformation in the last index
    def dense_layer_2d(self, input_tensor, output_size, activation, name=None):
        input_shape = self.get_shape_list(input_tensor)  # [B, F, H]
        hidden_size = input_shape[2]
        with tf.variable_scope(name):
            w = tf.get_variable(name="kernel", shape=[hidden_size, output_size], initializer=tf.truncated_normal_initializer(stddev=self.input_arg.initializer_range))
            b = tf.get_variable(name="bias", shape=[output_size], initializer=tf.zeros_initializer)
            output = tf.einsum("BFH,HO->BFO", input_tensor, w)  # [B, F, O]
            output += b
        if activation is not None:
            return activation(output)
        else:
            return output

    # make Matrix from 3D to 4D
    def dense_layer_3d(self, input_tensor, num_attention_heads, head_size, activation, name=None):
        input_shape = self.get_shape_list(input_tensor)  # [B, F, H]
        hidden_size = input_shape[2]
        with tf.variable_scope(name):
            w = tf.get_variable(name="kernel", shape=[hidden_size, num_attention_heads * head_size], initializer=tf.truncated_normal_initializer(stddev=self.input_arg.initializer_range))
            w = tf.reshape(w, [hidden_size, num_attention_heads, head_size])
            b = tf.get_variable(name="bias", shape=[num_attention_heads * head_size], initializer=tf.zeros_initializer)
            b = tf.reshape(b, [num_attention_heads, head_size])
            output = tf.einsum("BFH,HND->BFND", input_tensor, w)  #[B, F, N, D]
            output += b
        if activation is not None:
            return activation(output)
        else:
            return output

    def get_shape_list(self, tensor):
        """Returns a list of the shape of tensor, preferring static dimensions.
        """
        tensor_shape = tensor.shape.as_list()
        none_indexes = []
        for (index, dim) in enumerate(tensor_shape):
            if dim is None:
                none_indexes.append(index)
        if not none_indexes:
            return tensor_shape
        dynamic_shape = tf.shape(tensor)
        for index in none_indexes:
            tensor_shape[index] = dynamic_shape[index]
        return tensor_shape



def create_bidirectional_lm_training_op(input_arg, other_arg_dict):
    loss_op, model = create_bidirectional_lm_model(input_arg, other_arg_dict)
    train_op, learning_rate_op = create_optimizer(loss_op, input_arg.learning_rate, input_arg.train_step,
                                                  input_arg.warmup_step, input_arg.clip_norm, input_arg.weight_decay)
    model.loss_op = loss_op
    model.train_op = train_op
    model.learning_rate_op = learning_rate_op
    return model


def create_bidirectional_lm_model(input_arg, other_arg_dict):
    model = BiDirectionalLmModel(input_arg, other_arg_dict)
    model.build()
    max_predictions_per_seq = input_arg.max_predictions_per_seq

    model.global_step = tf.train.get_or_create_global_step()
    model.ph_labels = tf.placeholder(dtype=tf.int32, shape=[None, max_predictions_per_seq],
                                     name="ph_labels")  # [batch, max_predictions_per_seq]
    model.ph_positions = tf.placeholder(dtype=tf.int32, shape=[None, max_predictions_per_seq],
                                        name="ph_positions")  # [batch, max_predictions_per_seq]
    model.ph_weights = tf.placeholder(dtype=tf.float32, shape=[None, max_predictions_per_seq],
                                      name="ph_weights")  # [batch, max_predictions_per_seq]

    real_output = gather_indexes(model.output, model.ph_positions)  # [batch * max_predictions_per_seq, embedding_dim]

    bias = tf.get_variable("bias", shape=[model.token_num], initializer=tf.zeros_initializer())
    logits = tf.matmul(real_output, model.v_token_embedding, transpose_b=True)  #[batch * max_predictions_per_seq, token_num]
    logits = tf.nn.bias_add(logits, bias)

    log_probs = tf.nn.log_softmax(logits, axis=-1)  #[batch * max_predictions_per_seq, token_num]
    one_hot_labels = tf.one_hot(tf.reshape(model.ph_labels, [-1]), depth=model.token_num, dtype=tf.float32)  #[batch * max_predictions_per_seq, token_num]

    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])  # [batch * max_predictions_per_seq]
    weights = tf.reshape(model.ph_weights, [-1])  # [batch * max_predictions_per_seq]
    loss = (tf.reduce_sum(weights * per_example_loss)) / (tf.reduce_sum(weights) + 1e-5)

    return loss, model


def create_optimizer(loss, init_lr=5e-5, num_train_steps=1000000, num_warmup_steps=20000, clip_nom=1.0,
                     weight_decay=0.01):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(learning_rate, global_step, num_train_steps, end_learning_rate=0.0,
                                              power=1.0, cycle=False)  # linear warmup

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
    #seq_output:[batch, length, 2 * dim]
    #positions:[batch, max_predictions_per_seq]

    batch_size = tf.shape(seq_output)[0]
    length = tf.shape(seq_output)[1]
    dim = tf.shape(seq_output)[2]

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * length, [-1, 1])  #[batch_size, 1]
    output_tensor = tf.gather(tf.reshape(seq_output, [-1, dim]), tf.reshape(positions + flat_offsets, [-1]))
    return output_tensor  # [batch * max_predictions_per_seq, dim]


# finetune model utils
def create_finetune_classification_training_op(input_arg, other_arg_dict):
    model = create_finetune_classification_model(input_arg, other_arg_dict)
    repre = model.max_pool_output  # [batch, 2 * dim]

    model.ph_labels = tf.placeholder(dtype=tf.int32, shape=[None], name="ph_labels")  # [batch]
    logits = tf.layers.dense(repre, other_arg_dict["label_num"],
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="logits")
    model.softmax_op = tf.nn.softmax(logits, -1, name="softmax_pre")
    model.loss_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=model.ph_labels), -1)
    model.global_step_op = tf.train.get_or_create_global_step()

    tf.logging.info("learning_rate : {}".format(input_arg.learning_rate))
    if input_arg.opt_type == "sgd":
        tf.logging.info("use sgd")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=input_arg.learning_rate)
    elif input_arg.opt_type == "adagrad":
        tf.logging.info("use adagrad")
        optimizer = tf.train.AdagradOptimizer(learning_rate=input_arg.learning_rate)
    elif input_arg.opt_type == "adam":
        tf.logging.info("use adam")
        optimizer = tf.train.AdamOptimizer(learning_rate=input_arg.learning_rate)
    else:
        assert False

    list_g_v_pair = optimizer.compute_gradients(model.loss_op)
    model.train_op = optimizer.apply_gradients(list_g_v_pair, global_step=model.global_step_op)

    return model

#create finetune model for classification
def create_finetune_classification_model(input_arg, other_arg_dict):
    model = BiDirectionalLmModel(input_arg, other_arg_dict)

    model.build()

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if input_arg.init_checkpoint:
        tf.logging.info("init from checkpoint!")
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars,
                                                                                        input_arg.init_checkpoint)
        tf.train.init_from_checkpoint(input_arg.init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("name = {}, shape = {}{}".format(var.name, var.shape, init_string))

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
