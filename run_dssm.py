# coding=utf-8

"""
running lstm + dssm for short text matching

"""

import numpy as np
import tensorflow as tf
import os
from match_utils import DataHelper
from dssm import Dssm

flags = tf.app.flags
FLAGS = flags.FLAGS

# data parameters
flags.DEFINE_string('train_path', None, 'dir for train data')
flags.DEFINE_string('valid_path', None, 'dir for valid data')
flags.DEFINE_string('map_file_path', None, 'dir for label std question mapping')
flags.DEFINE_string('model_path', None, 'Model path')
flags.DEFINE_string('label2id_path', None, 'label2id file path')
flags.DEFINE_string('vocab2id_path', None, 'vocab2id file path')

# training parameters
flags.DEFINE_integer('softmax_r', 45, 'Smooth parameter for osine similarity')
flags.DEFINE_integer('embedding_size', 200, 'max_sequence_len')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

flags.DEFINE_float('keep_prob', 0.8, 'Dropout keep prob.')
flags.DEFINE_integer('num_epoches', 10, "Number of epochs.")
flags.DEFINE_integer('batch_size', 50, "Size of one batch.")
flags.DEFINE_integer('negative_size', 5, "Size of negtive sample.")
flags.DEFINE_integer('eval_every', 50, "Record summaries every n steps.")
flags.DEFINE_integer('num_units', 100, "Number of units of lstm(default: 100)")
flags.DEFINE_bool('use_same_cell', True, "whether to use sam cell")

def feed_dict_builder(batch, keep_prob, dssm):
    #batch: ([(q1_len, [q1_w1, q1_w2,...]), (q2_len, [q2_w1, q2_w2,...]), ...],    [(std1_len, [std1_w1, std1_w2,...]), (std2_len, [std2_w1, std2_w2,...]), ...])
    length_x = [x[0] for x in batch[0]]
    input_x = [x[1] for x in batch[0]]
    length_y = [y[0] for y in batch[1]]
    input_y = [y[1] for y in batch[1]]
    feed_dict = {
        dssm.input_x: np.array(input_x, dtype=np.int32),
        dssm.length_x: np.array(length_x, dtype=np.int32),
        dssm.input_y: np.array(input_y, dtype=np.int32),
        dssm.length_y: np.array(length_y, dtype=np.int32),
        dssm.keep_prob: keep_prob
    }
    return feed_dict

def cal_predict_acc_num(predict_prob, test_batch_q, predict_label_seq):
    #calculate acc
    assert (len(test_batch_q) == len(predict_prob))
    real_labels = []
    for ques in test_batch_q:
        label = ques[2]
        real_labels.append(label)
    acc_num = 0
    sorted_scores = []
    for i, scores in enumerate(predict_prob):
        label_scores = {}
        for index, score in enumerate(scores):
            label_scores[predict_label_seq[index]] = score
        #sort
        label_scores = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_scores.append(label_scores)
        top_label = label_scores[0][0]
        if top_label == real_labels[i]:
            acc_num = acc_num + 1
    return acc_num, real_labels, sorted_scores

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    data_help = DataHelper(FLAGS.train_path, FLAGS.valid_path, None, FLAGS.map_file_path, FLAGS.batch_size, FLAGS.num_epoches, None, None, True)
    dssm = Dssm(FLAGS.num_units, FLAGS.batch_size, FLAGS.negative_size, FLAGS.softmax_r, FLAGS.learning_rate,
                data_help.vocab_size, FLAGS.embedding_size, use_same_cell=False)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=1)
    train_batches = data_help.train_batch_iterator(data_help.train_id_ques, data_help.std_id_ques)
    best_valid_acc = 0
    # run_num = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for train_batch_step, train_batch in enumerate(train_batches):
            _, step, train_lr, train_loss = sess.run([dssm.train_step, dssm.global_step, dssm.learning_rate, dssm.loss], feed_dict=feed_dict_builder(train_batch, FLAGS.keep_prob, dssm))
            tf.logging.info("Training...... global_step {}, epcho {}, train_batch_step {}, learning rate {} "
                            "loss {}".format(step, round(step * FLAGS.batch_size / data_help.train_num, 2), train_batch_step, round(train_lr, 4), train_loss))
            if (train_batch_step + 1) % FLAGS.eval_every == 0:
                # run_num = run_num + 1
                # if run_num % 2 == 0:
                #     break
                all_valid_acc_num = 0
                all_valid_num = 0
                valid_batches = data_help.valid_batch_iterator()
                for _, valid_batch_q in enumerate(valid_batches):
                    all_valid_num = all_valid_num + len(valid_batch_q)
                    valid_batch = (valid_batch_q, data_help.std_batch)
                    valid_prob = sess.run([dssm.prob_pre], feed_dict=feed_dict_builder(valid_batch, 1.0, dssm))
                    valid_acc_num, real_labels, _ = cal_predict_acc_num(valid_prob[0], valid_batch_q, data_help.id2label)
                    all_valid_acc_num = all_valid_acc_num + valid_acc_num
                current_acc = all_valid_acc_num * 1.0 / all_valid_num
                tf.logging.info("validing...... global_step {}, valid_acc {}, current_best_acc {}".format(step, current_acc, best_valid_acc))
                if current_acc > best_valid_acc:
                    tf.logging.info("validing...... get the best acc {} and saving model and result".format(current_acc))
                    saver.save(sess, FLAGS.model_path + "dssm_{}".format(train_batch_step))
                    best_valid_acc = current_acc
                    #save label2id, vocab2id
                    vocabfile = open(FLAGS.vocab2id_path, 'w', encoding='utf-8')
                    for key, value in data_help.vocab2id.items():
                        vocabfile.write(str(key) + "\t" + str(value) + '\n')
                    vocabfile.close()
                    labelfile = open(FLAGS.label2id_path, 'w', encoding='utf-8')
                    for key, value in data_help.label2id.items():
                        labelfile.write(str(key) + "\t" + str(value) + '\n')
                    labelfile.close()
                    # break


if __name__ == "__main__":
    tf.app.run()
