# -*- coding: utf-8 -*-
"""
data utils
"""

import os
import sys
import codecs
import collections
import tensorflow as tf
import numpy as np
import six



def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

class Sentence:
    def __init__(self, raw_tokens, raw_label):
        self.raw_tokens = raw_tokens
        self.raw_label = raw_label
        self.label_id = None
        self.token_ids = []

    def to_ids(self, word2id, label2id, max_len):
        self.label_id = label2id[self.raw_label]
        self.raw_tokens = self.raw_tokens[:max_len]  # cut off to the max length
        all_unk = True
        for raw_token in self.raw_tokens:
            if raw_token not in  ['<UNK>', '<CLS>', '<SEP>', '<MASK>', '<S>', '<T>', '<PAD>']:
                raw_token = raw_token.lower()
            if raw_token in word2id:
                self.token_ids.append(word2id[raw_token])
                all_unk = False
            else:
                self.token_ids.append(word2id["<UNK>"])
        if all_unk:
            tf.logging.info("all unk" + self.raw_tokens)

        self.token_ids = self.token_ids + [0] * (max_len - len(self.token_ids))

def gen_ids(sens, word2id, label2id, max_len):
    for sen in sens:
        sen.to_ids(word2id, label2id, max_len)

# convert dataset to tensor
def make_full_tensors(sens):
    tokens = np.zeros((len(sens), len(sens[0].token_ids)), dtype=np.int32)
    labels = np.zeros((len(sens)), dtype=np.int32)
    length = np.zeros((len(sens)), dtype=np.int32)
    for idx, sen in enumerate(sens):
        tokens[idx] = sen.token_ids
        labels[idx] = sen.label_id
        length[idx] = len(sen.raw_tokens)
    return tokens, length, labels

def gen_batchs(full_tensors, batch_size, is_shuffle):
    tokens, labels, length = full_tensors
    # per = np.array([i for i in range(len(tokens))])
    per = np.array(list(range(len(tokens))))
    if is_shuffle:
        np.random.shuffle(per)

    cur_idx = 0
    token_batch = []
    label_batch = []
    length_batch = []
    while cur_idx < len(tokens):
        token_batch.append(tokens[per[cur_idx]])
        label_batch.append(labels[per[cur_idx]])
        length_batch.append(length[per[cur_idx]])
        if len(token_batch) == batch_size or cur_idx == len(tokens) - 1:
            yield token_batch, label_batch, length_batch
            token_batch = []
            label_batch = []
            length_batch = []
        cur_idx += 1

def load_sentences(file_path, skip_invalid):
    sens = []
    invalid_num = 0
    max_len = 0
    for raw_l in codecs.open(file_path, 'r', 'utf-8'):  # load as utf-8 encoding.
        if raw_l.strip() == "":
            continue
        file_s = raw_l.rstrip().split('\t')
        assert len(file_s) == 2
        tokens = file_s[1].split()  # discard empty strings
        for token in tokens:
            assert token != ""
        label = file_s[0]
        if skip_invalid:
            if label.find(',') >= 0 or label.find('NONE') >= 0:
                invalid_num += 1
                continue
        if len(tokens) > max_len:
            max_len = len(tokens)
        sens.append(Sentence(tokens, label))
    tf.logging.info("invalid sen num : " + str(invalid_num))
    tf.logging.info("valid sen num : " + str(len(sens)))
    tf.logging.info("max_len : " + str(max_len))
    return sens

def load_vocab(sens, vocab_file):
    label2id = {}
    id2label = {}
    for sen in sens:
        if sen.raw_label not in label2id:
            label2id[sen.raw_label] = len(label2id)
            id2label[len(id2label)] = sen.raw_label

    index = 0
    word2id = collections.OrderedDict()
    id2word = collections.OrderedDict()
    for l_raw in codecs.open(vocab_file, 'r', 'utf-8'):
        token = convert_to_unicode(l_raw)
        # if not token:
        #     break
        token = token.strip()
        word2id[token] = index
        # id2word[index] = token
        index += 1

    for k, value in word2id.items():
        id2word[value] = k

    assert len(word2id) == len(id2word)
    tf.logging.info("token num : " + str(len(word2id)))
    tf.logging.info("label num : " + str(len(label2id)))
    tf.logging.info("labels: " + str(id2label))
    return word2id, id2word, label2id, id2label

def evaluate(sess, full_tensors, args, model):
    total_num = 0
    right_num = 0
    for batch_data in gen_batchs(full_tensors, args.batch_size, is_shuffle=False):
        softmax_re = sess.run(model.softmax_op,
                                        feed_dict={model.ph_dropout_rate: 0,
                                                   model.ph_tokens: batch_data[0],
                                                   model.ph_labels: batch_data[1],
                                                   model.ph_length: batch_data[2]})
        pred_re = np.argmax(softmax_re, axis=1)
        total_num += len(pred_re)
        right_num += np.sum(pred_re == batch_data[1])
        acc = 1.0 * right_num / (total_num + 1e-5)

    tf.logging.info("dev total num: " + str(total_num) +  ", right num: " +  str(right_num) + ", acc: " + str(acc))
    return acc

def load_spec_centers(path):
    raw_f = open(path, "r", encoding="utf-8")
    f_lines = raw_f.readlines()

    res = []
    for line in f_lines:
        vec = [float(i) for i in line.strip().split(" ")]
        res.append(vec)
    return tf.convert_to_tensor(res), len(res)

def write_file(out_path, out_str):
    exists = os.path.isfile(out_path)
    if exists:
        os.remove(out_path)
        tf.logging.info("File Removed!")

    raw_f = open(out_path, "w", encoding="utf-8")
    raw_f.write(out_str)
    raw_f.close()

def load_vocab_file(vocab_file):
    word2id = {}
    id2word = {}
    for raw_l in codecs.open(vocab_file, 'r', 'utf8'):
        raw_l = raw_l.strip()
        assert raw_l != ""
        assert raw_l not in word2id
        word2id[raw_l] = len(word2id)
        id2word[len(id2word)] = raw_l
    tf.logging.info("uniq token num : " + str(len(word2id)) + "\n")
    return word2id, id2word
