# -*- coding: utf-8 -*-
"""
tools for runing pretrain and finetune language models

"""
import sys
import os
import numpy as np
import argparse
import codecs
from collections import namedtuple
import queue
import threading
import random
import re
import tensorflow as tf


# sample representation
class Sentence(object):
    def __init__(self, raw_tokens, raw_label=None):
        self.raw_tokens = raw_tokens
        self.raw_label = raw_label
        self.label_id = None
        self.token_ids = []

    # for pretrain
    def to_id(self, word2id, args):
        # for each epoch, this should be rerun to get random results for each sentence.
        self.positions = []
        self.labels = []
        self.weights = []
        self.fw_labels = []
        self.bw_labels = []

        for t in self.raw_tokens:
            self.token_ids.append(word2id[t])

        for ta in self.bidirectional_targets:
            # predict itself
            self.labels.append(self.token_ids[ta.position])
            # in-place modify the target token in the sentence
            self.token_ids[ta.position] = word2id[ta.replace_token]
            self.positions.append(ta.position)
            self.weights.append(1.0)

        # fix to tensors for the predictions of LM
        cur_len = len(self.labels)
        self.labels = self.labels + [0] * (args.max_predictions_per_seq - cur_len)
        self.positions = self.positions + [0] * (args.max_predictions_per_seq - cur_len)
        self.weights = self.weights + [0] * (args.max_predictions_per_seq - cur_len)

    # for finetune
    def to_ids(self, word2id, label2id, max_len):
        self.label_id = label2id[self.raw_label]
        self.raw_tokens = self.raw_tokens[:max_len]  # cut off to the max length
        all_unk = True
        for t in self.raw_tokens:
            if t in word2id:
                self.token_ids.append(word2id[t])
                all_unk = False
            else:
                self.token_ids.append(word2id["<UNK>"])
        assert not all_unk

        self.token_ids = self.token_ids + [0] * (max_len - len(self.token_ids))

    # file utils


def load_vocab_file(vocab_file):
    word2id = {}
    id2word = {}
    for l in codecs.open(vocab_file, 'r', 'utf-8'):
        l = l.strip()
        assert l != ""
        assert l not in word2id
        word2id[l] = len(word2id)
        id2word[len(id2word)] = l
    sys.stderr.write("uniq token num : " + str(len(word2id)) + "\n")
    return word2id, id2word


def load_vocab(sens, vocab_file):
    label2id = {}
    id2label = {}
    for sen in sens:
        if sen.raw_label not in label2id:
            label2id[sen.raw_label] = len(label2id)
            id2label[len(id2label)] = sen.raw_label

    word2id, id2word = load_vocab_file(vocab_file)
    assert len(word2id) == len(id2word)
    tf.logging.info("\ntoken num : " + str(len(word2id)))
    tf.logging.info(", label num : " + str(len(label2id)))
    tf.logging.info(", labels: " + str(id2label))
    return word2id, id2word, label2id, id2label


def load_id2label_file(id2label_file):
    di = {}
    for l in open(id2label_file, 'r'):
        fs = l.rstrip().split('\t')
        assert len(fs) == 2
        di[int(fs[0])] = fs[1]
    return di


def gen_test_data(test_file, word2id):
    """
    read and encode test file. 
    """
    sens = []
    for l in codecs.open(test_file, 'r', 'utf-8'):
        fs = l.rstrip().split('\t')[-1].split()
        sen = []
        for f in fs:
            if f in word2id:
                sen.append(word2id[f])
            else:
                sen.append(word2id['<UNK>'])
        sens.append(sen)
    return sens


def load_pretraining_data(train_file, max_seq_len):
    sens = []
    for l in codecs.open(train_file, 'r', 'utf-8'):
        sen = Sentence(l.rstrip().split("\t")[-1].split()[:max_seq_len])
        if len(sen.raw_tokens) == 0:
            continue
        sens.append(sen)
        if len(sens) % 2000000 == 0:
            tf.logging.info("load sens :" + str(len(sens)))
    tf.logging.info("training sens num :" + str(len(sens)))
    return sens


def load_training_data(file_path, skip_invalid=True):
    sens = []
    invalid_num = 0
    max_len = 0
    for l in codecs.open(file_path, 'r', 'utf-8'):  # load as utf-8 encoding.
        if l.strip() == "":
            continue
        fs = l.rstrip().split('\t')
        assert len(fs) == 3
        tokens = fs[2].split()  # discard empty strings
        for t in tokens:
            assert t != ""
        label = "__label__{}".format(fs[0])
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


# pretrain utils
BiReplacement = namedtuple("BiReplacement", ["position", "replace_token"])


def gen_pretrain_targets(raw_tokens, id2word, max_predictions_per_seq):
    assert max_predictions_per_seq > 0
    assert len(raw_tokens) > 0
    pred_num = min(max_predictions_per_seq, max(1, int(round(len(raw_tokens) * 0.15))))

    re = []
    covered_pos_set = set()
    for _ in range(pred_num):
        cur_pos = np.random.randint(0, len(raw_tokens))
        if cur_pos in covered_pos_set:
            continue
        covered_pos_set.add(cur_pos)

        prob = np.random.uniform()
        if prob < 0.8:
            replace_token = '<MASK>'
        elif prob < 0.9:
            replace_token = raw_tokens[cur_pos]  # itself
        else:
            while True:
                fake_pos = np.random.randint(0, len(id2word))  # random one
                replace_token = id2word[fake_pos]
                if raw_tokens[cur_pos] != replace_token:
                    break
        re.append(BiReplacement(position=cur_pos, replace_token=replace_token))
    return re


def gen_ids(sens, word2id, label2id, max_len):
    for sen in sens:
        sen.to_ids(word2id, label2id, max_len)


def to_ids(sens, word2id, args, id2word):
    num = 0
    for sen in sens:
        if num % 2000000 == 0:
            tf.logging.info("to_ids handling num : " + str(num))
        num += 1
        sen.bidirectional_targets = gen_pretrain_targets(sen.raw_tokens, id2word, args.max_predictions_per_seq)
        sen.to_id(word2id, args)


def gen_batches(sens, batch_size):
    per = np.array([i for i in range(len(sens))])
    np.random.shuffle(per)

    cur_idx = 0
    token_batch = []
    length_batch = []

    position_batch = []
    label_batch = []
    weight_batch = []

    while cur_idx < len(sens):
        token_batch.append(sens[per[cur_idx]].token_ids)
        length_batch.append(len(sens[per[cur_idx]].token_ids))

        label_batch.append(sens[per[cur_idx]].labels)
        position_batch.append(sens[per[cur_idx]].positions)
        weight_batch.append(sens[per[cur_idx]].weights)
        if len(token_batch) == batch_size or cur_idx == len(sens) - 1:
            max_len = max(length_batch)
            for ts in token_batch: ts.extend([0] * (max(length_batch) - len(ts)))

            yield token_batch, length_batch, label_batch, position_batch, weight_batch

            del token_batch
            del length_batch
            del label_batch
            del position_batch
            del weight_batch
            token_batch = []
            length_batch = []
            label_batch = []
            position_batch = []
            weight_batch = []
        cur_idx += 1


def queue_gen_batches(sens, args, word2id, id2word):
    def enqueue(sens, q):
        permu = np.arange(len(sens))
        np.random.shuffle(permu)
        idx = 0
        tf.logging.info("thread started!")
        while True:
            sen = sens[permu[idx]]
            sen.bidirectional_targets = gen_pretrain_targets(sen.raw_tokens, id2word,
                                                                args.max_predictions_per_seq)
            sen.to_id(word2id, args)
            q.put(sen)
            idx += 1
            if idx >= len(sens):
                np.random.shuffle(permu)
                idx = idx % len(sens)

    q = queue.Queue(maxsize=50000)

    for i in range(args.enqueue_thread_num):
        tf.logging.info("enqueue thread started : " + str(i))
        enqeue_thread = threading.Thread(target=enqueue, args=(sens, q))
        enqeue_thread.setDaemon(True)
        enqeue_thread.start()

    qu_sens = []
    while True:
        cur_sen = q.get()
        qu_sens.append(cur_sen)
        if len(qu_sens) >= args.batch_size:
            for data in gen_batches(qu_sens, args.batch_size):
                yield data
            qu_sens = []


def make_full_tensors(sens):
    tokens = np.zeros((len(sens), len(sens[0].token_ids)), dtype=np.int32)
    labels = np.zeros((len(sens)), dtype=np.int32)
    length = np.zeros((len(sens)), dtype=np.int32)
    for idx, sen in enumerate(sens):
        tokens[idx] = sen.token_ids
        labels[idx] = sen.label_id
        length[idx] = len(sen.raw_tokens)
    return tokens, labels, length


def gen_batchs(full_tensors, batch_size, is_shuffle):
    tokens, labels, length = full_tensors
    per = np.array([i for i in range(len(tokens))])
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
            # make the tokens to real max length
            real_max_len = max(length_batch)
            for i in range(len(token_batch)):
                token_batch[i] = token_batch[i][:real_max_len]

            yield token_batch, label_batch, length_batch
            token_batch = []
            label_batch = []
            length_batch = []
        cur_idx += 1


if __name__ == "__main__":
    pass
