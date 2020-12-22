# -*- coding: utf-8 -*-
"""
dataset class
"""
import random
import math
import numpy as np
import data_utils


class Dataset():
    def __init__(self, train_x=None, train_y=None, test_x=None,
                 test_y=None, train_length_x=None, test_length_x=None):
        self.train_x = train_x
        self.train_length_x = train_length_x
        self.test_length_x = test_length_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def gen_next_batch(self, batch_size, is_train_set, epoch=None, iteration=None):
        if is_train_set is True:
            raw_x = self.train_x
            x_length = self.train_length_x
            raw_y = self.train_y
        else:
            raw_x = self.test_x
            x_length = self.test_length_x
            raw_y = self.test_y

        assert len(raw_x) >= batch_size,\
            "batch size must be smaller than data size {}.".format(len(raw_x))

        if epoch is not None:
            until = math.ceil(float(epoch * len(raw_x)) / float(batch_size))
        elif iteration is not None:
            until = iteration
        else:
            assert False, "epoch or iteration must be set."

        iter_ = 0
        index_list = list(range(len(raw_x)))
        while iter_ <= until:
            idxs = random.sample(index_list, batch_size)
            iter_ += 1
            yield (raw_x[idxs], raw_y[idxs], idxs, x_length[idxs])


class ExpDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        train_file = args.train_file
        vocab_file = args.vocab_file

        train_sens = data_utils.load_sentences(train_file, skip_invalid=True)
        word2id, id2word, label2id, id2label = data_utils.load_vocab(train_sens, vocab_file)

        data_utils.gen_ids(train_sens, word2id, label2id, 100)
        train_full_tensors = data_utils.make_full_tensors(train_sens)

        raw_x = train_full_tensors[0]
        x_length = train_full_tensors[1]
        x_labels = train_full_tensors[2]

        raw_f = lambda t: id2label[t]
        x_labels_true = np.array(list(map(raw_f, x_labels)))

        n_train = int(len(raw_x) * 1)
        self.train_x, self.test_x = raw_x[:n_train], raw_x[n_train:]
        self.train_length_x, self.test_length_x = x_length[:n_train], x_length[n_train:]
        self.train_y, self.test_y = x_labels[:n_train], x_labels[n_train:]
        self.gt_label = x_labels_true
        self.raw_q = ["".join(i.raw_tokens) for i in train_sens]
