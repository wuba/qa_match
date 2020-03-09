# coding=utf-8

"""
tools for run bi-lstm short text classification
"""

import numpy as np

class TextLoader(object):
    def __init__(self, is_training, data_path, batch_size, seq_length, vocab, labels, encoding='utf8', is_reverse=False):
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        self.is_train = is_training
        if is_training:
            label_set = set()
            word_set = set()
            train_file = data_path
            with open(train_file, 'r') as fin:
                for line in fin:
                    tokens = line.strip().split()
                    if len(tokens) <= 1:
                        continue
                    label_set.add(tokens[0])
                    for token in tokens[1:]:
                        word_set.add(token)
            self.labels = dict(zip(list(label_set), range(0, len(label_set))))   #{__label__1:0, __label_2:1, __label_3:2, ...}
            self.id_2_label = {value:key for key, value in self.labels.items()}
            self.label_size = len(self.labels)
            self.vocab = dict(zip(list(word_set), range(1, len(word_set)+1)))
            self.id_2_vocab = {value:key for key, value in self.vocab.items()}
            self.vocab_size = len(self.vocab) + 1  #self.vocab.size + 1, 0 for pad
            self.load_preprocessed(data_path, is_reverse)
        elif vocab is not None and labels is not None:
            self.vocab = vocab
            self.id_2_vocab = {value: key for key, value in self.vocab.items()}
            self.vocab_size = len(vocab) + 1
            self.labels = labels
            self.id_2_label = {value: key for key, value in self.labels.items()}
            self.label_size = len(self.labels)
            self.load_preprocessed(data_path, is_reverse)
        self.num_batches = 1
        self.x_batches = None
        self.y_batches = None
        self.len_batches = None
        self.reset_batch_pointer()

    def load_preprocessed(self, data_path, is_reverse):
        train_file = data_path
        with open(train_file, 'r') as fin:
            train_x = []
            train_y = []
            train_len = []
            for line in fin:
                temp_x = []
                temp_y = []
                x_len = []
                tokens = line.strip().split()
                if len(tokens) <= 1:
                    continue
                if tokens[0] not in self.labels: #for predict
                    self.labels[tokens[0]] = len(self.labels)
                temp_y.append(self.labels[tokens[0]])
                for item in tokens[1:]:
                    if item in self.vocab:
                        temp_x.append(self.vocab[item])
                if len(temp_x) >= self.seq_length:  #40
                    x_len.append(self.seq_length)
                    temp_x = temp_x[:self.seq_length]
                    if is_reverse:
                        temp_x.reverse()
                else:
                    x_len.append(len(temp_x))
                    if is_reverse:
                        temp_x.reverse()
                    temp_x = temp_x + [0] * (self.seq_length - len(temp_x))
                train_x.append(temp_x)
                train_y.append(temp_y)
                train_len.append(x_len)
        tensor_x = np.array(train_x)
        tensor_y = np.array(train_y)
        tensor_len = np.array(train_len)
        self.tensor = np.c_[tensor_x, tensor_y, tensor_len].astype(int)   #tensor_x.size * (40+1+1)

    def create_batches(self):
        self.num_batches = int(self.tensor.shape[0] / self.batch_size)
        if int(self.tensor.shape[0] % self.batch_size):
            self.num_batches = self.num_batches + 1
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'
        if self.is_train:
            np.random.shuffle(self.tensor)
            tensor = self.tensor[:self.num_batches * self.batch_size]
        else:
            tensor = self.tensor[:self.num_batches * self.batch_size]
        self.x_batches = np.array_split(tensor[:, :-2], self.num_batches, 0)
        self.y_batches = np.array_split(tensor[:, -2], self.num_batches, 0)
        self.len_batches = np.array_split(tensor[:, -1], self.num_batches, 0)

    def next_batch(self):
        batch_x = self.x_batches[self.pointer]
        batch_y = self.y_batches[self.pointer]
        xlen = self.len_batches[self.pointer]
        self.pointer += 1
        return batch_x, batch_y, xlen

    def reset_batch_pointer(self):
        self.create_batches()
        self.pointer = 0

if __name__ == "__main__":
    pass
