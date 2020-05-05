# coding=utf-8

"""
tools for run bi-lstm short text classification
"""

import numpy as np
import math

class TextLoader(object):
    def __init__(self, is_training, data_path, map_file_path, batch_size, seq_length, vocab, labels, std_label_map, encoding='utf8', is_reverse=False):
        self.data_path = data_path
        self.map_file_path = map_file_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_train = is_training
        self.encoding = encoding
        #load label std mapping index
        self.std_label_map = {}
        self.label_num_map = {}
        label_set = set()
        word_set = set()
        if is_training:
            with open(map_file_path, 'r', encoding=encoding) as map_index_file:
                for line in map_index_file:
                    tokens = line.strip().split('\t')
                    assert len(tokens) == 3
                    label = tokens[0]
                    label_set.add(label)
                    std_id = tokens[1]
                    self.std_label_map[std_id] = label
                    words = tokens[2].split(" ")
                    for token in words:
                        word_set.add(token)

            train_file = data_path
            with open(train_file, 'r', encoding=encoding) as fin:
                for line in fin:
                    tokens = line.strip().split('\t')
                    assert len(tokens) == 3
                    std_ids = tokens[0].split(",")
                    words = tokens[2].split(" ")
                    if len(std_ids) > 1:
                        label = '__label__list'  #answer list
                        label_set.add(label)
                    elif std_ids[0] == '0':
                        label = '__label__none'  #refuse answer
                        label_set.add(label)
                    else:
                        assert std_ids[0] in self.std_label_map
                        label = self.std_label_map.get(std_ids[0])  #__label__xx:some label
                    for token in words:
                        word_set.add(token)
                    if label not in self.label_num_map:
                        self.label_num_map[label] = 1
                    else:
                        self.label_num_map[label] = self.label_num_map[label] + 1

            self.labels = dict(zip(list(label_set), range(0, len(label_set))))   #{__label__1:0, __label_2:1, __label_3:2, ...}
            # print("self.labels: " + str(self.labels))
            # print("self.std_label_map: " + str(self.std_label_map))
            self.id_2_label = {value:key for key, value in self.labels.items()}
            self.label_size = len(self.labels)
            self.vocab = dict(zip(list(word_set), range(1, len(word_set)+1)))
            self.id_2_vocab = {value:key for key, value in self.vocab.items()}
            self.vocab_size = len(self.vocab) + 1  #self.vocab.size + 1, 0 for pad, not encoding unknown, if care for it, you can modify here
            self.load_preprocessed(data_path, is_reverse)
        elif vocab is not None and labels is not None and std_label_map is not None:
            self.vocab = vocab
            self.id_2_vocab = {value: key for key, value in self.vocab.items()}
            self.vocab_size = len(vocab) + 1
            self.labels = labels
            self.id_2_label = {value: key for key, value in self.labels.items()}
            self.label_size = len(self.labels)
            self.std_label_map = std_label_map
            self.load_preprocessed(data_path, is_reverse)
        self.num_batches = 1
        self.x_batches = None
        self.y_batches = None
        self.len_batches = None
        self.reset_batch_pointer()

    def load_preprocessed(self, data_path, is_reverse):
        train_file = data_path
        self.raw_lines = []
        with open(train_file, 'r', encoding=self.encoding) as fin:
            train_x = []
            train_y = []
            train_len = []
            for line in fin:
                temp_x = []
                temp_y = []
                x_len = []
                tokens = line.strip().split('\t')
                assert len(tokens) == 3
                std_ids = tokens[0].split(",")
                words = tokens[2].split(" ")
                if len(std_ids) > 1:
                    label = '__label__list'  # answer list
                elif std_ids[0] == '0':
                    label = '__label__none'  # refuse answer
                else:
                    if std_ids[0] not in self.std_label_map:
                        label = '__label__none'
                    else:
                        label = self.std_label_map.get(std_ids[0])  # __label__xx:some label
                # if label not in self.labels:
                #     print("label: <" + label + ">")
                #     print("self.labels: ")
                #     print(str(self.labels))
                temp_y.append(self.labels[label])
                for item in words:
                    if item in self.vocab:  #not encoding unknown, if care for it, you can modify here
                        temp_x.append(self.vocab[item])
                if len(temp_x) == 0:
                    print("all word in line is not in vocab, line: " + line)
                    continue
                if len(temp_x) >= self.seq_length:
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
                self.raw_lines.append(tokens[2])
        tensor_x = np.array(train_x)
        tensor_y = np.array(train_y)
        tensor_len = np.array(train_len)
        # print("tensor_x.shape: " + str(tensor_x.shape))
        # print("len(self.raw_lines): " + str(len(self.raw_lines)))

        self.tensor = np.c_[tensor_x, tensor_y, tensor_len].astype(int)   #tensor_x.size * (40+1+1)

    def list_split_n(self, raw_items, split_len_batches):
        split_items = []
        j = 0
        for i in range(len(split_len_batches)):
            split_items.append(raw_items[j: j + split_len_batches[i]])
            j += split_len_batches[i]
        return split_items


    def create_batches(self):
        self.num_batches = int(self.tensor.shape[0] / self.batch_size)
        if int(self.tensor.shape[0] % self.batch_size):
            self.num_batches = self.num_batches + 1
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'
        if self.is_train:
            np.random.shuffle(self.tensor)
        # print("self.num_batches: " + str(self.num_batches))
        tensor = self.tensor[:self.num_batches * self.batch_size]
        # print("len(tensor): " + str(len(tensor)))
        raw_lines = self.raw_lines[:self.num_batches * self.batch_size]  #if train raw_lines order is different from tensor
        # print("len(raw_lines): " + str(len(raw_lines)))

        self.x_batches = np.array_split(tensor[:, :-2], self.num_batches, 0)
        self.y_batches = np.array_split(tensor[:, -2], self.num_batches, 0)
        self.len_batches = np.array_split(tensor[:, -1], self.num_batches, 0)
        split_len_batches = []
        for i in range(len(self.x_batches)):
            split_len_batches.append(len(self.x_batches[i]))
        self.raw_lines_batches = self.list_split_n(raw_lines, split_len_batches)  #should split by np.array_split
        sum = 0
        # for i in range(len(self.x_batches)):
        #     print("i: " + str(i) + "len(self.x_batches[i]): " + str(len(self.x_batches[i])))
        #     sum += len(self.x_batches[i])
        # print("sum: " + str(sum))
        #
        # print("len(self.x_batches): " + str(len(self.x_batches)))
        # print("len(self.y_batches): " + str(len(self.y_batches)))
        # print("len(self.len_batches): " + str(len(self.len_batches)))
        # print("len(self.raw_lines_batches): " + str(len(self.raw_lines_batches)))



    def next_batch(self):
        batch_x = self.x_batches[self.pointer]
        batch_y = self.y_batches[self.pointer]
        xlen = self.len_batches[self.pointer]
        batch_line = self.raw_lines_batches[self.pointer]

        self.pointer += 1
        return batch_x, batch_y, xlen, batch_line

    def reset_batch_pointer(self):
        self.create_batches()
        self.pointer = 0

if __name__ == "__main__":
    pass
