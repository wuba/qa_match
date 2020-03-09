# coding=utf-8

"""
tools for run bi-lstm + dssm  short text match
"""

import random
import math
import copy


class DataHelper(object):

    def __init__(self, train_path, valid_path, test_path, standard_path, batch_size, epcho_num):
        self.train_test_generator(train_path, valid_path, test_path, standard_path, batch_size, epcho_num)

    def train_test_generator(self, train_path, valid_path, test_path, standard_path, batch_size, epcho_num):
        self.label2id = {}
        self.id2label = {}
        self.vocab2id = {}
        self.vocab2id['PAD'] = 0
        self.vocab2id['UNK'] = 1
        self.id2vocab = {}
        self.id2vocab[0] = 'PAD'
        self.id2vocab[1] = 'UNK'
        #standard question
        file = open(standard_path, 'r', encoding='utf-8')
        self.std_id_ques = {}
        max_std_len = 0
        for line in file.readlines():
            label_words = line.strip().split(" ")
            label = label_words[0]
            if label not in self.label2id:
                self.label2id[label] = len(self.label2id)
                self.id2label[self.label2id[label]] = label
            w_temp = []
            for word in label_words[1:]:
                if word not in self.vocab2id:
                    self.vocab2id[word] = len(self.vocab2id)
                    self.id2vocab[self.vocab2id[word]] = word
                w_temp.append(self.vocab2id[word])
            if max_std_len < len(w_temp):
                max_std_len = len(w_temp)
            self.std_id_ques[self.label2id[label]] = (len(w_temp), w_temp, label, line.strip())
        file.close()
        self.std_batch = []
        self.predict_label_seq = []
        self.predict_id_seq = []
        #when predicted test data must order by this sequence
        for std_id, ques_info in self.std_id_ques.items():
            self.std_batch.append((ques_info[0], ques_info[1]))
            self.predict_label_seq.append(label)
            self.predict_id_seq.append(std_id)
        self.train_num = 0
        self.train_id_ques = {}
        file = open(train_path, 'r', encoding='utf-8')
        for line in file.readlines():
            label_words = line.strip().split()
            label = label_words[0]
            if label not in self.label2id:
                self.label2id[label] = len(self.label2id)
                self.id2label[self.label2id[label]] = label
            w_temp = []
            for word in label_words[1:]:
                if word not in self.vocab2id:
                    self.vocab2id[word] = len(self.vocab2id)
                    self.id2vocab[self.vocab2id[word]] = word
                w_temp.append(self.vocab2id[word])
            label_id = self.label2id[label]
            if label_id not in self.train_id_ques:
                self.train_id_ques[label_id] = []
                self.train_id_ques[label_id].append((len(w_temp), w_temp))
            else:
                self.train_id_ques[label_id].append((len(w_temp), w_temp))
            self.train_num = self.train_num + 1
        file.close()
        self.vocab_size = len(self.vocab2id)
        #std question padding
        for ques_info in self.std_batch:
            for _ in range(max_std_len - ques_info[0]):
                ques_info[1].append(self.vocab2id['PAD'])
        file = open(valid_path, 'r', encoding='utf-8')
        self.valid_num = 0
        self.valid_batch = []
        for line in file.readlines():
            label_words = line.strip().split()
            label = label_words[0]
            if label not in self.label2id:
                print("label not in label2id: strings is: " + line)
                continue
            w_temp = []
            for word in label_words[1:]:
                if word not in self.vocab2id:
                    w_temp.append(self.vocab2id['UNK'])
                else:
                    w_temp.append(self.vocab2id[word])
            self.valid_batch.append((len(w_temp), w_temp, label, line.strip()))
            self.valid_num = self.valid_num + 1
        file.close()
        file = open(test_path, 'r', encoding='utf-8')
        self.test_num = 0
        self.test_batch = []
        for line in file.readlines():
            label_words = line.strip().split()
            label = label_words[0]
            w_temp = []
            for word in label_words[1:]:
                if word not in self.vocab2id:
                    w_temp.append(self.vocab2id['UNK'])
                else:
                    w_temp.append(self.vocab2id[word])
            self.test_batch.append((len(w_temp), w_temp, label, line.strip()))
            self.test_num = self.test_num + 1
        file.close()
        self.batch_size = batch_size
        self.train_num_epcho = epcho_num
        self.train_num_batch = math.ceil(self.train_num / self.batch_size)
        self.valid_num_batch = math.ceil(self.valid_num / self.batch_size)
        self.valid_num_batch = math.ceil(self.valid_num / self.batch_size)
        self.test_num_batch = math.ceil(self.test_num / self.batch_size)

    def weight_random(self, label_questions, batch_size):
        def index_choice(weight):
            index_sum_weight = random.randint(0, sum(weight) - 1)
            for i, val in enumerate(weight):
                index_sum_weight -= val
                if index_sum_weight < 0:
                    return i
            return 0
        batch_keys = []
        keys = list(label_questions.keys()).copy()
        weights = [len(label_questions[key]) for key in keys]
        for _ in range(batch_size):
            index = index_choice(weights)
            key = keys.pop(index)
            batch_keys.append(key)
            weights.pop(index)
        return batch_keys

    def train_batch_iterator(self, label_questions, standard_label_question):
        '''
        select a couple question for each class
        '''
        num_batch = self.train_num_batch
        num_epcho = self.train_num_epcho
        for _ in range(num_batch * num_epcho):
            query_batch = []
            doc_batch = []
            batch_keys = self.weight_random(label_questions, self.batch_size)
            batch_query_max_num = 0
            for key in batch_keys:
                questions = copy.deepcopy(random.sample(label_questions[key], 1)[0])
                current_num = questions[0]
                if current_num > batch_query_max_num:
                    batch_query_max_num = current_num
                query_batch.append(questions)
                doc = standard_label_question[key]
                doc_batch.append(doc)
            #padding
            for query, doc in zip(query_batch, doc_batch):
                for _ in range(batch_query_max_num - query[0]):
                    query[1].append(self.vocab2id['PAD'])
            yield query_batch, doc_batch

    def valid_batch_iterator(self):
        num_batch = self.valid_num_batch
        num_epcho = 1
        for i in range(num_batch * num_epcho):
            if i * self.batch_size + self.batch_size < self.valid_num:
                query_batch = copy.deepcopy(self.valid_batch[i * self.batch_size : i * self.batch_size + self.batch_size])
            else:
                query_batch = copy.deepcopy(self.valid_batch[i * self.batch_size : ])
            batch_query_max_num = 0
            for q_len, _, _, _ in query_batch:
                if batch_query_max_num < q_len:
                    batch_query_max_num = q_len
            #padding
            for q_len, label_words, _, _ in query_batch:
                for _ in range(batch_query_max_num - q_len):
                    label_words.append(self.vocab2id['PAD'])
            yield query_batch

    def test_batch_iterator(self):
        num_batch = self.test_num_batch
        num_epcho = 1
        for i in range(num_batch * num_epcho):
            if i * self.batch_size + self.batch_size < self.test_num:
                query_batch = copy.deepcopy(self.test_batch[i * self.batch_size : i * self.batch_size + self.batch_size])
            else:
                query_batch = copy.deepcopy(self.test_batch[i * self.batch_size : ])
            batch_query_max_num = 0
            for q_len, _, _, _ in query_batch:
                if batch_query_max_num < q_len:
                    batch_query_max_num = q_len
            #padding
            for q_len, label_words, _, _ in query_batch:
                for _ in range(batch_query_max_num - q_len):
                    label_words.append(self.vocab2id['PAD'])
            yield query_batch

if __name__ == "__main__":
    pass
