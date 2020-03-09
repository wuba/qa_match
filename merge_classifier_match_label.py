#! /usr/bin/python
#encoding=utf-8
#author wangyong

"""
merge result for domain identification and intent recognition
"""

import sys

#none
MAXCLASS_NONE_HIGHSCORE = 0.6
MAXCLASS_NONE_MIDSCORE = 0.4
MINCLASS_NONE_HIGHSCORE1 = 0.985
MINCLASS_NONE_HIGHSCORE2 = 0.95
MINCLASS_NONE_MIDSCORE1 = 0.75
MINCLASS_NONE_MIDSCORE2 = 0.7
#list
MAXCLASS_LIST_HIGHSCORE = 0.92
MAXCLASS_LIST_MIDSCORE = 0.6
MINCLASS_LIST_HIGHSCORE1 = 0.985
MINCLASS_LIST_HIGHSCORE2 = 0.975
MINCLASS_LIST_MIDSCORE1 = 0.5
MINCLASS_LIST_MIDSCORE2 = 0.6
#only
MAXCLASS_ONLY_HIGHSCORE = 0.78
MINCLASS_ONLY_HIGHSCORE1 = 0.88
MINCLASS_ONLY_HIGHSCORE2 = 0.93
MINCLASS_ONLY_MIDSCORE1 = 0.57
MINCLASS_ONLY_MIDSCORE2 = 0.62

MINCLASS_ZERO_SCORE = 0.00001

MINCLASS_NUM = 3

class LabelScore:
    def __init__(self):
        self.label = ''
        self.score = 0

class MergeObj(object):
    def __init__(self):
        self.real_max_label = ''
        self.real_min_label = ''
        self.pre_max_top_label = ''
        self.pre_max_top_score = 0
        self.pre_min_top_label = ''
        self.pre_min_top_score = 0
        self.pre_min_label_scores = [0]
        self.merge_result = ''

def get_max2min_label(max_min_class_file_dir):
    min_max_m = {}
    max_min_class_file = open(max_min_class_file_dir, 'r', encoding='utf-8')
    for line in max_min_class_file.readlines():
        mems = line.split("\t")
        max_label = mems[0]
        min_label = mems[1]
        min_max_m[min_label] = max_label
    max_min_class_file.close()
    return min_max_m

def get_pre_label_scores(max_pre_file_d, min_pre_file_d):
    merge_items = []
    max_pre_file = open(max_pre_file_d, 'r', encoding='utf-8')
    for line in max_pre_file.readlines():
        line_items = line.split(" ")
        real_top_max_label = line_items[0]
        pre_top_max_label = line_items[1].split(":")[0]
        pre_top_max_score = float(line_items[1].split(":")[1])
        mer_obj = MergeObj()
        mer_obj.pre_max_top_label = pre_top_max_label
        mer_obj.pre_max_top_score = float(pre_top_max_score)
        mer_obj.real_max_label = real_top_max_label
        merge_items.append(mer_obj)
    max_pre_file.close()
    min_pre_file = open(min_pre_file_d, 'r', encoding='utf-8')
    index = 0
    for line in min_pre_file.readlines():
        mer_obj = merge_items[index]
        index = index + 1
        line_items = line.split(" ")
        real_min_label = line_items[0]
        pre_top_min_label = line_items[1].split(":")[0]
        pre_top_min_score = float(line_items[1].split(":")[1])
        mer_obj.real_min_label = real_min_label
        mer_obj.pre_min_top_label = pre_top_min_label
        mer_obj.pre_min_top_score = pre_top_min_score
        mer_obj.pre_min_label_scores = []
        scores_list = mer_obj.pre_min_label_scores
        for i in range(len(line_items) - 1):
            label_score = LabelScore()
            label_score.label = line_items[i + 1].split(":")[0]
            label_score.score = float(line_items[i + 1].split(":")[1])
            scores_list.append(label_score)
    min_pre_file.close()
    return merge_items

def get_merge_result_each(str_type, merge_item):
    assert str_type in ('__label__none', '__label__only', '__label__list')
    if str_type == "__label__none":
        merge_item.merge_result = "__label__none"
    elif str_type == "__label__only":
        merge_item.merge_result = "__label__only\t" + merge_item.pre_min_top_label + ":" + str(merge_item.pre_min_top_score)
    elif str_type == "__label__list":
        merge_item.merge_result = "__label__list\t"
        for i in range(len(merge_item.pre_min_label_scores)):
            if i > MINCLASS_NUM:
                break
            label = merge_item.pre_min_label_scores[i].label
            score = merge_item.pre_min_label_scores[i].score
            if score < MINCLASS_ZERO_SCORE:
                break
            merge_item.merge_result = merge_item.merge_result + label + ":" + str(score) + ","

def get_only_list_none_result(high_score, low_score, merge_item):
    if merge_item.pre_min_top_score >= high_score:  #one answer
        get_merge_result_each("__label__only", merge_item)
    elif merge_item.pre_min_top_score < high_score and merge_item.pre_min_top_score >= low_score:  # list answer
        get_merge_result_each("__label__list", merge_item)
    else:  # refuse to answer
        get_merge_result_each("__label__none", merge_item)

def get_merge_result(merge_items, min_max_m):
    for merge_item in merge_items:
        if merge_item.pre_max_top_label == "__label__none":  #none
            if merge_item.pre_max_top_score >= MAXCLASS_NONE_HIGHSCORE:  #direct rejection
                get_merge_result_each("__label__none", merge_item)
            elif merge_item.pre_max_top_score >= MAXCLASS_NONE_MIDSCORE and merge_item.pre_max_top_score < MAXCLASS_NONE_HIGHSCORE: #tendency to reject
                get_only_list_none_result(MINCLASS_NONE_HIGHSCORE1, MINCLASS_NONE_MIDSCORE1, merge_item)
            else:  #not tendency to reject
                get_only_list_none_result(MINCLASS_NONE_HIGHSCORE2, MINCLASS_NONE_MIDSCORE2, merge_item)
        elif merge_item.pre_max_top_label == "__label__list":  #list
            if merge_item.pre_max_top_score >= MAXCLASS_LIST_HIGHSCORE:  #direct answer a list
                get_merge_result_each("__label__list", merge_item)
            elif merge_item.pre_max_top_score >= MAXCLASS_LIST_MIDSCORE and merge_item.pre_max_top_score < MAXCLASS_LIST_HIGHSCORE:  #tendency to answer list
                get_only_list_none_result(MINCLASS_LIST_HIGHSCORE1, MINCLASS_LIST_MIDSCORE1, merge_item)
            else:  #not tendency to answer list
                get_only_list_none_result(MINCLASS_LIST_HIGHSCORE2, MINCLASS_LIST_MIDSCORE2, merge_item)
        else:  #only
            filter_pre_min_label_scores = []
            for label_score in merge_item.pre_min_label_scores:
                max_label = min_max_m[label_score.label]
                if max_label != merge_item.pre_max_top_label:
                    continue
                filter_pre_min_label_scores.append(label_score)
            merge_item.pre_min_label_scores = filter_pre_min_label_scores
            if len(filter_pre_min_label_scores) == 0:  #direct rejection
                get_merge_result_each("__label__none", merge_item)
            else:
                merge_item.pre_min_top_label = filter_pre_min_label_scores[0].label
                merge_item.pre_min_top_score = filter_pre_min_label_scores[0].score
                if merge_item.pre_max_top_score >= MAXCLASS_ONLY_HIGHSCORE:  #not tendency to reject
                    get_only_list_none_result(MINCLASS_ONLY_HIGHSCORE1, MINCLASS_ONLY_MIDSCORE1, merge_item)
                else:  #not tendency to one answer
                    get_only_list_none_result(MINCLASS_ONLY_HIGHSCORE2, MINCLASS_ONLY_MIDSCORE2, merge_item)

def write_result(merge_items, result_file_d):
    min_pre_file = open(result_file_d, 'w', encoding='utf-8')
    for merge_item in merge_items:
        min_pre_file.write(merge_item.real_max_label + "\t" + merge_item.real_min_label + "\t" + merge_item.merge_result + "\n")
    min_pre_file.close()

if __name__ == "__main__":
    max_min_label_file_dir = sys.argv[1]
    max_pre_file_dir = sys.argv[2]
    min_pre_file_dir = sys.argv[3]
    result_file_dir = sys.argv[4]
    min_max_map = get_max2min_label(max_min_label_file_dir)
    merge_items_list = get_pre_label_scores(max_pre_file_dir, min_pre_file_dir)
    get_merge_result(merge_items_list, min_max_map)
    write_result(merge_items_list, result_file_dir)
