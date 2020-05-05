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

MINCLASS_HIGH_SCORE_ONLY = 0.9
MINCLASS_MID_SCORE_ONLY = 0.75

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
        line_items = line.split("\t")
        real_top_max_label = line_items[0]
        pre_top_max_label = line_items[2].split(' ')[0].split(":")[0]
        pre_top_max_score = float(line_items[2].split(' ')[0].split(":")[1])
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
        line_items = line.split("<split>")
        real_min_label = line_items[0]
        label_scores_list = line_items[2].split(" ")
        pre_top_min_label = label_scores_list[0].split(":")[0]
        pre_top_min_score = float(label_scores_list[0].split(":")[1])
        mer_obj.real_min_label = real_min_label
        mer_obj.pre_min_top_label = pre_top_min_label
        mer_obj.pre_min_top_score = pre_top_min_score
        mer_obj.pre_min_label_scores = []
        scores_list = mer_obj.pre_min_label_scores

        for i in range(len(label_scores_list)):
            label_score = LabelScore()
            temp_labels= label_scores_list[i].split(":")
            if len(temp_labels) < 2:
                continue
            label_score.label = temp_labels[0]
            label_score.score = float(temp_labels[1])
            scores_list.append(label_score)
    min_pre_file.close()
    return merge_items

def get_merge_result_each(str_type, merge_item):
    assert str_type in ('__label__none', '__label__only', '__label__list')
    if str_type == "__label__none":
        merge_item.merge_result = "__label__none"
    elif str_type == "__label__only":
        merge_item.merge_result = merge_item.pre_min_top_label + ":" + str(merge_item.pre_min_top_score)
    elif str_type == "__label__list":
        merge_item.merge_result = ""
        for i in range(len(merge_item.pre_min_label_scores)):
            if i == MINCLASS_NUM:
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

def get_result_by_min(min_pre_file_dir, result_file_dir):
    with open(min_pre_file_dir, 'r', encoding='utf-8') as f_pre_min:
        with open(result_file_dir, 'w', encoding='utf-8') as f_res:
            for line in f_pre_min:
                lines = line.strip().split('<split>')
                real_label = lines[0]
                model_label_scores = lines[2].split(' ')
                temp_label_score_list = []
                write_str = '__label__\t' + str(real_label) + '\t'
                for label_score in model_label_scores:
                    label_scores = label_score.split(':')
                    temp_label_score = LabelScore()
                    temp_label_score.label = label_scores[0]
                    temp_label_score.score = (float)(label_scores[1])
                    temp_label_score_list.append(temp_label_score)
                if temp_label_score_list[0].score < MINCLASS_MID_SCORE_ONLY:  #refuse answer
                    write_str += '__label__none'
                elif temp_label_score_list[0].score >= MINCLASS_MID_SCORE_ONLY and temp_label_score_list[0].score < MINCLASS_HIGH_SCORE_ONLY:  #list answer
                    for i in range(len(temp_label_score_list)):
                        if i == MINCLASS_NUM:
                            break
                        write_str += str(temp_label_score_list[i].label) + ':' + str(temp_label_score_list[i].score) + ','
                else: #only answer
                    write_str += str(temp_label_score_list[0].label) + ':' + str(temp_label_score_list[0].score)
                f_res.write(write_str + "\n")

def get_acc_recall_f1(result_file_dir):
    only_real_num = 0
    only_model_num = 0
    only_right_num = 0
    list_real_num = 0
    list_model_num = 0
    list_right_num = 0
    none_real_num = 0
    none_model_num = 0
    none_right_num = 0
    num = 0
    with open(result_file_dir, 'r', encoding='utf-8') as f_pre:
        for line in f_pre:
            num =  num + 1
            lines = line.strip().split('\t')
            # print(lines)
            if lines[1] == '0':
                none_real_num = none_real_num + 1
            elif ',' in lines[1]:
                list_real_num = list_real_num + 1
            else:
                only_real_num = only_real_num + 1
            model_label_scores = lines[2].split(',')
            if lines[2] == '__label__none':
                none_model_num = none_model_num + 1
            elif len(model_label_scores) == 1:
                only_model_num = only_model_num + 1
            else:
                list_model_num = list_model_num + 1
            real_labels_set = set(lines[1].split(','))
            if lines[1] == '0' and lines[2] == '__label__none':
                none_right_num = none_right_num + 1
            if len(real_labels_set) == 1 and len(model_label_scores) == 1 and lines[1] in lines[2]:
                only_right_num = only_right_num + 1
            if len(real_labels_set) > 1 and len(model_label_scores) > 1:
                for i in range(len(model_label_scores)):
                    label_scores = model_label_scores[i].split(":")
                    if label_scores[0] in real_labels_set:
                        list_right_num = list_right_num + 1
                        break
    print('none_right_num: ' + str(none_right_num) + ', list_right_num: ' + str(list_right_num) + ', only_right_num: ' + str(only_right_num))
    print('none_real_num: ' + str(none_real_num) + ', list_real_num: ' + str(list_real_num) + ', only_real_num: ' + str(only_real_num))
    print('none_model_num: ' + str(none_model_num) + ', list_model_num: ' + str(list_model_num) + ', only_model_num: ' + str(only_model_num))
    all_right_num = list_right_num + only_right_num
    all_real_num = list_real_num + only_real_num
    all_model_num = list_model_num + only_model_num
    print('all_right_num: ' + str(all_right_num) + ', all_real_num: ' + str(all_real_num) + ', all_model_num: ' + str(all_model_num))
    all_acc = all_right_num / all_model_num
    all_recall = all_right_num / all_real_num
    all_f1 = 2 * all_acc * all_recall / (all_acc + all_recall)
    print("all_acc: " + str(all_acc) + ", all_recall: " + str(all_recall) + ", all_f1: " + str(all_f1))
    only_acc = only_right_num / only_model_num
    only_recall = only_right_num / only_real_num
    only_f1 = 2 * only_acc * only_recall / (only_acc + only_recall)
    print("only_acc: " + str(only_acc) + ", only_recall: " + str(only_recall) + ", only_f1: " + str(only_f1))
    list_acc = list_right_num / list_model_num
    list_recall = list_right_num / list_real_num
    list_f1 = 2 * list_acc * list_recall / (list_acc + list_recall)
    print("list_acc: " + str(list_acc) + ", list_recall: " + str(list_recall) + ", list_f1: " + str(list_f1))
    none_acc = none_right_num / none_model_num
    none_recall = none_right_num / none_real_num
    none_f1 = 2 * none_acc * none_recall / (none_acc + none_recall)
    print("none_acc: " + str(none_acc) + ", none_recall: " + str(none_recall) + ", none_f1: " + str(none_f1))

if __name__ == "__main__":
    max_pre_file_dir = sys.argv[1]
    min_pre_file_dir = sys.argv[2]
    result_file_dir = sys.argv[3]
    std_label_ques = sys.argv[4]
    if max_pre_file_dir == 'no' or std_label_ques == 'no':  #only use min_pre result
        get_result_by_min(min_pre_file_dir, result_file_dir)
    else:  #merge max_pre result and min_pre result
        merge_items_list = get_pre_label_scores(max_pre_file_dir, min_pre_file_dir)
        min_max_map = get_max2min_label(std_label_ques)
        get_merge_result(merge_items_list, min_max_map)
        write_result(merge_items_list, result_file_dir)
    #get acc recall f1
    get_acc_recall_f1(result_file_dir)
