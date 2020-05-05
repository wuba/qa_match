"""
format result by applying thresholds
"""
import codecs
import sys

# none
TH_ONLY = 0.9
LABEL_NONE = "__label__none"
# list
TH_LIST = 0.1
LABEL_LIST = "__label__list"
# only
LABEL_ONLY = "__label__only"

if __name__ == "__main__":
    model_result_file = sys.argv[1]
    result_file = sys.argv[2]

    fout = codecs.open(result_file, encoding='utf-8', mode='w+')
    with codecs.open(model_result_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            s_line = line.split('\t')
            label_type = LABEL_NONE
            label_list = LABEL_NONE
            if len(s_line) >= 2:
                model_res = s_line[0]

                model_res = model_res.replace('|'," ").replace(","," ").split(' ')
                s_item_1 = model_res[0:2]
                s_item_2 = model_res[2:4]

                # TODO we use 0 as deny class label. 
                #  consider updating the code below if the label is different.
                if s_item_1[0] == "__label__0":
                    pass
                elif float(s_item_1[1]) >= TH_ONLY:
                    label_type = LABEL_ONLY
                    label_list = s_item_1[0]
                elif abs(float(s_item_1[1]) - float(s_item_2[1])) <= TH_LIST:
                    if s_item_2[0] == "__label__0":
                        pass
                    else:
                        label_type = LABEL_LIST
                        label_list = ','.join([s_item_1[0], s_item_2[0]])
                else:
                    pass
                # __label__list 1,2 123 221 534 65
                fout.write("{}\t{}\t{}\n".format(label_type, label_list, s_line[1]))
        f.close()
    fout.close()