# -*- coding: utf-8 -*-

"""
format result to qa_match standard format
USAGE: 
    python format_result.py [test_file] [model_result] [standard_format_output] 
"""
import codecs
import sys

if __name__ == "__main__":
    # real result: label\tidx\tsentence
    real_result_file = sys.argv[1]
    # model result: __label__1|score_1,... \tsentence
    model_result_file = sys.argv[2]
    result_file = sys.argv[3]

    real_labels = []
    for line in codecs.open(real_result_file, encoding='utf-8'):
        if len(line.split('\t')) == 3:
            real_labels.append(line.split('\t')[0])

    fout = codecs.open(result_file, encoding='utf-8', mode='w+')
    with codecs.open(model_result_file, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            s_line = line.split('\t')
            if len(s_line) >= 2:
                model_res = s_line[0]
                sentence = s_line[1]

                model_res = model_res.replace("__label__", "") \
                    .replace('|', ":").replace(",", " ")
                # real_label\tsentence\tmodel_labels
                fout.write("{}\t{}\t{}\n".format(real_labels[idx],
                                                    sentence, model_res))
        f.close()
    fout.close()
