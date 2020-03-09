# coding=utf-8

import sys

#

if __name__ == "__main__":
    input_file_dir = sys.argv[1]
    raw_file_dir = sys.argv[2]
    output_file_dir = sys.argv[3]
    input_file = open(input_file_dir, 'r', encoding='utf-8')
    vocab_index = {}
    for line in input_file.readlines():
        tokens = line.split()
        for token in tokens[1:]:
            if token not in vocab_index:
                vocab_index[token] = len(vocab_index)
    input_file.close()
    input_file = open(raw_file_dir, 'r', encoding='utf-8')
    output_file = open(output_file_dir, 'w', encoding='utf-8')
    for line in input_file.readlines():
        tokens = line.split()
        token_str = tokens[0]
        for token in tokens[1:]:
            token_str = token_str + " " + str(vocab_index[token])
        output_file.write(token_str + '\n')
    input_file.close()
    output_file.close()

    pass