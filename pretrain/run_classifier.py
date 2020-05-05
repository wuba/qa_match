# -*- coding: UTF-8 -*-
"""
predict with finetuned model with testset
"""

import os
import sys
import tensorflow as tf
import numpy as np
import argparse
import gc
import time
import codecs
import utils

def get_output(g):
    return {"softmax": g.get_tensor_by_name("Softmax:0")}

def get_input(g):
    return {"tokens": g.get_tensor_by_name("ph_tokens:0"),
            "length": g.get_tensor_by_name("ph_length:0"),
            "dropout_rate": g.get_tensor_by_name("ph_dropout_rate:0")}

def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--vocab_file", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="", help="directory which contains model")
    parser.add_argument("--id2label_file", type=str, default="")
    args = parser.parse_args()

    word2id, id2word = utils.load_vocab_file(args.vocab_file)
    sys.stderr.write("vocab num : " + str(len(word2id)) + "\n")

    sens = utils.gen_test_data(args.input_file, word2id)
    sys.stderr.write("sens num : " + str(len(sens)) + "\n")

    id2label = utils.load_id2label_file(args.id2label_file)
    sys.stderr.write('label num : ' + str(len(id2label)) + "\n")

    # use latest checkpoint
    if "" == args.model_path:
        args.model_path = tf.train.latest_checkpoint(checkpoint_dir=args.model_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(args.model_path))
        saver.restore(sess, args.model_path)

        graph = tf.get_default_graph()
        input_dict = get_input(graph)
        output_dict = get_output(graph)

        for sen in sens:
            re = sess.run(output_dict['softmax'], feed_dict={input_dict['tokens']: [sen],
                                                            input_dict['length']: [len(sen)],
                                                            input_dict["dropout_rate"]: 0.0})
            sorted_idx = np.argsort(-1 * re[0])  # sort by desc
            s = ""
            for i in sorted_idx[:3]:
                s += id2label[i] + "|" + str(re[0][i]) + ","
            print(s + "\t" + str(" ".join([id2word[t] for t in sen])))

if __name__ == "__main__":
    tf.app.run()