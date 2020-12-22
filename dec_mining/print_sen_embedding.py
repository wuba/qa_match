# -*- coding: utf-8 -*-
"""
print sentence embedding
"""
import os
import sys
import argparse
import codecs
import tensorflow as tf
import numpy as np
import data_utils

# get graph output in different way: max mean concat
def get_output(g, embedding_way):
    if embedding_way == "concat":  # here may have problem, this is just for 4 layers of biLM !
        t = g.get_tensor_by_name("concat_4:0")
    elif embedding_way == "max":
        t = g.get_tensor_by_name("Max:0")
    elif embedding_way == 'mean':
        t = g.get_tensor_by_name("Mean:0")
    else:
        assert False
    return {"sen_embedding": t}

# get graph input
def get_input(g):
    return {"tokens": g.get_tensor_by_name("ph_tokens:0"),
            "length": g.get_tensor_by_name("ph_length:0"),
            "dropout_rate": g.get_tensor_by_name("ph_dropout_rate:0")}

def gen_test_data(input_file, word2id, max_seq_len):
    sens = []
    center_size = []
    for line in codecs.open(input_file, 'r', 'utf-8'):
        # separated by slash
        ls = line.strip().split("/")
        center_size.append(len(ls))
        for l in ls:
            l = l.replace("", " ")
            fs = l.rstrip().split()
            if len(fs) > max_seq_len:
                continue
            sen = []
            for f in fs:
                if f in word2id:
                    sen.append(word2id[f])
                else:
                    sen.append(word2id['<UNK>'])
            sens.append(sen)
    return sens, center_size

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--vocab_file", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="")
    # sentence representation output way : max mean concat
    parser.add_argument("--embedding_way", type=str, default="concat")
    args = parser.parse_args()

    word2id, id2word = data_utils.load_vocab_file(args.vocab_file)
    sys.stderr.write("vocab num : " + str(len(word2id)) + "\n")
    sens, center_size = gen_test_data(args.input_file, word2id, args.max_seq_len)
    sys.stderr.write("sens num : " + str(len(sens)) + "\n")
    tf.logging.info("embedding_way : ", args.embedding_way)

    # limit cpu resource
    cpu_num = int(os.environ.get('CPU_NUM', 15))
    config = tf.ConfigProto(device_count={"CPU": cpu_num},
                    inter_op_parallelism_threads = cpu_num,
                    intra_op_parallelism_threads = cpu_num,
                    log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(args.model_path))
        saver.restore(sess, args.model_path)

        graph = tf.get_default_graph()
        input_dict = get_input(graph)
        output_dict = get_output(graph, args.embedding_way)

        caches = []
        idx = 0
        while idx < len(sens):
            batch_sens = sens[idx:idx + args.batch_size]
            batch_tokens = []
            batch_length = []
            for sen in batch_sens:
                batch_tokens.append(sen)
                batch_length.append(len(sen))

            real_max_len = max([len(b) for b in batch_tokens])
            for b in batch_tokens:
                b.extend([0] * (real_max_len - len(b)))

            re = sess.run(output_dict['sen_embedding'],
                          feed_dict={input_dict['tokens']: batch_tokens,
                          input_dict['length']: batch_length,
                          input_dict["dropout_rate"]: 0.0})
            if len(caches) % 200 == 0:
                tf.logging.info(len(caches))
            caches.append(re)
            idx += len(batch_sens)

    sen_embeddings = np.concatenate(caches, 0)
    # calculate average embedding
    avg_centers = []

    idx = 0
    for size in center_size:
        avg_center_emb = np.average(sen_embeddings[idx: idx + size], axis=0)
        avg_centers.append(avg_center_emb)
        idx = idx + size

    np.savetxt(args.output_file, avg_centers, fmt='%.3e')
