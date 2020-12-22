# -*- coding: utf-8 -*-
"""
inference cluster labels
"""
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import silhouette_score, silhouette_samples
import dataset
import dec_model
import data_utils


def write_results(z, y_true, raw_q, out_name, prob):
    assert len(z) == len(raw_q)
    out_str = ""
    label_map = {} # sort samples order by y_pred
    for (y_pred, gt_label, q, pro) in zip(z, y_true, raw_q, prob):
        prob = -np.sort(-prob)
        if y_pred in label_map:
            label_map[y_pred].append("__label__" + str(y_pred) + "\t" + q  +
                                     ": ground truth label is " + str(gt_label) + str(pro))
        else:
            label_map[y_pred] = []
            label_map[y_pred].append("__label__" + str(y_pred) + "\t" + q  +
                                     ": ground truth label is" + str(gt_label) + str(pro))

    for _, lines in label_map.items():
        for line in lines:
            out_str += line + "\n"
    data_utils.write_file(out_name, out_str)

def print_metrics(x, labels):
    sil_avg = silhouette_score(x, labels) # avg silhouette score
    sils = silhouette_samples(x, labels) # silhouette score of each sample
    tf.logging.info("avg silhouette：" + str(sil_avg))

def inference(data, model, params):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
    batch_size = params.batch_size

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, params.model_path)

        train_size = len(data.train_x)
        step_by_batch = train_size // batch_size + 1
        tf.logging.info("step by batch " + str(step_by_batch))
        z_total = [] # z: transformed representation
        prob_total = [] # predict cluster probability
        pred_total = [] # predict cluster label

        for idx in range(step_by_batch):
            if idx == step_by_batch - 1:
                tf.logging.info("start/ end idx " + str(idx * batch_size) + "  " + str(idx * batch_size + batch_size))
                cur_pred, cur_prob, cur_z  = sess.run(
                    [model.pred, model.pred_prob, model.z], feed_dict={
                    model.pretrained_model.ph_tokens: data.train_x[idx * batch_size:],
                    model.pretrained_model.ph_length: data.train_length_x[idx * batch_size:],
                    model.pretrained_model.ph_dropout_rate: 0
                })
            else:
                cur_pred, cur_prob, cur_z = sess.run(
                    [model.pred, model.pred_prob, model.z], feed_dict={
                    model.pretrained_model.ph_tokens:
                        data.train_x[idx * batch_size: idx * batch_size + batch_size],
                    model.pretrained_model.ph_length:
                        data.train_length_x[idx * batch_size: idx * batch_size + batch_size],
                    model.pretrained_model.ph_dropout_rate: 0
                })

            now = datetime.now()
            # tf.logging.info("sess run index  " + str(idx) + " " +  str(len(cur_pred)) + now.strftime("%H:%M:%S"))
            pred_total.extend(cur_pred)
            prob_total.extend(cur_prob)
            z_total.extend(cur_z)
        tf.logging.info("pred total " + str(len(pred_total)) + " , sample total " + str(len(data.train_x)))
        assert len(pred_total) == len(data.train_x)
        clust_label = np.array(pred_total)
        prob = np.array(prob_total)
    print_metrics(z_total, clust_label)
    # write inference result file
    write_results(clust_label, data.gt_label, data.raw_q, params.pred_score_path, prob)
    return clust_label

if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lstm_dim", type=int, default=500)
    parser.add_argument("--embedding_dim", type=int, default=1000)
    parser.add_argument("--vocab_file", type=str, default="./vocab")
    parser.add_argument("--external_cluster_center", type=str, default="")
    parser.add_argument("--n_clusters", type=int, default=20)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--layer_num", type=int, default=1)
    parser.add_argument("--token_num", type=int, default=7820)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pred_score_path", type=str, default='')
    args = parser.parse_args()

    word2id, id2word = data_utils.load_vocab_file(args.vocab_file)
    TRAINSET_SIZE = len(data_utils.load_sentences(args.train_file, skip_invalid=True))
    other_arg_dict = {}
    other_arg_dict['token_num'] = len(word2id)
    other_arg_dict['trainset_size'] = TRAINSET_SIZE

    exp_data = dataset.ExpDataset(args)
    dec_model = dec_model.DEC(args, other_arg_dict)
    inference(exp_data, dec_model, args)
