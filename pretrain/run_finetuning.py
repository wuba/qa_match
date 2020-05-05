# -*- coding: utf-8 -*-
"""
funtune on pretrained model with trainset and devset
"""

import sys
import os
import tensorflow as tf
import numpy as np
import argparse
import models
import utils

from tensorflow.python.framework import graph_util

def evaluate(sess, full_tensors, args, model):
    total_num = 0
    right_num = 0
    for batch_data in utils.gen_batchs(full_tensors, args.batch_size, is_shuffle=False):
        softmax_re = sess.run(model.softmax_op,
                                        feed_dict={model.ph_dropout_rate: 0,
                                                   model.ph_tokens: batch_data[0],
                                                   model.ph_labels: batch_data[1],
                                                   model.ph_length: batch_data[2]})
        pred_re = np.argmax(softmax_re, axis=1)
        total_num += len(pred_re)
        right_num += np.sum(pred_re == batch_data[1])
        acc = 1.0 * right_num / (total_num + 1e-5)

    print("dev total num: ", total_num, ", right num: ", right_num, ", acc: ", acc)
    return acc

def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--dev_file", type=str, default="")
    parser.add_argument("--vocab_file", type=str, default="")
    parser.add_argument("--output_id2label_file", type=str, default="")
    parser.add_argument("--model_save_dir", type=str, default="")
    parser.add_argument("--lstm_dim", type=int, default=500)
    parser.add_argument("--embedding_dim", type=int, default=1000)
    parser.add_argument("--opt_type", type=str, default='adam')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print_step", type=int, default=1000)
    parser.add_argument("--init_checkpoint", type=str, default='')
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--layer_num", type=int, default=4)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    print(args)
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    print("load training sens")
    train_sens = utils.load_training_data(args.train_file, skip_invalid=True)
    print("\nload dev sens")
    dev_sens = utils.load_training_data(args.dev_file, skip_invalid=True)

    word2id, id2word, label2id, id2label = utils.load_vocab(train_sens + dev_sens, args.vocab_file)
    fw = open(args.output_id2label_file, 'w+')
    for k, v in id2label.items():
        fw.write(str(k) + "\t" + v + "\n")
    fw.close()

    utils.gen_ids(train_sens, word2id, label2id, args.max_len)
    utils.gen_ids(dev_sens, word2id, label2id, args.max_len)

    train_full_tensors = utils.make_full_tensors(train_sens)
    dev_full_tensors = utils.make_full_tensors(dev_sens)

    other_arg_dict = {}
    other_arg_dict['token_num'] = len(word2id)
    other_arg_dict['label_num'] = len(label2id)
    model = models.create_finetune_classification_training_op(args, other_arg_dict)

    steps_in_epoch = int(len(train_sens) // args.batch_size)
    print("batch size: ", args.batch_size, ", training sample num : ", len(train_sens), ", print step : ", args.print_step)
    print("steps_in_epoch : ", steps_in_epoch, ", epoch num :", args.epoch, ", total steps : ", args.epoch * steps_in_epoch)
    print_step = min(args.print_step, steps_in_epoch)
    print("eval dev every {print_step} step")

    save_vars = [v for v in tf.global_variables() if v.name.find('adam') < 0 and v.name.find('Adam') < 0 and v.name.find('ADAM') < 0]
    print(save_vars)
    print(tf.all_variables())

    saver = tf.train.Saver(max_to_keep=2)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        total_loss = 0
        dev_best_so_far = 0
        for epoch in range(1, args.epoch + 1):
            print("\n" + "*" * 20 + "epoch num :", epoch, "*" * 20)
            for batch_data in utils.gen_batchs(train_full_tensors, args.batch_size, is_shuffle=True):
                _, global_step, loss = sess.run([model.train_op, model.global_step_op, model.loss_op],
                                                feed_dict={model.ph_dropout_rate: args.dropout_rate,
                                                        model.ph_tokens: batch_data[0],
                                                        model.ph_labels: batch_data[1],
                                                        model.ph_length: batch_data[2]})
                total_loss += loss
                if global_step % print_step == 0:
                    print("\nglobal step : ", global_step, ", avg loss so far : ", total_loss / global_step)
                    print("begin to eval dev set: ")
                    acc = evaluate(sess, dev_full_tensors, args, model)
                    if acc > dev_best_so_far:
                        dev_best_so_far = acc
                        print("!" * 20, "best got : ", acc, flush=True)
                        # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["scores"])
                        saver.save(sess, args.model_save_dir + '/finetune.ckpt', global_step=global_step)

            print("\n----------------------eval after one epoch: ")
            print("global step : ", global_step, ", avg loss so far : ", total_loss / global_step)
            print("begin to eval dev set: ")
            sys.stdout.flush()
            acc = evaluate(sess, dev_full_tensors, args, model)
            if acc > dev_best_so_far:
                dev_best_so_far = acc
                print("!" * 20, "best got : ", acc)
                saver.save(sess, args.model_save_dir + '/finetune.ckpt', global_step=global_step)

if __name__ == "__main__":
    tf.app.run()