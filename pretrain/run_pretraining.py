# -*- coding: utf-8 -*-
"""
pretrain a specified language model(modified bi-lstm as default)
"""

from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import argparse
import models
import utils
import gc
import time


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--vocab_file", type=str, default="")
    parser.add_argument("--model_save_dir", type=str, default="")
    parser.add_argument("--lstm_dim", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--layer_num", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_step", type=int, default=10000)
    parser.add_argument("--warmup_step", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_step", type=int, default=1000)
    parser.add_argument("--max_predictions_per_seq", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--clip_norm", type=float, default=1)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--use_queue", type=int, default=0)
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--enqueue_thread_num", type=int, default=5)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    print(args)
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    # load data
    word2id, id2word = utils.load_vocab_file(args.vocab_file)
    training_sens = utils.load_pretraining_data(args.train_file, args.max_seq_len)

    if not args.use_queue:
        utils.to_ids(training_sens, word2id, args, id2word)

    other_arg_dict = {}
    other_arg_dict['token_num'] = len(word2id)

    # load model 
    model = models.create_bidirectional_lm_training_op(args, other_arg_dict)

    gc.collect()
    saver = tf.train.Saver(max_to_keep=2)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        if args.init_checkpoint:
            print('restore the checkpoint : ', args.init_checkpoint, flush=True)
            saver.restore(sess, args.init_checkpoint)

        total_loss = 0
        num = 0
        global_step = 0
        while global_step < args.train_step:
            if not args.use_queue:
                iterator = utils.gen_batches(training_sens, args.batch_size)
            else:
                iterator = utils.queue_gen_batches(training_sens, args, word2id, id2word)
            for batch_data in iterator:
                feed_dict = {model.ph_tokens: batch_data[0],
                            model.ph_length: batch_data[1],
                            model.ph_labels: batch_data[2],
                            model.ph_positions: batch_data[3],
                            model.ph_weights: batch_data[4],
                            model.ph_dropout_rate: args.dropout_rate}
                _, global_step, loss, learning_rate = sess.run([model.train_op, \
                    model.global_step, model.loss_op, model.learning_rate_op], feed_dict=feed_dict)

                total_loss += loss
                num += 1
                if global_step % args.print_step == 0:
                    print("\nglobal step : ", global_step, 
                        ", avg loss so far : ", total_loss / num, 
                        ", instant loss : ", loss,
                        ", learning_rate : ", learning_rate,
                        ", time :", time.strftime('%Y-%m-%d %H:%M:%S'))
                    sys.stdout.flush()
                    print("save model ...")
                    saver.save(sess, args.model_save_dir + '/lm_pretrain.ckpt', global_step=global_step)
                    gc.collect()

            if not args.use_queue:
                utils.to_ids(training_sens, word2id, args, id2word)  # MUST run this for randomization for each sentence
            gc.collect()

if __name__ == "__main__":
    tf.app.run()