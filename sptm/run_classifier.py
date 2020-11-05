# -*- coding: utf-8 -*-
"""
finetune on pretrained model with trainset and devset
"""

import sys
import os
import tensorflow as tf
import numpy as np
import argparse
import models
import utils


def evaluate(sess, full_tensors, args, model):
    total_num = 0
    right_num = 0
    for batch_data in utils.gen_batchs(full_tensors, args.batch_size, is_shuffle=False):
        softmax_re = sess.run(model.softmax_op,
                              feed_dict={model.ph_dropout_rate: 0,
                                         model.ph_tokens: batch_data[0],
                                         model.ph_labels: batch_data[1],
                                         model.ph_length: batch_data[2],
                                         model.ph_input_mask: batch_data[3]})
        pred_re = np.argmax(softmax_re, axis=1)
        total_num += len(pred_re)
        right_num += np.sum(pred_re == batch_data[1])
        acc = 1.0 * right_num / (total_num + 1e-5)

    tf.logging.info("dev total num: " + str(total_num) + ", right num: " + str(right_num) + ", acc: " + str(acc))
    return acc


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="", help="Input train file.")
    parser.add_argument("--dev_file", type=str, default="", help="Input dev file.")
    parser.add_argument("--vocab_file", type=str, default="", help="Input vocab file.")
    parser.add_argument("--output_id2label_file", type=str, default="./id2label",
                        help="File containing (id, class label) map.")
    parser.add_argument("--model_save_dir", type=str, default="",
                        help="Specified the directory in which the model should stored.")
    parser.add_argument("--lstm_dim", type=int, default=500, help="Dimension of LSTM cell.")
    parser.add_argument("--embedding_dim", type=int, default=1000, help="Dimension of word embedding.")
    parser.add_argument("--opt_type", type=str, default='adam', help="Type of optimizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epoch", type=int, default=20, help="Epoch.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed value.")
    parser.add_argument("--print_step", type=int, default=1000, help="Print log every x step.")
    parser.add_argument("--init_checkpoint", type=str, default='',
                        help="Initial checkpoint (usually from a pre-trained model).")
    parser.add_argument("--max_len", type=int, default=100, help="Max seqence length.")
    parser.add_argument("--layer_num", type=int, default=2, help="LSTM layer num.")

    parser.add_argument("--representation_type", type=str, default="lstm",
                        help="representation type include:lstm, transformer")

    # transformer args
    parser.add_argument("--initializer_range", type=float, default="0.02", help="Embedding initialization range")
    parser.add_argument("--max_position_embeddings", type=int, default=512, help="max position num")
    parser.add_argument("--hidden_size", type=int, default=768, help="hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="num hidden layer")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="num attention heads")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="intermediate_size")

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    tf.logging.info(str(args))
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    tf.logging.info("load training sens")
    train_sens = utils.load_training_data(args.train_file, skip_invalid=True)
    tf.logging.info("\nload dev sens")
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
    tf.logging.info("batch size: " + str(args.batch_size) + ", training sample num : " + str(
        len(train_sens)) + ", print step : " + str(args.print_step))
    tf.logging.info(
        "steps_in_epoch : " + str(steps_in_epoch) + ", epoch num :" + str(args.epoch) + ", total steps : " + str(
            args.epoch * steps_in_epoch))
    print_step = min(args.print_step, steps_in_epoch)
    tf.logging.info("eval dev every {} step".format(print_step))

    save_vars = [v for v in tf.global_variables() if
                 v.name.find('adam') < 0 and v.name.find('Adam') < 0 and v.name.find('ADAM') < 0]
    tf.logging.info(str(save_vars))
    tf.logging.info(str(tf.all_variables()))

    saver = tf.train.Saver(max_to_keep=2)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        total_loss = 0
        dev_best_so_far = 0
        for epoch in range(1, args.epoch + 1):
            tf.logging.info("\n" + "*" * 20 + "epoch num :" + str(epoch) + "*" * 20)
            for batch_data in utils.gen_batchs(train_full_tensors, args.batch_size, is_shuffle=True):
                _, global_step, loss = sess.run([model.train_op, model.global_step_op, model.loss_op],
                                                feed_dict={model.ph_dropout_rate: args.dropout_rate,
                                                           model.ph_tokens: batch_data[0],
                                                           model.ph_labels: batch_data[1],
                                                           model.ph_length: batch_data[2],
                                                           model.ph_input_mask: batch_data[3]})
                total_loss += loss
                if global_step % print_step == 0:
                    tf.logging.info(
                        "\nglobal step : " + str(global_step) + ", avg loss so far : " + str(total_loss / global_step))
                    tf.logging.info("begin to eval dev set: ")
                    acc = evaluate(sess, dev_full_tensors, args, model)
                    if acc > dev_best_so_far:
                        dev_best_so_far = acc
                        tf.logging.info("!" * 20 + "best got : " + str(acc))
                        # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["scores"])
                        saver.save(sess, args.model_save_dir + '/finetune.ckpt', global_step=global_step)

            tf.logging.info("\n----------------------eval after one epoch: ")
            tf.logging.info(
                "global step : " + str(global_step) + ", avg loss so far : " + str(total_loss / global_step))
            tf.logging.info("begin to eval dev set: ")
            sys.stdout.flush()
            acc = evaluate(sess, dev_full_tensors, args, model)
            if acc > dev_best_so_far:
                dev_best_so_far = acc
                tf.logging.info("!" * 20 + "best got : " + str(acc))
                saver.save(sess, args.model_save_dir + '/finetune.ckpt', global_step=global_step)


if __name__ == "__main__":
    tf.app.run()
