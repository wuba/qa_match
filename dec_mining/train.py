# -*- coding: utf-8 -*-
"""
finetune on pretrained model with dataset to be clustered
"""

import argparse
import tensorflow as tf
import numpy as np
import dataset
import dec_model
import data_utils


def train(data, model, args):
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=2)
    best_acc = 0
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        train_size = len(data.train_x)
        batch_size = args.batch_size

        steps_in_epoch = train_size // args.batch_size + 1
        tf.logging.info("step by batch " + str(steps_in_epoch))
        z_total = []

        # z: transformed representation of x
        for idx in range(steps_in_epoch):
            if idx == steps_in_epoch - 1:  # last batch
                tf.logging.info("start/ end idx " + str(idx * batch_size) + " " + str(idx * batch_size + batch_size))
                cur_z = sess.run(model.z, feed_dict={
                    model.pretrained_model.ph_tokens: data.train_x[idx * batch_size:],
                    model.pretrained_model.ph_length: data.train_length_x[idx * batch_size:],
                    model.pretrained_model.ph_dropout_rate: 0
                })
            else:
                cur_z = sess.run(model.z, feed_dict={
                    model.pretrained_model.ph_tokens: data.train_x[
                                                      idx * batch_size: idx * batch_size + batch_size],
                    model.pretrained_model.ph_length: data.train_length_x[
                                                      idx * batch_size: idx * batch_size + batch_size],
                    model.pretrained_model.ph_dropout_rate: 0
                })
            z_total.extend(cur_z)

        tf.logging.info("z total size : " + str(len(z_total))) # sample size
        assert len(z_total) == len(data.train_x)
        z = np.array(z_total)

        # Customize the cluster center
        if args.external_cluster_center != "":
            # load external centers file
            external_center = model.external_cluster_center_vec
            assign_mu_op = tf.assign(model.mu, external_center)
        else:
            # kmeans init centers
            assign_mu_op = model.get_assign_cluster_centers_op(z)

        amu = sess.run(assign_mu_op) # get cluster center

        for cur_epoch in range(args.epochs):
            q_list = []

            for idx in range(steps_in_epoch):
                start_idx = idx * batch_size
                end_idx = idx * batch_size + batch_size
                if idx == steps_in_epoch - 1:
                    q_batch = sess.run(
                        model.q, feed_dict={
                            model.pretrained_model.ph_tokens: data.train_x[start_idx:],
                            model.pretrained_model.ph_length: data.train_length_x[start_idx:],
                            model.pretrained_model.ph_dropout_rate: 0
                        })
                else:
                    q_batch = sess.run(
                        model.q, feed_dict={
                            model.pretrained_model.ph_tokens: data.train_x[start_idx: end_idx],
                            model.pretrained_model.ph_length: data.train_length_x[start_idx: end_idx],
                            model.pretrained_model.ph_dropout_rate: 0
                        })

                q_list.extend(q_batch)

            q = np.array(q_list)
            p = model.target_distribution(q)

            for iter_, (batch_x, batch_y, batch_idxs, batch_x_lengths) in enumerate(
                    data.gen_next_batch(batch_size=batch_size, \
                                        is_train_set=True, epoch=1)):
                batch_p = p[batch_idxs]
                _, loss, pred, global_step, lr = sess.run([model.trainer, model.loss, model.pred, model.global_step_op, model.optimizer._lr], \
                                         feed_dict={model.pretrained_model.ph_tokens: batch_x, \
                                                    model.pretrained_model.ph_length: batch_x_lengths, \
                                                    model.p: batch_p, \
                                                    model.pretrained_model.ph_dropout_rate: 0
                                                    })
            # NOTE: acc 只用于有监督数据查看聚类效果，ground truth label不会参与到train，如果是无监督数据，此acc 无用
            acc = model.cluster_acc(batch_y, pred)
            tf.logging.info("[DEC] epoch: {}\tloss: {}\tacc: {}\t lr {} \t global_step {}  ".format(cur_epoch, loss, acc, lr, global_step))
            if acc > best_acc:
                best_acc = acc
                tf.logging.info("!!!!!!!!!!!!! best acc got " + str(best_acc))
            # save model each epoch
            saver.save(sess, args.model_save_dir + '/finetune.ckpt', global_step=global_step)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    # pass params
    parser = argparse.ArgumentParser()
    # sptm pretrain model path
    parser.add_argument("--init_checkpoint", type=str, default='')
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    # customized cluster centers file path, pass either of params 'external_cluster_center' or 'n_clusters'
    parser.add_argument("--external_cluster_center", type=str, default="")
    # number of clusters (init with kmeans)
    parser.add_argument("--n_clusters", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    # DEC model q distribution param, alpha=1 in paper
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--layer_num", type=int, default=1)
    parser.add_argument("--token_num", type=int, default=7820)
    parser.add_argument("--lstm_dim", type=int, default=500)
    parser.add_argument("--embedding_dim", type=int, default=1000)
    parser.add_argument("--vocab_file", type=str, default="./vocab")
    parser.add_argument("--model_save_dir", type=str, default="./saved_model")
    args = parser.parse_args()

    word2id, id2word = data_utils.load_vocab_file(args.vocab_file)
    trainset_size = len(data_utils.load_sentences(args.train_file, skip_invalid=True))
    other_arg_dict = {}
    other_arg_dict['token_num'] = len(word2id)
    other_arg_dict['trainset_size'] = trainset_size

    exp_data = dataset.ExpDataset(args)
    dec_model = dec_model.DEC(args, other_arg_dict)
    train(exp_data, dec_model, args)
