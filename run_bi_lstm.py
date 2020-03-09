# coding=utf-8

"""
running bi-lstm for short text classification

"""

import os
import tensorflow as tf
from classifier_utils import TextLoader
from bilstm import BiLSTM

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_path", None, "dir for train data")
flags.DEFINE_string("test_path", None, "dir for test data")
flags.DEFINE_string("model_path", None, "dir for save checkpoint data")
flags.DEFINE_string("result_file", None, "file for result file")
flags.DEFINE_integer("embedding_size", 256, "size of word embedding")
flags.DEFINE_integer("num_units", 256, "The number of units in the LSTM cell")
flags.DEFINE_integer("vocab_size", 256, "The size of vocab")
flags.DEFINE_integer("label_size", 256, "The size of vocab")
flags.DEFINE_integer("batch_size", 128, "batch_size of train data")
flags.DEFINE_integer("seq_length", 50, "the length of sequence")
flags.DEFINE_integer("num_epcho", 30, "the epcho num")
flags.DEFINE_integer("check_every", 100, "the epcho num")
flags.DEFINE_integer("lstm_layers", 2, "the layers of lstm")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_float("dropout_keep_prob", 0.8, "drop_out keep prob")

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    data_loader = TextLoader(True, FLAGS.train_path, FLAGS.batch_size, FLAGS.seq_length, None, None, 'utf8', False)
    test_data_loader = TextLoader(False, FLAGS.test_path, FLAGS.batch_size, FLAGS.seq_length, data_loader.vocab, data_loader.labels, 'utf8', False)
    tf.logging.info("vocab_size: " + str(data_loader.vocab_size))
    FLAGS.vocab_size = data_loader.vocab_size
    tf.logging.info("label_size: " + str(data_loader.label_size))
    FLAGS.label_size = data_loader.label_size
    bilstm = BiLSTM(FLAGS)
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        idx = 0
        test_best_acc = 0
        for epcho in range(FLAGS.num_epcho):  #for each epoch
            data_loader.reset_batch_pointer()
            for train_batch_num in range(data_loader.num_batches):  #for each batch
                input_x, input_y, x_len = data_loader.next_batch()
                feed = {bilstm.input_x:input_x, bilstm.input_y:input_y, bilstm.x_len:x_len, bilstm.dropout_keep_prob:FLAGS.dropout_keep_prob}
                _, global_step_op, train_loss, train_acc = sess.run(
                    [bilstm.train_step, bilstm.global_step, bilstm.loss, bilstm.acc], feed_dict=feed)
                tf.logging.info("training...........global_step = {}, epoch = {}, current_batch = {}, "
                                "train_loss = {:.4f}, accuracy = {:.4f}".format(global_step_op, epcho, train_batch_num, train_loss, train_acc))
                idx += 1
                if idx % FLAGS.check_every == 0:
                    test_acc = 0
                    all_num = 0
                    acc_num = 0
                    test_data_loader.reset_batch_pointer()
                    write_result = []
                    for _ in range(test_data_loader.num_batches):
                        input_x_test, input_y_test, x_len_test = test_data_loader.next_batch()
                        feed = {bilstm.input_x: input_x_test, bilstm.input_y: input_y_test, bilstm.x_len: x_len_test, bilstm.dropout_keep_prob: 1.0}
                        prediction, arg_index = sess.run([bilstm.prediction, bilstm.arg_index], feed_dict=feed)
                        all_num = all_num + len(input_y_test)
                        write_str = ""
                        for i, indexs in enumerate(arg_index):
                            pre_label_id = indexs[0]
                            real_label_id = input_y_test[i]
                            if pre_label_id == real_label_id:
                                acc_num = acc_num + 1
                            if real_label_id in test_data_loader.id_2_label:
                                write_str = test_data_loader.id_2_label.get(real_label_id)
                            else:
                                write_str = "__label__unknown"
                            for index in indexs:
                                cur_label = test_data_loader.id_2_label.get(index)
                                cur_score = prediction[i][index]
                                write_str = write_str + " " + cur_label + ":" + str(cur_score)
                            write_str = write_str + "\n"
                            write_result.append(write_str)
                    test_acc = acc_num * 1.0 / all_num
                    tf.logging.info("testing...........global_step = {}, epoch = {}, accuracy = {:.4f}, cur_best_acc = {}".format(global_step_op, epcho, test_acc, test_best_acc))
                    if test_best_acc < test_acc:
                        test_best_acc = test_acc
                        # save_model
                        checkpoint_path = os.path.join(FLAGS.model_path, 'lstm.ckpt')
                        saver.save(sess, checkpoint_path, global_step=global_step_op)
                        resultfile = open(FLAGS.result_file, 'w', encoding='utf-8')
                        for pre_sen in write_result:
                            resultfile.write(pre_sen)
                        tf.logging.info("has saved model and write.result...................................................................")
                        resultfile.close()


if __name__ == "__main__":
    tf.app.run()
