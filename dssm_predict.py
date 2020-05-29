import numpy as np
import tensorflow as tf
import shutil
import os
from utils.match_utils import DataHelper

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('map_file_path', None, 'standard data path')
flags.DEFINE_string("model_path", None, "checkpoint dir from predicting")
flags.DEFINE_string("export_model_dir", None, "export model dir")
flags.DEFINE_string('test_data_path', None, 'test data path')
flags.DEFINE_string('test_result_path', None, 'test data result path')
flags.DEFINE_integer('softmax_r', 45, 'Smooth parameter for osine similarity')  # must be similar as train
flags.DEFINE_integer('batch_size', 100, 'batch_size for train')
flags.DEFINE_string('label2id_file', None, 'label2id file path')
flags.DEFINE_string('vocab2id_file', None, 'vocab2id_file path')

dh = DataHelper(None, None, FLAGS.test_data_path, FLAGS.map_file_path, FLAGS.batch_size, None, FLAGS.label2id_file,
                FLAGS.vocab2id_file, False)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    model_file = tf.train.latest_checkpoint(FLAGS.model_path)
    saver = tf.train.import_meta_graph("{}.meta".format(model_file))
    saver.restore(sess, model_file)
    graph = tf.get_default_graph()
    input_x = graph.get_tensor_by_name("input_x:0")
    length_x = graph.get_tensor_by_name("length_x:0")
    input_y = graph.get_tensor_by_name("input_y:0")
    length_y = graph.get_tensor_by_name("length_y:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    q_y_raw = graph.get_tensor_by_name("representation/q_y_raw:0")
    qs_y_raw = graph.get_tensor_by_name("representation/qs_y_raw:0")
    # first get std tensor value
    length_y_value = [y[0] for y in dh.std_batch]
    input_y_value = [y[1] for y in dh.std_batch]
    # print("input_y_value: " + str(input_y_value))
    # print("input_y_value.shape: " + str(np.array(input_y_value, dtype=np.int32).shape))
    # print("length_y_value.shape: " + str(np.array(length_y_value, dtype=np.int32).shape))
    qs_y_raw_out = sess.run(qs_y_raw, feed_dict={input_y: np.array(input_y_value, dtype=np.int32),
                                                 length_y: np.array(length_y_value, dtype=np.int32), keep_prob: 1.0})

    with tf.name_scope('cosine_similarity_pre'):
        # Cosine similarity
        q_norm_pre = tf.sqrt(tf.reduce_sum(tf.square(q_y_raw), 1, True))  # b*1
        qs_norm_pre = tf.transpose(tf.sqrt(tf.reduce_sum(tf.square(qs_y_raw_out), 1, True)))  # 1*sb
        prod_nu_pre = tf.matmul(q_y_raw, tf.transpose(qs_y_raw_out))  # b*sb
        norm_prod_de = tf.matmul(q_norm_pre, qs_norm_pre)  # b*sb
        cos_sim_pre = tf.truediv(prod_nu_pre, norm_prod_de) * FLAGS.softmax_r  # b*sb

    with tf.name_scope('prob_pre'):
        prob_pre = tf.nn.softmax(cos_sim_pre)  # b*sb

    test_batches = dh.test_batch_iterator()
    test_result_file = open(FLAGS.test_result_path, 'w', encoding='utf-8')
    # print(dh.predict_label_seq)
    for _, test_batch_q in enumerate(test_batches):
        # test_batch_q:[(l1, ws1, label1, line1), (l2, ws2, label2, line2), ...]
        length_x_value = [x[0] for x in test_batch_q]
        input_x_value = [x[1] for x in test_batch_q]
        test_prob = sess.run(prob_pre, feed_dict={input_x: np.array(input_x_value, dtype=np.int32),
                                                  length_x: np.array(length_x_value, dtype=np.int32),
                                                  keep_prob: 1.0})  # b*sb
        # print("test_prob: " + str(test_prob))
        for index, example in enumerate(test_batch_q):
            (_, _, real_label, words) = example
            result_str = str(real_label) + '\t' + str(words) + '\t'
            label_scores = {}
            # print(test_prob[index])
            sample_scores = test_prob[index]
            for score_index, real_label_score in enumerate(sample_scores):
                label_scores[dh.predict_label_seq[score_index]] = real_label_score
            sorted_list = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
            # print(str(sorted_list))
            for label, score in sorted_list:
                result_str = result_str + str(label) + ":" + str(score) + " "
            # write result
            test_result_file.write(result_str + '\n')
    test_result_file.close()
    # export model
    if os.path.isdir(FLAGS.export_model_dir):
        shutil.rmtree(FLAGS.export_model_dir)
    builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.export_model_dir)
    pred_x = tf.saved_model.utils.build_tensor_info(input_x)
    pred_len_x = tf.saved_model.utils.build_tensor_info(length_x)
    drop_keep_prob = tf.saved_model.utils.build_tensor_info(keep_prob)
    probs = tf.saved_model.utils.build_tensor_info(prob_pre)
    # 定义方法名和输入输出
    signature_def_map = {
        "predict": tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"input": pred_x, "length": pred_len_x, "keep_prob": drop_keep_prob},
            outputs={
                "probs": probs
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    }
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map=signature_def_map)
    builder.save()
