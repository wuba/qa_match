import tensorflow as tf
from utils.classifier_utils import TextLoader

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('map_file_path', None, 'standard data path')
flags.DEFINE_string("model_path", None, "checkpoint dir from predicting")
flags.DEFINE_string('test_data_path', None, 'test data path')
flags.DEFINE_string('test_result_path', None, 'test data result path')
flags.DEFINE_integer('batch_size', 100, 'batch_size for train')
flags.DEFINE_integer('seq_length', 40, 'seq_length')
flags.DEFINE_string('label2id_file', None, 'label2id file path')
flags.DEFINE_string('vocab2id_file', None, 'vocab2id_file path')

# load vocab and label mapping
vocab_id = {}
vocab_file = open(FLAGS.vocab2id_file, 'r', encoding='utf-8')
for line in vocab_file:
    word_ids = line.strip().split('\t')
    vocab_id[word_ids[0]] = word_ids[1]
vocab_file.close()
label_id = {}
id_label = {}
label_file = open(FLAGS.label2id_file, 'r', encoding='utf-8')
for line in label_file:
    std_label_ids = line.strip().split('\t')
    label_id[std_label_ids[0]] = std_label_ids[1]
    id_label[std_label_ids[1]] = std_label_ids[0]
# print("id_label: " + str(id_label))

label_file.close()
std_label_map = {}
std_label_map_file = open(FLAGS.map_file_path, 'r', encoding='utf-8')
for line in std_label_map_file:
    tokens = line.strip().split('\t')
    label = tokens[0]
    std_id = tokens[1]
    std_label_map[std_id] = label

std_label_map_file.close()

test_data_loader = TextLoader(False, FLAGS.test_data_path, FLAGS.map_file_path, FLAGS.batch_size, FLAGS.seq_length,
                              vocab_id, label_id, std_label_map, 'utf8', False)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    model_file = tf.train.latest_checkpoint(FLAGS.model_path)
    saver = tf.train.import_meta_graph("{}.meta".format(model_file))
    saver.restore(sess, model_file)
    graph = tf.get_default_graph()
    input_x = graph.get_tensor_by_name("input_x:0")
    length_x = graph.get_tensor_by_name("x_len:0")
    keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
    test_data_loader.reset_batch_pointer()
    prediction = graph.get_tensor_by_name("acc/prediction_softmax:0")  # [batchsize, label_size]
    test_result_file = open(FLAGS.test_result_path, 'w', encoding='utf-8')
    for n in range(test_data_loader.num_batches):
        input_x_test, input_y_test, x_len_test, raw_lines = test_data_loader.next_batch()
        prediction_result = sess.run(prediction,
                                     feed_dict={input_x: input_x_test, length_x: x_len_test, keep_prob: 1.0})
        # print("n: " + str(n))
        # print("len(input_x_test): " + str(len(input_x_test)))
        # print("len(input_y_test): " + str(len(input_y_test)))
        # print("len(raw_lines): " + str(len(raw_lines)))
        assert len(input_y_test) == len(raw_lines)
        for i in range(len(raw_lines)):
            raw_line = raw_lines[i]
            # print("input_y_test[i]: " + str(input_y_test[i]))
            real_label = id_label.get(str(input_y_test[i]))
            label_scores = {}
            for j in range(len(prediction_result[i])):
                label = id_label.get(str(j))
                score = prediction_result[i][j]
                label_scores[label] = score
            sorted_list = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
            # print("real_label: " + str(type(real_label)))
            # print("raw_lines: " + str(raw_lines))
            result_str = str(real_label) + "\t" + str(raw_line) + "\t";
            for label, score in sorted_list:
                result_str = result_str + str(label) + ":" + str(score) + " "
            test_result_file.write(result_str + '\n')
    test_result_file.close()
