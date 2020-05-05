cd /workspace/

#1.Category train
python run_bi_lstm.py  --train_path=./data_demo/train_data --valid_path=./data_demo/valid_data --map_file_path=./data_demo/std_data --result_file=./data_demo/result_max_valid --model_path=./model/model_max --vocab_file=./model/model_max/vocab_max --label_file=./model/model_max/label_max --embedding_size=256 --num_units=256 --batch_size=200 --seq_length=40 --num_epcho=30 --check_every=20 --lstm_layers=2 --lr=0.01 --dropout_keep_prob=0.8
#2.Category predict
python lstm_predict.py --map_file_path=./data_demo/std_data --model_path=./model/model_max --test_data_path=./data_demo/test_data --test_result_path=./model/model_max/result_max_test --batch_size=250 --seq_length=40 --label2id_file=./model/model_max/label_max --vocab2id_file=./model/model_max/vocab_max

#3.Intent recognition train
python run_dssm.py --train_path=./data_demo/train_data --valid_path=./data_demo/valid_data --map_file_path=./data_demo/std_data --model_path=./model/model_min/ --result_file_path=./data/result_min --softmax_r=45 --embedding_size=256 --learning_rate=0.001 --keep_prob=0.8 --batch_size=250 --num_epoches=30 --negative_size=200 --eval_every=10 --num_units=256 --use_same_cell=False --label2id_path=./model/model_min/min_label2id --vocab2id_path=./model/model_min/min_vocab2id
#4.Intent recognition predict
python dssm_predict.py --map_file_path=./data_demo/std_data --model_path=./model/model_min/ --export_model_dir=./model/model_min/dssm_tf_serving/ --test_data_path=./data_demo/test_data --test_result_path=./model/model_min/result_min_test --softmax_r=45 --batch_size=250 --label2id_file=./model/model_min/min_label2id --vocab2id_file=./model/model_min/min_vocab2id

#5.1merge and evaluating for 2 level knowledge base
python merge_classifier_match_label.py ./model/model_max/result_max_test ./model/model_min/result_min_test ./data_demo/merge_result_2_level ./data_demo/std_data

#5.2merge and evaluating for 1 level knowledge base
python merge_classifier_match_label.py no ./model/model_min/result_min_test ./data_demo/merge_result_1_level no










set -x

#Category
python run_bi_lstm.py  --train_path=./data/max.train  \
--test_path=./data/max.test \
--result_file=./data/result_max  \
--model_path=./model_max \
--embedding_size=256 \
--num_units=256 \
--batch_size=100 \
--seq_length=40 \
--num_epcho=50 \
--check_every=20 \
--lstm_layers=2 \
--lr=0.01 \
--dropout_keep_prob=0.8

#Intent recognition
python run_dssm.py --std_data_path=./data/standard \
--train_data_path=./data/min.train \
--test_data_path=./data/min.test \
--valid_data_path=./data/min.valid \
--model_path=./model_min/ \
--result_file_path=./data/result_min \
--softmax_r=45 \
--embedding_size=256 \
--learning_rate=0.001 \
--keep_prob=0.8 \
--batch_size=150 \
--num_epoches=30 \
--negative_size=250 \
--eval_every=10 \
--num_units=256 \
--use_same_cell=False

#merge
python merge_classifier_match_label.py \
./data/max_min.map \
./data/result_max \
./data/result_min \
./data/result_max_min_merge
>>>>>>> Stashed changes
