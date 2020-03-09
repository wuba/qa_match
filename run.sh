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
