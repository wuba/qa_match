#! /usr/bin/env bash
set -xeuo pipefail
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

## run pretrain
# pretrain data is equivalent to train data without label
python run_pretraining.py --train_file="../data_demo/pre_train_data" --vocab_file="../data_demo/vocab" --model_save_dir="./model/pretrain" --batch_size=256 --print_step=10 --weight_decay=0 --embedding_dim=1000 --lstm_dim=500 --layer_num=1 --train_step=10 --warmup_step=1 --learning_rate=5e-5 --dropout_rate=0.1 --max_predictions_per_seq=10 --clip_norm=1.0 --max_seq_len=100 --use_queue=0

## run finetune
# change checkpoint to your result
python run_finetuning.py --output_id2label_file="model/id2label.has_init" --vocab_file="../data_demo/vocab" --train_file="../data_demo/train_data" --dev_file="../data_demo/valid_data" --model_save_dir="model/finetune" --lstm_dim=500 --embedding_dim=1000 --opt_type=adam --batch_size=256 --epoch=1 --learning_rate=1e-4 --opt_type=adam --seed=1 --max_len=100 --print_step=10 --dropout_rate=0.1 --init_checkpoint="model/pretrain/lm_pretrain.ckpt-360" --layer_num=1

## run predict
python run_classifier.py --input_file="../data_demo/test_data" --vocab_file="../data_demo/vocab" --id2label_file="model/id2label.has_init" --model_dir="model/finetune" > "../data_demo/result_pretrain_raw"

## format output
python format_result.py "../data_demo/result_pretrain_raw" "../data_demo/result_pretrain"