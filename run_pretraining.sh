#!/usr/bin/env bash

BERT_BASE_DIR=./config
python3 run_pretraining.py \
 --bert_config_file=$BERT_BASE_DIR/bert_config.json \
 --input_file=$STORAGE_BUCKET/data/tf_news_2016_zh_raw_news2016zh_1.tfrecord \
 --output_dir=$STORAGE_BUCKET/bert_output/ \
 --do_train=true \
 --do_eval=true \
 --use_tpu=true \
 --vocab_file=$BERT_BASE_DIR/vocab.txt \
 --do_lower_case=True \
 --max_seq_length=512 \
 --max_predictions_per_seq=51 \
 --tpu_name=$TPU_NAME