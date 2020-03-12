#!/usr/bin/env bash

BERT_BASE_DIR=./albert_config
python3 create_pretraining_data.py --do_whole_word_mask=True --input_file=data/news_zh_1.txt \
--output_file=data/tf_news_2016_zh_raw_news2016zh_1.tfrecord --vocab_file=$BERT_BASE_DIR/vocab.txt --do_lower_case=True \
--max_seq_length=512 --max_predictions_per_seq=51 --masked_lm_prob=0.10