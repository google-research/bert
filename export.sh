#!/usr/bin/env bash
export BERT_BASE_DIR=bert_model/models/chinese_L-12_H-768_A-12
export GLUE_DIR=Bert/data
export MODEL_DIR=Bert/output
export MODEL_PB_DIR=Bert/api/

python export.py \
  --task_name=setiment \
  --do_predict=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --model_dir=$MODEL_DIR/ \
  --serving_model_save_path=$MODEL_PB_DIR



