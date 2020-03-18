#!/usr/bin/env bash
#--init_checkpoint=/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12/bert_model.ckpt

python3 run_text_infilling.py \
 --bert_config_file=./config/bert_config.json \
 --vocab_file=./config/vocab.txt \
 --input_file=./sample_text.txt \
 --output_file=./predictions.txt \
 --use_tpu=true \
 --init_checkpoint=$STORAGE_BUCKET/chinese/ \
 --tpu_name=$TPU_NAME
