#!/usr/bin/env bash
# -*- coding: utf-8 -*- 



# Author: xiaoy li 
# description:
# 


if [[ $1 == "tpu" ]]; then
    REPO_PATH=/home/xiaoyli1110/bert
    export TPU_NAME=xiaoya-tpu
    export PYTHONPATH="$PYTHONPATH:/home/xiaoyli1110/bert"
    DATA_DIR=gs://xiaoy-data/data 
    SQUAD_DIR=${DATA_DIR}/squad2
    BERT_DIR=${DATA_DIR}/cased_L-12_H-768_A-12
    OUTPUT_DIR=gs://corefqa-output/2020-06-10/spanbert-base 

    python3 ${REPO_PATH}/run_squad.py \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --init_checkpoint=$BERT_DIR/bert_model.ckpt \
    --do_train=True \
    --train_file=$SQUAD_DIR/train-v2.0.json \
    --do_predict=True \
    --predict_file=$SQUAD_DIR/dev-v2.0.json \
    --train_batch_size=24 \
    --learning_rate=3e-5 \
    --num_train_epochs=2.0 \
    --max_seq_length=384 \
    --doc_stride=128 \
    --output_dir=${OUTPUT_DIR} \
    --use_tpu=True \
    --tpu_name=$TPU_NAME \
    --version_2_with_negative=True

elif [[ $1 == "gpu" ]]; then 
    REPO_PATH=/home/lixiaoya/bert
    export PYTHONPATH="$PYTHONPATH:/home/lixiaoya/bert"
    DATA_DIR=/xiaoya/origin_data
    export CUDA_VISIBLE_DEVICES=0,1
fi




