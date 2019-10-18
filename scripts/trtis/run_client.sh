#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

batch_size=${1:-"8"}
seq_length=${2:-"384"}
doc_stride=${3:-"128"}
trtis_version_name=${4:-"1"}
trtis_model_name=${5:-"bert"}
BERT_DIR=${6:-"data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16"}
squad_version=${7:-"1.1"}

export SQUAD_DIR=data/download/squad/v${squad_version}
if [ "$squad_version" = "1.1" ] ; then
    version_2_with_negative="False"
else
    version_2_with_negative="True"
fi

echo "Squad directory set as " $SQUAD_DIR
if [ ! -d "$SQUAD_DIR" ] ; then
   echo "Error! $SQUAD_DIR directory missing. Please mount SQuAD dataset."
   exit -1
fi

bash scripts/docker/launch.sh \
   "python run_squad_trtis_client.py \
      --trtis_model_name=$trtis_model_name \
      --trtis_model_version=$trtis_version_name \
      --vocab_file=$BERT_DIR/vocab.txt \
      --bert_config_file=$BERT_DIR/bert_config.json \
      --predict_file=$SQUAD_DIR/dev-v${squad_version}.json \
      --predict_batch_size=$batch_size \
      --max_seq_length=${seq_length} \
      --doc_stride=${doc_stride} \
      --output_dir=/results \
      --version_2_with_negative=${version_2_with_negative}"

bash scripts/docker/launch.sh "python $SQUAD_DIR/evaluate-v${squad_version}.py \
    $SQUAD_DIR/dev-v${squad_version}.json /results/predictions.json"
