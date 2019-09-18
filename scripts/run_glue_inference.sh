#!/usr/bin/env bash

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

echo "Container nvidia build = " $NVIDIA_BUILD_ID
task_name=${1:-"MRPC"}
init_checkpoint=${2:-"$BERT_DIR/bert_model.ckpt"}
batch_size=${3:-"32"}
precision=${4:-"fp16"}
use_xla=${5:-"true"}
seq_length=${6:-"128"}
doc_stride=${7:-"64"}
bert_model=${8:-"large"}

if [ "$bert_model" = "large" ] ; then
    BERT_DIR=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
else
    BERT_DIR=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12
fi
GLUE_DIR=data/download

echo "GLUE directory set as " $GLUE_DIR " BERT directory set as " $BERT_DIR

use_fp16=""
if [ "$precision" = "fp16" ] ; then
        echo "fp16 activated!"
        use_fp16="--use_fp16"
fi

if [ "$use_xla" = "true" ] ; then
    use_xla_tag="--use_xla"
    echo "XLA activated"
else
    use_xla_tag=""
fi


export GBS=$(expr $batch_size \* $num_gpu)
printf -v TAG "tf_bert_finetuning_glue_%s_inf_%s_%s_gbs%d_ckpt_%s" "$task_name" "$bert_model" "$precision" $GBS "$init_checkpoint"
DATESTAMP=`date +'%y%m%d%H%M%S'`
#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
printf "Logs written to %s\n" "$LOGFILE"

#Check if all necessary files are available before training
for DIR_or_file in $GLUE_DIR $RESULTS_DIR $BERT_DIR/vocab.txt $BERT_DIR/bert_config.json; do
  if [ ! -d "$DIR_or_file" ] && [ ! -f "$DIR_or_file" ]; then
     echo "Error! $DIR_or_file directory missing. Please mount correctly"
     exit -1
  fi
done

$mpi_command python run_classifier.py \
  --task_name=$task_name \
  --predict_batch_size=$batch_size \
  --eval_batch_size=$batch_size \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$task_name \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$init_checkpoint \
  --max_seq_length=$seq_length \
  --doc_stride=$doc_stride \
  --output_dir=$RESULTS_DIR \
  --horovod "$use_fp16" \
  $use_xla_tag |& tee $LOGFILE