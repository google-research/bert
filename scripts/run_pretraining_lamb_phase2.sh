#! /bin/bash

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

train_batch_size_phase1=${1:-64}
train_batch_size_phase2=${2:-8}
eval_batch_size=${3:-8}
learning_rate_phase1=${4:-"7.5e-4"}
learning_rate_phase2=${5:-"5e-4"}
precision=${6:-"fp16"}
use_xla=${7:-"true"}
num_gpus=${8:-2}
warmup_steps_phase1=${9:-"2000"}
warmup_steps_phase2=${10:-"200"}
train_steps=${11:-7820}
save_checkpoints_steps=${12:-100}
num_accumulation_steps_phase1=${13:-128}
num_accumulation_steps_phase2=${14:-512}
bert_model=${15:-"large"}

DATA_DIR=${DATA_DIR:-data}
#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${RESULTS_DIR:-/results}

if [ "$bert_model" = "large" ] ; then
    export BERT_CONFIG=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json
else
    export BERT_CONFIG=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_config.json
fi

echo "Container nvidia build = " $NVIDIA_BUILD_ID

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--use_fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "manual_fp16" ] ; then
   PREC="--manual_fp16"
else
   echo "Unknown <precision> argument"
   exit -2
fi

if [ "$use_xla" = "true" ] ; then
    PREC="$PREC --use_xla"
    echo "XLA activated"
fi

mpi=""
if [ $num_gpus -gt 1 ] ; then
   mpi="mpiexec --allow-run-as-root -np $num_gpus --bind-to socket"
fi

#PHASE 1 Config

train_steps_phase1=$(expr $train_steps \* 9 \/ 10) #Phase 1 is 10% of training
gbs_phase1=$(expr $train_batch_size_phase1 \* $num_accumulation_steps_phase1)
PHASE1_CKPT=${RESULTS_DIR}/phase_1/model.ckpt-${train_steps_phase1}

#PHASE 2

seq_len=512
max_pred_per_seq=80
train_steps_phase2=$(expr $train_steps \* 1 \/ 10) #Phase 2 is 10% of training
gbs_phase2=$(expr $train_batch_size_phase2 \* $num_accumulation_steps_phase2)
train_steps_phase2=$(expr $train_steps_phase2 \* $gbs_phase1 \/ $gbs_phase2) # Adjust for batch size

RESULTS_DIR_PHASE2=${RESULTS_DIR}/phase_2
mkdir -m 777 -p $RESULTS_DIR_PHASE2

INPUT_FILES="$DATA_DIR/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training"
EVAL_FILES="$DATA_DIR/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/test"

#Check if all necessary files are available before training
for DIR_or_file in $DATA_DIR $RESULTS_DIR $BERT_CONFIG ${PHASE1_CKPT}.meta; do
  if [ ! -d "$DIR_or_file" ] && [ ! -f "$DIR_or_file" ]; then
     echo "Error! $DIR_or_file directory missing. Please mount correctly"
     exit -1
  fi
done

$mpi python /workspace/bert/run_pretraining.py \
    --input_files_dir=$INPUT_FILES \
    --init_checkpoint=$PHASE1_CKPT \
    --eval_files_dir=$EVAL_FILES \
    --output_dir=$RESULTS_DIR_PHASE2 \
    --bert_config_file=$BERT_CONFIG \
    --do_train=True \
    --do_eval=True \
    --train_batch_size=$train_batch_size_phase2 \
    --eval_batch_size=$eval_batch_size \
    --max_seq_length=$seq_len \
    --max_predictions_per_seq=$max_pred_per_seq \
    --num_train_steps=$train_steps_phase2 \
    --num_accumulation_steps=$num_accumulation_steps_phase2 \
    --num_warmup_steps=$warmup_steps_phase2 \
    --save_checkpoints_steps=$save_checkpoints_steps \
    --learning_rate=$learning_rate_phase2 \
    --horovod $PREC \
    --allreduce_post_accumulation=True

