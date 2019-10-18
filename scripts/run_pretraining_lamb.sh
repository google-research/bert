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

train_batch_size_phase1=${1:-64}
train_batch_size_phase2=${2:-8}
eval_batch_size=${3:-8}
learning_rate_phase1=${4:-"7.5e-4"}
learning_rate_phase2=${5:-"5e-4"}
precision=${6:-"fp16"}
use_xla=${7:-"true"}
num_gpus=${8:-8}
warmup_steps_phase1=${9:-"2000"}
warmup_steps_phase2=${10:-"200"}
train_steps=${11:-7820}
save_checkpoints_steps=${12:-100}
num_accumulation_steps_phase1=${13:-128}
num_accumulation_steps_phase2=${14:-512}
bert_model=${15:-"large"}

DATA_DIR=data
export DATA_DIR=$DATA_DIR

GBS1=$(expr $train_batch_size_phase1 \* $num_gpus \* $num_accumulation_steps_phase1)
GBS2=$(expr $train_batch_size_phase2 \* $num_gpus \* $num_accumulation_steps_phase2)
printf -v TAG "tf_bert_pretraining_lamb_%s_%s_gbs1%d_gbs2%d" "$bert_model" "$precision" $GBS1 $GBS2
DATESTAMP=`date +'%y%m%d%H%M%S'`

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${RESULTS_DIR:-/results/${TAG}_${DATESTAMP}}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"
export RESULTS_DIR=$RESULTS_DIR

printf -v SCRIPT_ARGS "%d %d %d %e %e %s %s %d %d %d %d %d %d %d %s %s" \
                      $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 \
                      $learning_rate_phase2 "$precision" "$use_xla" $num_gpus $warmup_steps_phase1 \
                      $warmup_steps_phase2 $train_steps $save_checkpoints_steps \
                      $num_accumulation_steps_phase1 $num_accumulation_steps_phase2 "$bert_model"

# RUN PHASE 1
bash scripts/run_pretraining_lamb_phase1.sh $SCRIPT_ARGS |& tee -a $LOGFILE

# RUN PHASE 2
bash scripts/run_pretraining_lamb_phase2.sh $SCRIPT_ARGS |& tee -a $LOGFILE
