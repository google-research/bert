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

train_batch_size=${1:-14}
eval_batch_size=${2:-8}
learning_rate=${3:-"1e-4"}
precision=${4:-"manual_fp16"}
use_xla=${5:-"true"}
num_gpus=${6:-8}
warmup_steps=${7:-"10000"}
train_steps=${8:-1144000}
save_checkpoints_steps=${9:-5000}
bert_model=${10:-"large"}
num_accumulation_steps=${11:-1}
seq_len=${12:-512}
max_pred_per_seq=${13:-80}

DATA_DIR=data/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus

if [ "$bert_model" = "large" ] ; then
    export BERT_CONFIG=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json
else
    export BERT_CONFIG=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_config.json
fi

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

export GBS=$(expr $train_batch_size \* $num_gpus \* $num_accumulation_steps)
printf -v TAG "tf_bert_pretraining_adam_%s_%s_gbs%d" "$bert_model" "$precision" $GBS
DATESTAMP=`date +'%y%m%d%H%M%S'`

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${RESULTS_DIR:-/results/${TAG}_${DATESTAMP}}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

INPUT_FILES="$DATA_DIR/training"
EVAL_FILES="$DATA_DIR/test"

CMD="python3 /workspace/bert/run_pretraining.py"
CMD+=" --input_files_dir=$INPUT_FILES"
CMD+=" --eval_files_dir=$EVAL_FILES"
CMD+=" --output_dir=$RESULTS_DIR"
CMD+=" --bert_config_file=$BERT_CONFIG"
CMD+=" --do_train=True"
CMD+=" --do_eval=True"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --eval_batch_size=$eval_batch_size"
CMD+=" --max_seq_length=$seq_len"
CMD+=" --max_predictions_per_seq=$max_pred_per_seq"
CMD+=" --num_train_steps=$train_steps"
CMD+=" --num_warmup_steps=$warmup_steps"
CMD+=" --num_accumulation_steps=$num_accumulation_steps"
CMD+=" --save_checkpoints_steps=$save_checkpoints_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --optimizer_type=adam"
CMD+=" --horovod $PREC"
CMD+=" --allreduce_post_accumulation=True"

#Check if all necessary files are available before training
for DIR_or_file in $DATA_DIR $BERT_CONFIG $RESULTS_DIR; do
  if [ ! -d "$DIR_or_file" ] && [ ! -f "$DIR_or_file" ]; then
     echo "Error! $DIR_or_file directory missing. Please mount correctly"
     exit -1
  fi
done

if [ $num_gpus -gt 1 ] ; then
   CMD="mpiexec --allow-run-as-root -np $num_gpus --bind-to socket $CMD"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi
set +x
