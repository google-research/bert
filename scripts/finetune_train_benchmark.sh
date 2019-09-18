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

bert_model=${1:-"large"}
use_xla=${2:-"true"}
num_gpu=${3:-"8"}
task=${4:-"squad"}

if [ "$bert_model" = "large" ] ; then
    export BERT_DIR=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
else
    export BERT_DIR=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12
fi

echo  "BERT directory set as " $BERT_DIR

init_checkpoint="$BERT_DIR/bert_model.ckpt"
learning_rate=5e-6

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
echo "Results directory set as " $RESULTS_DIR


if [ "$use_xla" = "true" ] ; then
    use_xla_tag="--use_xla"
else
    use_xla_tag=""
fi

if [ $num_gpu -gt 1 ] ; then
    mpi_command="mpirun -np $num_gpu -H localhost:$num_gpu \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib"
    use_hvd="--horovod"
else
    mpi_command=""
    use_hvd=""
fi

LOGFILE="${RESULTS_DIR}/${task}_training_benchmark_bert_${bert_model}_gpu_${num_gpu}.log"

if [ "$task" = "squad" ] ; then
    export SQUAD_DIR=data/download/squad/v1.1
    epochs="2.0"
    echo "Squad directory set as " $SQUAD_DIR

    echo "Training performance benchmarking for BERT $bert_model from $BERT_DIR" >> $LOGFILE
    echo "Precision $precision" >> $LOGFILE
    echo "Sequence Length   Batch size  Performance(sent/sec)" >> $LOGFILE

    for seq_len in 128 384; do

        if [ "$seq_len" = "128" ] ; then
            doc_stride=64
        else
            doc_stride=128
        fi

        for batch_size in 1 2 4; do
            for precision in fp16 fp32; do
                res_dir=${RESULTS_DIR}/bert_${bert_model}_gpu_${num_gpu}_sl_${seq_len}_prec_${precision}_bs_${batch_size}
                tmp_file="${res_dir}/${task}_training_benchmark.log"

                if [ "$precision" = "fp16" ] ; then
                    echo "fp16 activated!"
                    use_fp16="--use_fp16"
                else
                    echo "fp32 activated!"
                    use_fp16=""
                fi

                $mpi_command python run_squad.py \
                --vocab_file=$BERT_DIR/vocab.txt \
                --bert_config_file=$BERT_DIR/bert_config.json \
                --init_checkpoint=$init_checkpoint \
                --do_train=True \
                --train_file=$SQUAD_DIR/train-v1.1.json \
                --train_batch_size=$batch_size \
                --learning_rate=$learning_rate \
                --num_train_epochs=$epochs \
                --max_seq_length=$seq_len \
                --doc_stride=$doc_stride \
                --output_dir=$res_dir \
                "$use_hvd" \
                "$use_fp16" \
                $use_xla_tag |& tee $tmp_file

                perf=`cat $tmp_file | grep -F 'Training Performance' | awk -F'= ' '{print $2}'`
                echo "$seq_len  $batch_size $perf"

            done
        done
    done

else

    echo "Benchmarking for " $task "currently not supported. Sorry!"

fi