#!/bin/bash
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit

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

set -eux

readonly docker_image="nvcr.io/nvidia/tensorflow:19.08-py3"
readonly datadir="/raid/data/bert"
readonly checkpointdir="$PWD/checkpoints"

readonly mounts=".:/workspace/bert,${datadir}:/workspace/bert/data,${checkpointdir}:/results"


srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 mkdir -p "${checkpointdir}/phase_1"
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 mkdir -p "${checkpointdir}/phase_2"

PHASE1="\
     --train_batch_size=${BATCHSIZE:-16} \
     --learning_rate=${LEARNING_RATE:-1.875e-4} \
     --num_accumulation_steps=${NUM_ACCUMULATION_STEPS:-128} \
     --input_files_dir=/workspace/bert/data/tfrecord/lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training \
     --eval_files_dir=/workspace/bert/data/tfrecord/lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/test \
     --max_seq_length=128 \
     --max_predictions_per_seq=20 \
     --num_train_steps=7038 \
     --num_warmup_steps=2000 \
     --output_dir=/results/phase_1 \
     "

PHASE2="\
     --train_batch_size=${BATCHSIZE:-2} \
     --learning_rate=${LEARNING_RATE:-1.25e-4} \
     --num_accumulation_steps=${NUM_ACCUMULATION_STEPS:-512} \
     --input_files_dir=/workspace/bert/data/tfrecord/lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training \
     --eval_files_dir=/workspace/bert/data/tfrecord/lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/test \
     --max_seq_length=512 \
     --max_predictions_per_seq=80 \
     --num_train_steps=1564 \
     --num_warmup_steps=200 \
     --output_dir=/results/phase_2 \
     --init_checkpoint=/results/phase_1/model.ckpt-7038 \
    "

PHASES=( "$PHASE1" "$PHASE2" )

PHASE=${PHASE:-1}

BERT_CMD="\
    python /workspace/bert/run_pretraining.py \
     ${PHASES[$((PHASE-1))]} \
     --bert_config_file=/workspace/bert/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json \
     --do_train=True \
     --do_eval=True \
     --save_checkpoints_steps=100 \
     --horovod --use_fp16 --use_xla \
     --allreduce_post_accumulation=True \
     --eval_batch_size=8"

srun --mpi=pmi2 -l --container-image="${docker_image}" --container-mounts="${mounts}" bash -c "${BERT_CMD}"
