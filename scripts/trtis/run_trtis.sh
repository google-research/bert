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

init_checkpoint=${1:-"/results/models/bert_large_fp16_384_v1/model.ckpt-5474"}
batch_size=${2:-"8"}
precision=${3:-"fp16"}
use_xla=${4:-"true"}
seq_length=${5:-"384"}
doc_stride=${6:-"128"}
bert_model=${7:-"large"}
squad_version=${8:-"1.1"}
trtis_version_name=${9:-1}
trtis_model_name=${10:-"bert"}
trtis_export_model=${11:-"true"}
trtis_dyn_batching_delay=${12:-0}
trtis_engine_count=${13:-1}
trtis_model_overwrite=${14:-"False"}

if [ "$bert_model" = "large" ] ; then
    export BERT_DIR=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
else
    export BERT_DIR=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12
fi

if [ ! -d "$BERT_DIR" ] ; then
   echo "Error! $BERT_DIR directory missing. Please mount pretrained BERT dataset."
   exit -1
fi

# Need to ignore case on some variables
trtis_export_model=$(echo "$trtis_export_model" | tr '[:upper:]' '[:lower:]')

# Explicitly save this variable to pass down to new containers
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}

echo " BERT directory set as " $BERT_DIR
echo
echo "Argument: "
echo "   init_checkpoint = $init_checkpoint"
echo "   batch_size      = $batch_size"
echo "   precision       = $precision"
echo "   use_xla         = $use_xla"
echo "   seq_length      = $seq_length"
echo "   doc_stride      = $doc_stride"
echo "   bert_model      = $bert_model"
echo "   squad_version   = $squad_version"
echo "   version_name    = $trtis_version_name"
echo "   model_name      = $trtis_model_name"
echo "   export_model    = $trtis_export_model"
echo
echo "Env: "
echo "   NVIDIA_VISIBLE_DEVICES = $NV_VISIBLE_DEVICES"
echo

# Export Model in SavedModel format if enabled
if [ "$trtis_export_model" = "true" ] ; then
   echo "Exporting model as: Name - $trtis_model_name Version - $trtis_version_name"

      bash scripts/trtis/export_model.sh $init_checkpoint $batch_size $precision $use_xla $seq_length \
         $doc_stride $BERT_DIR $RESULTS_DIR $trtis_version_name $trtis_model_name \
         $trtis_dyn_batching_delay $trtis_engine_count $trtis_model_overwrite
fi

# Start TRTIS server in detached state
bash scripts/docker/launch_server.sh $precision

# Wait until server is up. curl on the health of the server and sleep until its ready
bash scripts/trtis/wait_for_trtis_server.sh localhost

# Start TRTIS client for inference and evaluate results
bash scripts/trtis/run_client.sh $batch_size $seq_length $doc_stride $trtis_version_name $trtis_model_name \
    $BERT_DIR $squad_version


#Kill the TRTIS Server
docker kill trt_server_cont
