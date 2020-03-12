#!/usr/bin/env bash
# @Author: bo.shi, https://github.com/chineseGLUE/chineseGLUE
# @Date:   2019-11-04 09:56:36
# @Last Modified by:   bright
# @Last Modified time: 2019-11-10 09:00:00

TASK_NAME="lcqmc"
MODEL_NAME="albert_tiny_zh"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

export CUDA_VISIBLE_DEVICES="0"
export ALBERT_CONFIG_DIR=$CURRENT_DIR/albert_config
export ALBERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
export ALBERT_TINY_DIR=$ALBERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
#mkdir chineseGLUEdatasets
export GLUE_DATA_DIR=$CURRENT_DIR/chineseGLUEdatasets

# download and unzip dataset
if [ ! -d $GLUE_DATA_DIR ]; then
  mkdir -p $GLUE_DATA_DIR
  echo "makedir $GLUE_DATA_DIR"
fi
cd $GLUE_DATA_DIR
if [ ! -d $TASK_NAME ]; then
  mkdir $TASK_NAME
  echo "makedir $GLUE_DATA_DIR/$TASK_NAME"
fi
cd $TASK_NAME
echo "Please try again if the data is not downloaded successfully."
wget -c https://raw.githubusercontent.com/pengming617/text_matching/master/data/train.txt
wget -c https://raw.githubusercontent.com/pengming617/text_matching/master/data/dev.txt
wget -c https://raw.githubusercontent.com/pengming617/text_matching/master/data/test.txt
echo "Finish download dataset."

# download model
if [ ! -d $ALBERT_TINY_DIR ]; then
  mkdir -p $ALBERT_TINY_DIR
  echo "makedir $ALBERT_TINY_DIR"
fi
cd $ALBERT_TINY_DIR
if [ ! -f "albert_config_tiny.json" ] || [ ! -f "vocab.txt" ] || [ ! -f "checkpoint" ] || [ ! -f "albert_model.ckpt.index" ] || [ ! -f "albert_model.ckpt.meta" ] || [ ! -f "albert_model.ckpt.data-00000-of-00001" ]; then
  rm *
  wget https://storage.googleapis.com/albert_zh/albert_tiny_489k.zip
  unzip albert_tiny_489k.zip
  rm albert_tiny_489k.zip
else
  echo "model exists"
fi
echo "Finish download model."

# run task
cd $CURRENT_DIR
echo "Start running..."
python run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DATA_DIR/$TASK_NAME \
  --vocab_file=$ALBERT_CONFIG_DIR/vocab.txt \
  --bert_config_file=$ALBERT_CONFIG_DIR/albert_config_tiny.json \
  --init_checkpoint=$ALBERT_TINY_DIR/albert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=64 \
  --learning_rate=1e-4 \
  --num_train_epochs=5.0 \
  --output_dir=$CURRENT_DIR/${TASK_NAME}_output/
