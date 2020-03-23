#!/bin/bash

SCRIPTPATH=$(dirname $(realpath $0))

# Source parameters
source $SCRIPTPATH/params.sh

MODEL_CONFIG_DIR=$CODE_DIR/configs/$MODEL

#rm -rf $TRAIN_DIR
mkdir -p $TRAIN_DIR

# Prep the train dir
cp $MODEL_CONFIG_DIR/vocab.txt $TRAIN_DIR/vocab.txt
cp $MODEL_CONFIG_DIR/bert_config.json $TRAIN_DIR/bert_config.json

# Iterate through configs (Sequence Length, Batch)
for CONFIG in $CONFIGS; do

  IFS=","
  set -- $CONFIG

  SEQ=$1
  BATCH=$2

  CUR_TRAIN_DIR=$TRAIN_DIR/seq${SEQ}_ba${BATCH}_step$STEPS
  rm -rf $CUR_TRAIN_DIR
  mkdir -p $CUR_TRAIN_DIR

done
