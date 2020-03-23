#!/bin/bash

# Note: Unset the following to run with real data
TEST=1

# Leanring rate
LRN_RT=5e-5

# LM Probability
LM_PROB=0.15

# Function to calculate max_perdictions_per_seq
# Input is the seq length
calc_max_pred()
{
  echo $(python3 -c "print(int($1*$LM_PROB/1+1))")
}

# Configurations of the training
MODEL=bert_large
#CONFIGS="128,10 256,4"
CONFIGS="128,40"
#CONFIGS="512,6"
#STEPS=10000000
#WARMUP=10000
STEPS=1600
WARMUP=200

# Horovod
NP=64

# Directories
CODE_DIR=$SCRIPTPATH/..
DATA_DIR=/data/wikipedia
TRAIN_DIR=/data/run_tmp/train_$MODEL
