#!/bin/bash

#export NCCL_P2P_LEVEL=4
#export HSA_FORCE_FINE_GRAIN_PCIE=1
#export NCCL_MIN_NRINGS=4
#export NCCL_DEBUG=INFO

#export NCCL_P2P_DISABLE=1
#export NCCL_P2P_LEVEL=0

#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=eth

#export NCCL_SOCKET_IFNAME=enp
#export NCCL_IB_HCA=mlx

SCRIPTPATH=$(dirname $(realpath $0))

# Source parameters
source $SCRIPTPATH/params.sh

# Iterate through configs (Sequence Length, Batch)
for CONFIG in $CONFIGS; do

  IFS=","
  set -- $CONFIG

  SEQ=$1
  BATCH=$2
  MAX_PRED=$(calc_max_pred $SEQ)

  CUR_TRAIN_DIR=$TRAIN_DIR/seq${SEQ}_ba${BATCH}_step$STEPS

  if [ -z $TEST ]; then
    WIKI_TFRECORD_DIR=$DATA_DIR/wiki_tfrecord_seq${SEQ}
  else
    WIKI_TFRECORD_DIR=$DATA_DIR/wiki_tsrecord_seq${SEQ}
  fi

  echo "Model        : $MODEL"           > $CUR_TRAIN_DIR/run_record.txt
  echo "Test         : $TEST"           >> $CUR_TRAIN_DIR/run_record.txt
  echo "Seq/Batch    : $SEQ/$BATCH"     >> $CUR_TRAIN_DIR/run_record.txt
  echo "Max Pred     : $MAX_PRED"       >> $CUR_TRAIN_DIR/run_record.txt
  echo "Steps/Warmup : $STEPS/$WARMUP"  >> $CUR_TRAIN_DIR/run_record.txt
  echo "STARTED on " $(date)            >> $CUR_TRAIN_DIR/run_record.txt

  # run pretraining     -x HOROVOD_AUTOTUNE=1 \
    #   -x NCCL_P2P_LEVEL=4 \
    # -x NCCL_SOCKET_IFNAME=ib \
  mpirun --allow-run-as-root -np $NP \
    -hostfile /data/run/hostfile \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
  python3 $CODE_DIR/run_pretraining.py \
    --input_file=$WIKI_TFRECORD_DIR/*.tfrecord \
    --output_dir=$CUR_TRAIN_DIR \
    --do_train=True \
    --do_eval=True \
    --use_horovod=True \
    --bert_config_file=$TRAIN_DIR/bert_config.json \
    --train_batch_size=$BATCH \
    --max_seq_length=$SEQ \
    --max_predictions_per_seq=$MAX_PRED \
    --num_train_steps=$STEPS \
    --num_warmup_steps=$WARMUP \
    --learning_rate=$LRN_RT \
  2>&1 | tee $CUR_TRAIN_DIR/run_output.txt

  echo "Run time     :" $SECONDS sec >> $CUR_TRAIN_DIR/run_record.txt
  echo "times output :"              >> $CUR_TRAIN_DIR/run_record.txt
  times                              >> $CUR_TRAIN_DIR/run_record.txt
  echo "FINISHED on " $(date)        >> $CUR_TRAIN_DIR/run_record.txt

done
