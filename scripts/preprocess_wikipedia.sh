#!/bin/bash

# Check input 
if [ $# -eq 0 ]; then
    echo "Please specify sequence length"
    exit -1
fi

# Determine sequence length and max_predictions_per_seq
SCRIPTPATH=$(dirname $(realpath $0))
source $SCRIPTPATH/params.sh

SEQ=$1
MAX_PRED=$(calc_max_pred $SEQ)
echo "Sequence Length: $SEQ"
echo "Max predictions: $MAX_PRED"
echo "Test           : $TEST"

if [ -z $TEST ]; then
    WIKI_TEXT_DIR=${DATA_DIR}/wiki_text
    WIKI_TFRECORD_DIR=$DATA_DIR/wiki_tfrecord_seq${SEQ}
else
    WIKI_TEXT_DIR=${DATA_DIR}/wiki_test
    WIKI_TFRECORD_DIR=$DATA_DIR/wiki_tsrecord_seq${SEQ}
fi

# Prepare training data folder
if [ -d $WIKI_TFRECORD_DIR ]; then
    echo $WIKI_TFRECORD_DIR already exist. Do nothing.
    exit 0
fi
mkdir -p $WIKI_TFRECORD_DIR

# generate tfrecord of data in parallel
for DIR in ${WIKI_TEXT_DIR}/*; do
    for FILE in $DIR/*; do
        DIR_BASENAME=$(basename $DIR)
        FILE_BASENAME=$(basename $FILE)
        python3 create_pretraining_data.py \
            --input_file=${FILE} \
            --output_file=$WIKI_TFRECORD_DIR/${DIR_BASENAME}--${FILE_BASENAME}.tfrecord \
            --vocab_file=configs/bert_large/vocab.txt \
            --do_lower_case=True \
            --max_seq_length=$SEQ \
            --max_predictions_per_seq=$MAX_PRED \
            --masked_lm_prob=$LM_PROB \
            --random_seed=12345 \
            --dupe_factor=5 &
        sleep 1
    done
done

wait
echo "Done preprocessing wikipedia $SEQ"
