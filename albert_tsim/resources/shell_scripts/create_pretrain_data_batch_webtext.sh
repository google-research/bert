#!/usr/bin/env bash
echo $1,$2

BERT_BASE_DIR=./bert_config
for((i=$1;i<=$2;i++));
do
python3 create_pretraining_data.py --do_whole_word_mask=True --input_file=gs://raw_text/web_text_zh_raw/web_text_zh_$i.txt \
--output_file=gs://albert_zh/tf_records/tf_web_text_zh_$i.tfrecord --vocab_file=$BERT_BASE_DIR/vocab.txt --do_lower_case=True \
--max_seq_length=512 --max_predictions_per_seq=76 --masked_lm_prob=0.15
done
