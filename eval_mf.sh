export MF_DIR=../mf_data
export BERT_BASE_DIR=gs://mf-data/bert
python3 run_classifier.py \
  --task_name=MF \
  --do_eval=true \
  --data_dir=$MF_DIR/mf \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=gs://mf-tmp/bert/mf_output/ \
  --do_lower_case=False \
  --use_tpu=True --tpu_name=adam

