
export MF_DIR=../mf_data
# export BERT_BASE_DIR=multi_cased_L-12_H-768_A-12
export BERT_BASE_DIR=gs://mf-data/bert
python3 run_classifier.py \
  --task_name=MF \
  --do_train=true \
  --do_eval=true \
  --data_dir=$MF_DIR/mf \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=256 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=gs://mf-tmp/bert/mf_output/ \
  --do_lower_case=False \
  --use_tpu=True --tpu_name=adam

# TODO: when finished, somehow
# ctpu pause

