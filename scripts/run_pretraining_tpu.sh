TPU_NAME=$1

python ../run_pretraining.py --bert_config_file=../config.json \
    --input_file=gs://bert-pretraining-v3/google_bert/*.tfrecord \
    --output_dir=gs://bert-pretraining-v3/checkpoints \
    --do_train=True \
    --do_eval=True \
    --train_batch_size=32 \
    --eval_batch_size=64 \
    --learning_rate=5e-5 \
    --num_train_steps=500000 \
    --num_warmup_steps=30000 \
    --use_tpu=True \
    --tpu_name=$TPU_NAME \
    --tpu_zone=europe-west4-a \
    --gcp_project=pingpong-joohong-tfrc \
    --num_tpu_cores=1
