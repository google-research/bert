#!/usr/bin/env bash

##Following commands to execute in Google Console

# Create Tensorflow Serving Container and host on Dockerhub
IMAGE_NAME=tf_serving_bert_agnews
VER=1547919083
MODEL_NAME=bert
DOCKER_USER=lapolonio
cd ~
docker run -d --name $IMAGE_NAME tensorflow/serving
mkdir ~/models
gsutil cp -r  gs://bert-finetuning-ag-news/bert/export/AGNE/1547919083 ~/models
docker cp ~/models/1547919083 $IMAGE_NAME:/models/$MODEL_NAME
docker commit --change "ENV MODEL_NAME $MODEL_NAME" $IMAGE_NAME $USER/$IMAGE_NAME
docker tag $USER/$IMAGE_NAME $DOCKER_USER/$IMAGE_NAME:$VER
docker push $DOCKER_USER/$IMAGE_NAME:$VER

# Create client to call Bert Model
CLIENT_IMAGE_NAME=bert_agnews_client
CLIENT_VER=v1
git clone https://github.com/lapolonio/bert.git
cd ~/bert
mkdir asset
gsutil cp gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/vocab.txt asset/
docker build -t $USER/$CLIENT_IMAGE_NAME .
docker tag $USER/$CLIENT_IMAGE_NAME $DOCKER_USER/$CLIENT_IMAGE_NAME:$CLIENT_VER
docker push $DOCKER_USER/$CLIENT_IMAGE_NAME:$CLIENT_VER
