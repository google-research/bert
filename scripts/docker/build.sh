#!/bin/bash

docker pull nvcr.io/nvidia/tensorrtserver:19.06-py3

#The follow has been commented out since we need fixes for the perf_client from Guan
#Uncomment to enable building. 
#For now, the tensorrt_client can be downloaded from https://drive.google.com/drive/u/1/folders/1CeOMZbnFT1VUIlIMoDEZJb3kOKbXBDbZ
git submodule update --init --recursive && cd tensorrt-inference-server && docker build -t tensorrtserver_client -f Dockerfile.client . && cd -

docker build . --rm -t bert
