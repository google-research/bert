precision=${1:-"fp16"}
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}

if [ "$precision" = "fp16" ] ; then
   echo "fp16 activated!"
   export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=1
else
   echo "fp32 activated!"
   export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0
fi

# Start TRTIS server in detached state
nvidia-docker run -d --rm \
   --shm-size=1g \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   -p8000:8000 \
   -p8001:8001 \
   -p8002:8002 \
   --name trt_server_cont \
   -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
   -e TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE \
   -v $PWD/results/trtis_models:/models \
   nvcr.io/nvidia/tensorrtserver:19.06-py3 trtserver --model-store=/models --strict-model-config=false