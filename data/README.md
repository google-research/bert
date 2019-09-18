Steps to reproduce datasets from web

1) Build the container
  * docker build -t bert_tf .
2) Run the container interactively
  * nvidia-docker run -it --ipc=host bert_tf
  * Optional: Mount data volumes
    * -v yourpath:/workspace/bert/data/wikipedia_corpus/download
    * -v yourpath:/workspace/bert/data/wikipedia_corpus/extracted_articles
    * -v yourpath:/workspace/bert/data/wikipedia_corpus/raw_data
    * -v yourpath:/workspace/bert/data/wikipedia_corpus/intermediate_files
    * -v yourpath:/workspace/bert/data/wikipedia_corpus/final_text_file_single
    * -v yourpath:/workspace/bert/data/wikipedia_corpus/final_text_files_sharded
    * -v yourpath:/workspace/bert/data/wikipedia_corpus/final_tfrecords_sharded
    * -v yourpath:/workspace/bert/data/bookcorpus/download
    * -v yourpath:/workspace/bert/data/bookcorpus/final_text_file_single
    * -v yourpath:/workspace/bert/data/bookcorpus/final_text_files_sharded
    * -v yourpath:/workspace/bert/data/bookcorpus/final_tfrecords_sharded
  * Optional: Select visible GPUs
    * -e CUDA_VISIBLE_DEVICES=0

** Inside of the container starting here**
3) Download pretrained weights (they contain vocab files for preprocessing)
  * cd data/pretrained_models_google && python3 download_models.py
4) "One-click" SQuAD download
  * cd /workspace/bert/data/squad && . squad_download.sh
5) "One-click" Wikipedia data download and prep (provides tfrecords)
  * Set your configuration in data/wikipedia_corpus/config.sh
  * cd /data/wikipedia_corpus && ./run_preprocessing.sh
6) "One-click" BookCorpus data download and prep (provided tfrecords)
  * Set your configuration in data/wikipedia_corpus/config.sh
  * cd /data/bookcorpus && ./run_preprocessing.sh
