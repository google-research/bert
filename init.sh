
gcloud auth login
gcloud auth application-default login

sudo apt update
sudo apt install python3
sudo apt install python3-pip
sudo apt install unzip

wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py download_glue_data.py
python3 download_glue_data.py

gsutil cp gs://bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip .
unzip multi_cased_L-12_H-768_A-12.zip

wget https://dl.google.com/cloud_tpu/ctpu/latest/linux/ctpu && chmod a+x ctpu
sudo cp ctpu /usr/bin/

git clone https://github.com/google-research/bert.git
pip3 install -r bert/requirements.txt

gsutil cp multi_cased_L-12_H-768_A-12/* gs://mf-data/bert/

mkdir mf_data
gsutil cp gs://mf-data/bert/tmp/mf/* mf_data/

mkdir mf_data_mini
gsutil cp gs://mf-data/bert/tmp/mf-mini/* mf_data_mini/

