import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

file_path = os.path.dirname(__file__)


#模型目录
model_dir = os.path.join(file_path, 'albert_lcqmc_checkpoints/')

#config文件
config_name = os.path.join(file_path, 'albert_config/albert_config_tiny.json')
#ckpt文件名称
ckpt_name = os.path.join(model_dir, 'model.ckpt')
#输出文件目录
output_dir = os.path.join(file_path, 'albert_lcqmc_checkpoints/')
#vocab文件目录
vocab_file = os.path.join(file_path, 'albert_config/vocab.txt')
#数据目录
data_dir = os.path.join(file_path, 'data/')

num_train_epochs = 10
batch_size = 128
learning_rate = 0.00005

# gpu使用率
gpu_memory_fraction = 0.8

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 128

# graph名字
graph_file = os.path.join(file_path, 'albert_lcqmc_checkpoints/graph')