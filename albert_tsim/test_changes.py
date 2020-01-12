# coding=utf-8
import tensorflow as tf
from modeling import embedding_lookup_factorized,transformer_model
import os

"""
测试albert主要的改进点：词嵌入的因式分解、层间参数共享、段落间连贯性
test main change of albert from bert
"""
batch_size = 2048
sequence_length = 512
vocab_size = 30000
hidden_size = 1024
num_attention_heads = int(hidden_size / 64)

def get_total_parameters():
    """
    get total parameters of a graph
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    return total_parameters

def test_factorized_embedding():
    """
    test of Factorized embedding parameterization
    :return:
    """
    input_ids=tf.zeros((batch_size, sequence_length),dtype=tf.int32)
    output, embedding_table, embedding_table_2=embedding_lookup_factorized(input_ids,vocab_size,hidden_size)
    print("output:",output)

def test_share_parameters():
    """
    test of share parameters across all layers: how many parameter after share parameter across layers of transformer.
    :return:
    """
    def total_parameters_transformer(share_parameter_across_layers):
        input_tensor=tf.zeros((batch_size, sequence_length, hidden_size),dtype=tf.float32)
        print("transformer_model. input:",input_tensor)
        transformer_result=transformer_model(input_tensor,hidden_size=hidden_size,num_attention_heads=num_attention_heads,share_parameter_across_layers=share_parameter_across_layers)
        print("transformer_result:",transformer_result)
        total_parameters=get_total_parameters()
        print('total_parameters(not share):',total_parameters)

    share_parameter_across_layers=False
    total_parameters_transformer(share_parameter_across_layers) # total parameters, not share: 125,976,576 = 125 million

    tf.reset_default_graph() # Clears the default graph stack and resets the global default graph
    share_parameter_across_layers=True
    total_parameters_transformer(share_parameter_across_layers) #  total parameters,   share: 10,498,048 = 10.5 million

def test_sentence_order_prediction():
    """
    sentence order prediction.

    check method of create_instances_from_document_albert from create_pretrining_data.py

    :return:
    """
    # 添加运行权限
    os.system("chmod +x create_pretrain_data.sh")

    os.system("./create_pretrain_data.sh")


# 1.test of Factorized embedding parameterization
#test_factorized_embedding()

# 2. test of share parameters across all layers: how many parameter after share parameter across layers of transformer.
# before share parameter: 125,976,576; after share parameter:
#test_share_parameters()

# 3. test of sentence order prediction(SOP)
test_sentence_order_prediction()

