"""
Created on @Time:2019/12/27 14:06
@Author:sliderSun 
@FileName: predict.py
"""
import tensorflow as tf

import args
import tokenization
from run_classifier import convert_examples_to_features
from similarity import SimProcessor

sess = tf.Session()
with tf.gfile.GFile(
        'F:\python_work\github\\albert_zh\\albert_lcqmc_checkpoints_base\\albert.pb',
        'rb') as f:  # 加载模型
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图
# 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())
input_mask = sess.graph.get_operation_by_name("input_mask").outputs[0]
input_ids = sess.graph.get_operation_by_name("input_ids").outputs[0]
segment_ids = sess.graph.get_operation_by_name("segment_ids").outputs[0]
loss_softmax = sess.graph.get_operation_by_name("loss/Softmax").outputs[0]
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name)

processor = SimProcessor()
while True:
    sentence1 = input('sentence1: ')
    sentence2 = input('sentence2: ')
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    predict_examples = processor.get_sentence_examples([(sentence1, sentence2)])
    features = convert_examples_to_features(predict_examples, processor.get_labels(), args.max_seq_len,
                                            tokenizer)
    a = {
        'input_ids': [f.input_ids for f in features],
        'input_mask': [f.input_mask for f in features],
        'segment_ids': [f.segment_ids for f in features],
        'label_ids': [f.label_id for f in features]
    }
    sim = sess.run([loss_softmax], {input_mask: a["input_mask"], input_ids: a["input_ids"],
                              segment_ids: a["segment_ids"]})
    for i in sim:
        print(i)
