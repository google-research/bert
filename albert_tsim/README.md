# albert_zh

An Implementation of <a href="https://arxiv.org/pdf/1909.11942.pdf">A Lite Bert For Self-Supervised Learning Language Representations</a> with TensorFlow

ALBert is based on Bert, but with some improvements. It achieves state of the art performance on main benchmarks with 30% parameters less. 

For albert_base_zh it only has ten percentage parameters compare of original bert model, and main accuracy is retained. 


Different version of ALBERT pre-trained model for Chinese, including TensorFlow, PyTorch and Keras, is available now.

海量中文语料上预训练ALBERT模型：参数更少，效果更好。预训练小模型也能拿下13项NLP任务，ALBERT三大改造登顶GLUE基准

一键运行10个数据集、9个基线模型、不同任务上模型效果的详细对比，见<a href="https://github.com/CLUEbenchmark/CLUE">中文任务基准测评 CLUE benchmark</a>

<img src="https://github.com/brightmart/albert_zh/blob/master/resources/albert_tiny_compare_s.jpg"  width="90%" height="70%" />


模型下载 Download Pre-trained Models of Chinese
-----------------------------------------------
1、<a href="https://storage.googleapis.com/albert_zh/albert_tiny.zip">albert_tiny_zh</a>, <a href="https://storage.googleapis.com/albert_zh/albert_tiny_489k.zip">albert_tiny_zh(训练更久，累积学习20亿个样本)</a>，文件大小16M、参数为4M

    训练和推理预测速度提升约10倍，精度基本保留，模型大小为bert的1/25；语义相似度数据集LCQMC测试集上达到85.4%，相比bert_base仅下降1.5个点。

    lcqmc训练使用如下参数： --max_seq_length=128 --train_batch_size=64   --learning_rate=1e-4   --num_train_epochs=5 
    
    albert_tiny使用同样的大规模中文语料数据，层数仅为4层、hidden size等向量维度大幅减少; 尝试使用如下学习率来获得更好效果：{2e-5, 6e-5, 1e-4} 
    
    【使用场景】任务相对比较简单一些或实时性要求高的任务，如语义相似度等句子对任务、分类任务；比较难的任务如阅读理解等，可以使用其他大模型。

     例如，可以使用[Tensorflow Lite](https://www.tensorflow.org/lite)在移动端进行部署，本文[随后](#use_tflite)针对这一点进行了介绍，包括如何把模型转换成Tensorflow Lite格式和对其进行性能测试等。
     
     一键运行albert_tiny_zh(linux,lcqmc任务)：
     1) git clone https://github.com/brightmart/albert_zh
     2) cd albert_zh
     3) bash run_classifier_lcqmc.sh
1.1、<a href="https://storage.googleapis.com/albert_zh/albert_tiny_zh_google.zip">albert_tiny_google_zh(累积学习10亿个样本,google版本)</a>，模型大小16M、性能与albert_tiny_zh一致

1.2、<a href="https://storage.googleapis.com/albert_zh/albert_small_zh_google.zip">albert_small_google_zh(累积学习10亿个样本,google版本)</a>，
     
     速度比bert_base快4倍；LCQMC测试集上比Bert下降仅0.9个点；去掉adam后模型大小18.5M；使用方法，见 #下游任务 Fine-tuning on Downstream Task     
     
2、<a href="https://storage.googleapis.com/albert_zh/albert_large_zh.zip">albert_large_zh</a>,参数量，层数24，文件大小为64M
   
    参数量和模型大小为bert_base的六分之一；在口语化描述相似性数据集LCQMC的测试集上相比bert_base上升0.2个点

3、<a href="https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip">albert_base_zh(额外训练了1.5亿个实例即 36k steps * batch_size 4096)</a>; <a href="https://storage.googleapis.com/albert_zh/albert_base_zh.zip"> albert_base_zh(小模型体验版)</a>, 参数量12M, 层数12，大小为40M

    参数量为bert_base的十分之一，模型大小也十分之一；在口语化描述相似性数据集LCQMC的测试集上相比bert_base下降约0.6~1个点；
    相比未预训练，albert_base提升14个点

4、<a href="https://storage.googleapis.com/albert_zh/albert_xlarge_zh_177k.zip">albert_xlarge_zh_177k </a>; 
<a href="https://storage.googleapis.com/albert_zh/albert_xlarge_zh_183k.zip">albert_xlarge_zh_183k(优先尝试)</a>参数量，层数24，文件大小为230M
   
    参数量和模型大小为bert_base的二分之一；需要一张大的显卡；完整测试对比将后续添加；batch_size不能太小，否则可能影响精度


预训练 Pre-training
-----------------------------------------------

#### 生成特定格式的文件(tfrecords) Generate tfrecords Files

Run following command 运行以下命令即可。项目自动了一个示例的文本文件(data/news_zh_1.txt)
   
       bash create_pretrain_data.sh
   
如果你有很多文本文件，可以通过传入参数的方式，生成多个特定格式的文件(tfrecords）

###### Support English and Other Non-Chinese Language: 
    If you are doing pre-train for english or other language,which is not chinese, 
    you should set hyperparameter of non_chinese to True on create_pretraining_data.py; 
    otherwise, by default it is doing chinese pre-train using whole word mask of chinese.

#### 执行预训练 pre-training on GPU/TPU using the command
    GPU(brightmart版, tiny模型):
    export BERT_BASE_DIR=./albert_tiny_zh
    nohup python3 run_pretraining.py --input_file=./data/tf*.tfrecord  \
    --output_dir=./my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
    --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=51 \
    --num_train_steps=125000 --num_warmup_steps=12500 --learning_rate=0.00176    \
    --save_checkpoints_steps=2000  --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt &
    
    GPU(Google版本, small模型):
    export BERT_BASE_DIR=./albert_small_zh_google
    nohup python3 run_pretraining_google.py --input_file=./data/tf*.tfrecord --eval_batch_size=64 \
    --output_dir=./my_new_model_path --do_train=True --do_eval=True --albert_config_file=$BERT_BASE_DIR/albert_config_small_google.json  --export_dir=./my_new_model_path_export \
    --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=20 \
    --num_train_steps=125000 --num_warmup_steps=12500 --learning_rate=0.00176   \
    --save_checkpoints_steps=2000 --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt
    
    TPU, add something like this:
        --use_tpu=True  --tpu_name=grpc://10.240.1.66:8470 --tpu_zone=us-central1-a
        
    注：如果你重头开始训练，可以不指定init_checkpoint；
    如果你从现有的模型基础上训练，指定一下BERT_BASE_DIR的路径，并确保bert_config_file和init_checkpoint两个参数的值能对应到相应的文件上；
    领域上的预训练，根据数据的大小，可以不用训练特别久。

环境 Environment
-----------------------------------------------
Use Python3 + Tensorflow 1.x 

e.g. Tensorflow 1.4 or 1.5


下游任务 Fine-tuning on Downstream Task
-----------------------------------------------
##### 使用TensorFlow:

以使用albert_base做LCQMC任务为例。LCQMC任务是在口语化描述的数据集上做文本的相似性预测。

We will use LCQMC dataset for fine-tuning, it is oral language corpus, it is used to train and predict semantic similarity of a pair of sentences.

下载<a href="https://drive.google.com/open?id=1HXYMqsXjmA5uIfu_SFqP7r_vZZG-m_H0">LCQMC</a>数据集，包含训练、验证和测试集，训练集包含24万口语化描述的中文句子对，标签为1或0。1为句子语义相似，0为语义不相似。

通过运行下列命令做LCQMC数据集上的fine-tuning:
    
    1. Clone this project:
          
          git clone https://github.com/brightmart/albert_zh.git
          
    2. Fine-tuning by running the following command.
        brightmart版本的tiny模型
        export BERT_BASE_DIR=./albert_tiny_zh
        export TEXT_DIR=./lcqmc
        nohup python3 run_classifier.py   --task_name=lcqmc_pair   --do_train=true   --do_eval=true   --data_dir=$TEXT_DIR   --vocab_file=./albert_config/vocab.txt  \
        --bert_config_file=./albert_config/albert_config_tiny.json --max_seq_length=128 --train_batch_size=64   --learning_rate=1e-4  --num_train_epochs=5 \
        --output_dir=./albert_lcqmc_checkpoints --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt &
        
        google版本的small模型
        export BERT_BASE_DIR=./albert_small_zh
        export TEXT_DIR=./lcqmc
        nohup python3 run_classifier_sp_google.py --task_name=lcqmc_pair   --do_train=true   --do_eval=true   --data_dir=$TEXT_DIR   --vocab_file=./albert_config/vocab.txt  \
        --albert_config_file=./$BERT_BASE_DIR/albert_config_small_google.json --max_seq_length=128 --train_batch_size=64   --learning_rate=1e-4   --num_train_epochs=5 \
        --output_dir=./albert_lcqmc_checkpoints --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt &

    Notice/注：
        1) you need to download pre-trained chinese albert model, and also download LCQMC dataset 
        你需要下载预训练的模型，并放入到项目当前项目，假设目录名称为albert_tiny_zh; 需要下载LCQMC数据集，并放入到当前项目，
        假设数据集目录名称为lcqmc

        2) for Fine-tuning, you can try to add small percentage of dropout(e.g. 0.1) by changing parameters of 
          attention_probs_dropout_prob & hidden_dropout_prob on albert_config_xxx.json. By default, we set dropout as zero. 
        
        3) you can try different learning rate {2e-5, 6e-5, 1e-4} for better performance 


Updates
-----------------------------------------------
**\*\*\*\*\* 2019-11-03: add google version of albert_small, albert_tiny; 

add method to deploy ablert_tiny to mobile devices with only 0.1 second inference time for sequence length 128, 60M memory \*\*\*\*\***

**\*\*\*\*\* 2019-10-30: add a simple guide about converting the model to Tensorflow Lite for edge deployment \*\*\*\*\***

**\*\*\*\*\* 2019-10-15: albert_tiny_zh, 10 times fast than bert base for training and inference, accuracy remains \*\*\*\*\***

**\*\*\*\*\* 2019-10-07: more models of albert \*\*\*\*\***

add albert_xlarge_zh; albert_base_zh_additional_steps, training with more instances

**\*\*\*\*\* 2019-10-04: PyTorch and Keras versions of albert were supported \*\*\*\*\***

a.Convert to PyTorch version and do your tasks through <a href="https://github.com/lonePatient/albert_pytorch">albert_pytorch</a>

b.Load pre-trained model with keras using one line of codes through <a href="https://github.com/bojone/bert4keras">bert4keras</a>

c.Use albert with TensorFlow 2.0: Use or load pre-trained model with tf2.0 through <a href="https://github.com/kpe/bert-for-tf2">bert-for-tf2</a>

Releasing albert_xlarge on 6th Oct

**\*\*\*\*\* 2019-10-02: albert_large_zh,albert_base_zh \*\*\*\*\***

Relesed albert_base_zh with only 10% parameters of bert_base, a small model(40M) & training can be very fast. 

Relased albert_large_zh with only 16% parameters of bert_base(64M)

**\*\*\*\*\* 2019-09-28: codes and test functions \*\*\*\*\*** 

Add codes and test functions for three main changes of albert from bert

ALBERT模型介绍 Introduction of ALBERT
-----------------------------------------------
ALBERT模型是BERT的改进版，与最近其他State of the art的模型不同的是，这次是预训练小模型，效果更好、参数更少。

它对BERT进行了三个改造 Three main changes of ALBert from Bert：

1）词嵌入向量参数的因式分解 Factorized embedding parameterization
   
     O(V * H) to O(V * E + E * H)
     
     如以ALBert_xxlarge为例，V=30000, H=4096, E=128
       
     那么原先参数为V * H= 30000 * 4096 = 1.23亿个参数，现在则为V * E + E * H = 30000*128+128*4096 = 384万 + 52万 = 436万，
       
     词嵌入相关的参数变化前是变换后的28倍。


2）跨层参数共享 Cross-Layer Parameter Sharing

     参数共享能显著减少参数。共享可以分为全连接层、注意力层的参数共享；注意力层的参数对效果的减弱影响小一点。

3）段落连续性任务 Inter-sentence coherence loss.
     
     使用段落连续性任务。正例，使用从一个文档中连续的两个文本段落；负例，使用从一个文档中连续的两个文本段落，但位置调换了。
     
     避免使用原有的NSP任务，原有的任务包含隐含了预测主题这类过于简单的任务。

      We maintain that inter-sentence modeling is an important aspect of language understanding, but we propose a loss 
      based primarily on coherence. That is, for ALBERT, we use a sentence-order prediction (SOP) loss, which avoids topic 
      prediction and instead focuses on modeling inter-sentence coherence. The SOP loss uses as positive examples the 
      same technique as BERT (two consecutive segments from the same document), and as negative examples the same two 
      consecutive segments but with their order swapped. This forces the model to learn finer-grained distinctions about
      discourse-level coherence properties. 

其他变化，还有 Other changes：

    1）去掉了dropout  Remove dropout to enlarge capacity of model.
        最大的模型，训练了1百万步后，还是没有过拟合训练数据。说明模型的容量还可以更大，就移除了dropout
        （dropout可以认为是随机的去掉网络中的一部分，同时使网络变小一些）
        We also note that, even after training for 1M steps, our largest models still do not overfit to their training data. 
        As a result, we decide to remove dropout to further increase our model capacity.
        其他型号的模型，在我们的实现中我们还是会保留原始的dropout的比例，防止模型对训练数据的过拟合。
        
    2）为加快训练速度，使用LAMB做为优化器 Use LAMB as optimizer, to train with big batch size
      使用了大的batch_size来训练(4096)。 LAMB优化器使得我们可以训练，特别大的批次batch_size，如高达6万。
    
    3）使用n-gram(uni-gram,bi-gram, tri-gram）来做遮蔽语言模型 Use n-gram as make language model
       即以不同的概率使用n-gram,uni-gram的概率最大，bi-gram其次，tri-gram概率最小。
       本项目中目前使用的是在中文上做whole word mask，稍后会更新一下与n-gram mask的效果对比。n-gram从spanBERT中来。


训练语料/训练配置 Training Data & Configuration
-----------------------------------------------
30g中文语料，超过100亿汉字，包括多个百科、新闻、互动社区。

预训练序列长度sequence_length设置为512，批次batch_size为4096，训练产生了3.5亿个训练数据(instance)；每一个模型默认会训练125k步，albert_xxlarge将训练更久。

作为比较，roberta_zh预训练产生了2.5亿个训练数据、序列长度为256。由于albert_zh预训练生成的训练数据更多、使用的序列长度更长，
 
    我们预计albert_zh会有比roberta_zh更好的性能表现，并且能更好处理较长的文本。

训练使用TPU v3 Pod，我们使用的是v3-256，它包含32个v3-8。每个v3-8机器，含有128G的显存。


模型性能与对比(英文) Performance and Comparision
-----------------------------------------------    
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/state_of_the_art.jpg"  width="80%" height="40%" />
  
   
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/albert_performance.jpg"  width="80%" height="40%" />


<img src="https://github.com/brightmart/albert_zh/blob/master/resources/add_data_removing_dropout.jpg"  width="80%" height="40%" />


中文任务集上效果对比测试 Performance on Chinese datasets
----------------------------------------------- 

###  问题匹配语任务：LCQMC(Sentence Pair Matching)

| 模型 | 开发集(Dev) | 测试集(Test) |
| :------- | :---------: | :---------: |
| BERT | 89.4(88.4) | 86.9(86.4) | 
| ERNIE | 89.8 (89.6) | 87.2 (87.0) | 
| BERT-wwm |89.4 (89.2) | 87.0 (86.8) | 
| BERT-wwm-ext | - |-  |
| RoBERTa-zh-base | 88.7 | 87.0  |
| RoBERTa-zh-Large | ***89.9(89.6)*** | 87.2(86.7) |
| RoBERTa-zh-Large(20w_steps) | 89.7| 87.0 |
| ALBERT-zh-tiny | -- | 85.4 |
| ALBERT-zh-small | -- | 86.0 |
| ALBERT-zh-small(Pytorch) | -- | 86.8 |
| ALBERT-zh-base-additional-36k-steps | 87.8 | 86.3 |
| ALBERT-zh-base | 87.2 | 86.3 |
| ALBERT-large | 88.7 | 87.1 |
| ALBERT-xlarge | 87.3 | ***87.7*** |

注：只跑了一次ALBERT-xlarge，效果还可能提升

### 自然语言推断：XNLI of Chinese Version

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 77.8 (77.4) | 77.8 (77.5) | 
| ERNIE | 79.7 (79.4) | 78.6 (78.2) | 
| BERT-wwm | 79.0 (78.4) | 78.2 (78.0) | 
| BERT-wwm-ext | 79.4 (78.6) | 78.7 (78.3) |
| XLNet | 79.2  | 78.7 |
| RoBERTa-zh-base | 79.8 |78.8  |
| RoBERTa-zh-Large | 80.2 (80.0) | 79.9 (79.5) |
| ALBERT-base | 77.0 | 77.1 |
| ALBERT-large | 78.0 | 77.5 |
| ALBERT-xlarge | ? | ? |

注：BERT-wwm-ext来自于<a href="https://github.com/ymcui/Chinese-BERT-wwm">这里</a>；XLNet来自于<a href="https://github.com/ymcui/Chinese-PreTrained-XLNet">这里</a>; RoBERTa-zh-base，指12层RoBERTa中文模型
   

###  阅读理解任务：CRMC2018

<img src="https://github.com/brightmart/albert_zh/blob/master/resources/crmc2018_compare_s.jpg"  width="90%" height="70%" />


### 语言模型、文本段预测准确性、训练时间 Mask Language Model Accuarcy & Training Time

| Model | MLM eval acc | SOP eval acc | Training(Hours) | Loss eval |
| :------- | :---------: | :---------: | :---------: |:---------: |
| albert_zh_base | 79.1% | 99.0% | 6h | 1.01|
| albert_zh_large | 80.9% | 98.6% | 22.5h | 0.93|
| albert_zh_xlarge | ? | ? | 53h(预估) | ? |
| albert_zh_xxlarge | ? | ? | 106h(预估) | ? |

注：? 将很快替换

模型参数和配置 Configuration of Models
-----------------------------------------------
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/albert_configuration.jpg"  width="80%" height="40%" />

代码实现和测试 Implementation and Code Testing
-----------------------------------------------
通过运行以下命令测试主要的改进点，包括但不限于词嵌入向量参数的因式分解、跨层参数共享、段落连续性任务等。

    python test_changes.py

##### <a name="use_tflite"></a>使用TensorFlow Lite(TFLite)在移动端进行部署:
这里我们主要介绍TFLite模型格式转换和性能测试。转换成TFLite模型后，对于如何在移
动端使用该模型，可以参考TFLite提供的[Android/iOS应用完整开发案例教程页面](https://www.tensorflow.org/lite/examples)。
该页面目前已经包含了[文本分类](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android)，
[文本问答](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android)两个Android案例。

下面以<a href="https://storage.googleapis.com/albert_zh/albert_tiny.zip">albert_tiny_zh</a>
为例来介绍TFLite模型格式转换和性能测试：

1. Freeze graph from the checkpoint

Ensure to have >=1.14 1.x installed to use the freeze_graph tool as it is removed from 2.x distribution

    pip install tensorflow==1.15

    freeze_graph --input_checkpoint=./albert_model.ckpt \
      --output_graph=/tmp/albert_tiny_zh.pb \
      --output_node_names=cls/predictions/truediv \
      --checkpoint_version=1 --input_meta_graph=./albert_model.ckpt.meta --input_binary=true

2. Convert to TFLite format

We are going to use the new experimental tf->tflite converter that's distributed with the Tensorflow nightly build.

    pip install tf-nightly

    tflite_convert --graph_def_file=/tmp/albert_tiny_zh.pb \
      --input_arrays='input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,masked_lm_weights' \
      --output_arrays='cls/predictions/truediv' \
      --input_shapes=1,128:1,128:128:1,128:1,128:1,128 \
      --output_file=/tmp/albert_tiny_zh.tflite \
      --enable_v1_converter --experimental_new_converter

3. Benchmark the performance of the TFLite model

See [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) 
for details about the performance benchmark tools in TFLite. For example: after
building the benchmark tool binary for an Android phone, do the following to
get an idea of how the TFLite model performs on the phone

    adb push /tmp/albert_tiny_zh.tflite /data/local/tmp/
    adb shell /data/local/tmp/benchmark_model_performance_options --graph=/data/local/tmp/albert_tiny_zh.tflite --perf_options_list=cpu

On an Android phone w/ Qualcomm's SD845 SoC, via the above benchmark tool, as
of 2019/11/01, the inference latency is ~120ms w/ this converted TFLite model
using 4 threads on CPU, and the memory usage is ~60MB for the model during
inference. Note the performance will improve further with future TFLite
implementation optimizations.

##### 使用PyTorch版本:

    download pre-trained model, and convert to PyTorch using:
     
      python convert_albert_tf_checkpoint_to_pytorch.py     
     
   using <a href="https://github.com/lonePatient/albert_pytorch">albert_pytorch
   
##### 使用Keras加载:

<a href="https://github.com/bojone/bert4keras">bert4keras</a> 适配albert，能成功加载albert_zh的权重，只需要在load_pretrained_model函数里加上albert=True

load pre-trained model with bert4keras

##### 使用tf2.0加载:

<a href="https://github.com/kpe/bert-for-tf2">bert-for-tf2</a>


使用案例-基于用户输入预测文本相似性 Use Case-Text Similarity Based on User Input
-------------------------------------------------

功能说明：用户可以通过本例了解如何加载训训练集实现基于用户输入的短文本相似度判断。可以基于该代码将程序灵活地拓展为后台服务或增加文本分类等示例。

涉及代码：similarity.py、args.py

步骤：

1、使用本模型进行文本相似性训练，保存模型文件至相应目录下

2、根据实际情况，修改args.py中的参数，参数说明如下：

```python
#模型目录，存放ckpt文件
model_dir = os.path.join(file_path, 'albert_lcqmc_checkpoints/')

#config文件，存放模型的json文件
config_name = os.path.join(file_path, 'albert_config/albert_config_tiny.json')

#ckpt文件名称
ckpt_name = os.path.join(model_dir, 'model.ckpt')

#输出文件目录，训练时的模型输出目录
output_dir = os.path.join(file_path, 'albert_lcqmc_checkpoints/')

#vocab文件目录
vocab_file = os.path.join(file_path, 'albert_config/vocab.txt')

#数据目录，训练使用的数据集存放目录
data_dir = os.path.join(file_path, 'data/')
```

本例中的文件结构为：

    |__args.py
    
    |__similarity.py
    
    |__data
    
    |__albert_config
    
    |__albert_lcqmc_checkpoints
    
    |__lcqmc

3、修改用户输入单词

打开similarity.py，最底部如下代码：

```python
if __name__ == '__main__':
    sim = BertSim()
    sim.start_model()
    sim.predict_sentences([("我喜欢妈妈做的汤", "妈妈做的汤我很喜欢喝")])
```

其中sim.start_model()表示加载模型，sim.predict_sentences的输入为一个元组数组，元组中包含两个元素分别为需要判定相似的句子。

4、运行python文件：similarity.py


支持的序列长度与批次大小的关系,12G显存 Trade off between batch Size and sequence length
-------------------------------------------------

System       | Seq Length | Max Batch Size
------------ | ---------- | --------------
`albert-base`  | 64         | 64
...          | 128        | 32
...          | 256        | 16
...          | 320        | 14
...          | 384        | 12
...          | 512        | 6
`albert-large` | 64         | 12
...          | 128        | 6
...          | 256        | 2
...          | 320        | 1
...          | 384        | 0
...          | 512        | 0
`albert-xlarge` | -         | -

学习曲线 Training Loss of xlarge of albert_zh
-------------------------------------------------
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/xlarge_loss.jpg"  width="80%" height="40%" />

所有的参数 Parameters of albert_xlarge
-------------------------------------------------
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/albert_large_zh_parameters.jpg"  width="80%" height="40%" />


#### 技术交流与问题讨论QQ群: 836811304 Join us on QQ group

If you have any question, you can raise an issue, or send me an email: brightmart@hotmail.com;

Currently how to use PyTorch version of albert is not clear yet, if you know how to do that, just email us or open an issue.

You can also send pull request to report you performance on your task or add methods on how to load models for PyTorch and so on.

If you have ideas for generate best performance pre-training Chinese model, please also let me know.

##### Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC)

Cite Us
-----------------------------------------------
Bright Liang Xu, albert_zh, (2019), GitHub repository, https://github.com/brightmart/albert_zh

Reference
-----------------------------------------------
1、<a href="https://arxiv.org/pdf/1909.11942.pdf">ALBERT: A Lite BERT For Self-Supervised Learning Of Language Representations</a>

2、<a href="https://arxiv.org/pdf/1810.04805.pdf">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>

3、<a href="https://arxiv.org/abs/1907.10529">SpanBERT: Improving Pre-training by Representing and Predicting Spans</a>

4、<a href="https://arxiv.org/pdf/1907.11692.pdf">RoBERTa: A Robustly Optimized BERT Pretraining Approach</a>

5、<a href="https://arxiv.org/pdf/1904.00962.pdf">Large Batch Optimization for Deep Learning: Training BERT in 76 minutes(LAMB)</a>

6、<a href="https://github.com/ymcui/LAMB_Optimizer_TF">LAMB Optimizer,TensorFlow version</a>

7、<a href="http://baijiahao.baidu.com/s?id=1645712785366950083&wfr=spider&for=pc">预训练小模型也能拿下13项NLP任务，ALBERT三大改造登顶GLUE基准</a>

8、 <a href="https://github.com/lonePatient/albert_pytorch">albert_pytorch</a>

9、<a href="https://github.com/bojone/bert4keras">load albert with keras</a>

10、<a href="https://github.com/kpe/bert-for-tf2">load albert with tf2.0</a>

11、<a href="https://github.com/google-research/google-research/tree/master/albert">repo of albert from google</a>

12、<a href="https://github.com/chineseGLUE/chineseGLUE">chineseGLUE-中文任务基准测评：公开可用多个任务、基线模型、广泛测评与效果对比</a>




