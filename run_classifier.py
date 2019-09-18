# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import horovod.tensorflow as hvd
import time
from utils.utils import LogEvalRunHook, LogTrainRunHook
from utils.create_glue_data import *
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_bool("use_trt", False, "Whether to use TF-TRT")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("num_accumulation_steps", 1,
                     "Number of accumulation steps before gradient update" 
                      "Global batch size = num_accumulation_steps * train_batch_size")
flags.DEFINE_bool("use_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")

flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")
flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")


def file_based_input_fn_builder(input_file, batch_size, seq_length, is_training,
                                drop_remainder, hvd=None):
  """Creates an `input_fn` closure to be passed to Estimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn():
    """The actual input function."""

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      compute_type=tf.float16 if FLAGS.use_fp16 else tf.float32)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias, name='cls_logits')
    probabilities = tf.nn.softmax(logits, axis=-1, name='cls_probabilities')
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1, name='cls_per_example_loss')
    loss = tf.reduce_mean(per_example_loss, name='cls_loss')

    return (loss, per_example_loss, logits, probabilities)

def get_frozen_tftrt_model(bert_config, shape, num_labels, use_one_hot_embeddings, init_checkpoint):
  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth = True
  output_node_names = ['loss/cls_loss', 'loss/cls_per_example_loss', 'loss/cls_logits', 'loss/cls_probabilities']

  with tf.Session(config=tf_config) as tf_sess:
    input_ids = tf.placeholder(tf.int32, shape, 'input_ids')
    input_mask = tf.placeholder(tf.int32, shape, 'input_mask')
    segment_ids = tf.placeholder(tf.int32, shape, 'segment_ids')
    label_ids = tf.placeholder(tf.int32, (None), 'label_ids')

    create_model(bert_config, False, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    tf_sess.run(tf.global_variables_initializer())
    print("LOADED!")
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      else:
        init_string = ", *NOTTTTTTTTTTTTTTTTTTTTT"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    frozen_graph = tf.graph_util.convert_variables_to_constants(tf_sess, 
            tf_sess.graph.as_graph_def(), output_node_names)

    num_nodes = len(frozen_graph.node)
    print('Converting graph using TensorFlow-TensorRT...')
    import tensorflow.contrib.tensorrt as trt
    frozen_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_node_names, 
        max_batch_size=FLAGS.predict_batch_size,
        max_workspace_size_bytes=(4096 << 20) - 1000,
        precision_mode = "FP16" if FLAGS.use_fp16 else "FP32",
        minimum_segment_size=4,
        is_dynamic_op=True,
        maximum_cached_engines=1000
    )

    print('Total node count before and after TF-TRT conversion:',
          num_nodes, '->', len(frozen_graph.node))
    print('TRT node count:',
          len([1 for n in frozen_graph.node if str(n.op) == 'TRTEngineOp']))
    
    with tf.gfile.GFile("frozen_modelTRT.pb", "wb") as f:
      f.write(frozen_graph.SerializeToString())      
        
  return frozen_graph



def model_fn_builder(task_name, bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings, hvd=None):
  """Returns `model_fn` closure for Estimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator."""

    def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        if task_name == "cola":
            FN, FN_op = tf.metrics.false_negatives(labels=label_ids, predictions=predictions)
            FP, FP_op = tf.metrics.false_positives(labels=label_ids, predictions=predictions)
            TP, TP_op = tf.metrics.true_positives(labels=label_ids, predictions=predictions)
            TN, TN_op = tf.metrics.true_negatives(labels=label_ids, predictions=predictions)

            MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
            MCC_op = tf.group(FN_op, TN_op, TP_op, FP_op, tf.identity(MCC, name="MCC"))
            return {"MCC": (MCC, MCC_op)}
        else:
            accuracy = tf.metrics.accuracy(
                labels=label_ids, predictions=predictions)
            loss = tf.metrics.mean(values=per_example_loss)
            return {
                "eval_accuracy": accuracy,
                "eval_loss": loss,
            }
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if not is_training and FLAGS.use_trt:
        trt_graph = get_frozen_tftrt_model(bert_config, input_ids.shape, num_labels, use_one_hot_embeddings, init_checkpoint)
        (total_loss, per_example_loss, logits, probabilities)  = tf.import_graph_def(trt_graph,
                input_map={'input_ids':input_ids, 'input_mask':input_mask, 'segment_ids':segment_ids, 'label_ids':label_ids},
                return_elements=['loss/cls_loss:0', 'loss/cls_per_example_loss:0', 'loss/cls_logits:0', 'loss/cls_probabilities:0'],
                name='')
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"probabilities": probabilities}
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = metric_fn(per_example_loss, label_ids, logits)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metric_ops)
        return output_spec
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint and (hvd is None or hvd.rank() == 0):
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if FLAGS.verbose_logging:
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
          init_string = ""
          if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
          tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                          init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          hvd, False, FLAGS.use_fp16, FLAGS.num_accumulation_steps)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = metric_fn(per_example_loss, label_ids, logits)
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metric_ops)
    else:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode, predictions=probabilities)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, batch_size, seq_length, is_training, drop_remainder, hvd=None):
  """Creates an `input_fn` closure to be passed to Estimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn():
    """The actual input function."""

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.horovod:
    hvd.init()
  if FLAGS.use_fp16:
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
  }

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  master_process = True
  training_hooks = []
  global_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulation_steps
  hvd_rank = 0

  config = tf.ConfigProto()
  if FLAGS.horovod:

      tf.logging.info("Multi-GPU training with TF Horovod")
      tf.logging.info("hvd.size() = %d hvd.rank() = %d", hvd.size(), hvd.rank())
      global_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulation_steps * hvd.size()
      master_process = (hvd.rank() == 0)
      hvd_rank = hvd.rank()
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      if hvd.size() > 1:
          training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))
  if FLAGS.use_xla:
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir if master_process else None,
      session_config=config,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps if master_process else None,
      keep_checkpoint_max=1)

  if master_process:
      tf.logging.info("***** Configuaration *****")
      for key in FLAGS.__flags.keys():
          tf.logging.info('  {}: {}'.format(key, getattr(FLAGS, key)))
      tf.logging.info("**************************")

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  training_hooks.append(LogTrainRunHook(global_batch_size, hvd_rank))

  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / global_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    start_index = 0
    end_index = len(train_examples)
    tmp_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record")]

    if FLAGS.horovod:
      tmp_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record{}".format(i)) for i in range(hvd.size())]
      num_examples_per_rank = len(train_examples) // hvd.size()
      remainder = len(train_examples) % hvd.size()
      if hvd.rank() < remainder:
        start_index = hvd.rank() * (num_examples_per_rank+1)
        end_index = start_index + num_examples_per_rank + 1
      else:
        start_index = hvd.rank() * num_examples_per_rank + remainder
        end_index = start_index + (num_examples_per_rank)

  model_fn = model_fn_builder(
      task_name=task_name,
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate if not FLAGS.horovod else FLAGS.learning_rate * hvd.size(),
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_one_hot_embeddings=False,
      hvd=None if not FLAGS.horovod else hvd)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

  if FLAGS.do_train:

    file_based_convert_examples_to_features(
        train_examples[start_index:end_index], label_list, FLAGS.max_seq_length, tokenizer, tmp_filenames[hvd_rank])

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=tmp_filenames,
        batch_size=FLAGS.train_batch_size,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        hvd=None if not FLAGS.horovod else hvd)

    train_start_time = time.time()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=training_hooks)
    train_time_elapsed = time.time() - train_start_time
    train_time_wo_overhead = training_hooks[-1].total_time
    avg_sentences_per_second = num_train_steps * global_batch_size * 1.0 / train_time_elapsed
    ss_sentences_per_second = (num_train_steps - training_hooks[-1].skipped) * global_batch_size * 1.0 / train_time_wo_overhead

    if master_process:
        tf.logging.info("-----------------------------")
        tf.logging.info("Total Training Time = %0.2f for Sentences = %d", train_time_elapsed,
                        num_train_steps * global_batch_size)
        tf.logging.info("Total Training Time W/O Overhead = %0.2f for Sentences = %d", train_time_wo_overhead,
                        (num_train_steps - training_hooks[-1].skipped) * global_batch_size)
        tf.logging.info("Throughput Average (sentences/sec) with overhead = %0.2f", avg_sentences_per_second)
        tf.logging.info("Throughput Average (sentences/sec) = %0.2f", ss_sentences_per_second)
        tf.logging.info("-----------------------------")

  if FLAGS.do_eval and master_process:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_drop_remainder = False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        batch_size=FLAGS.eval_batch_size,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    eval_hooks = [LogEvalRunHook(FLAGS.eval_batch_size)]
    eval_start_time = time.time()
    result = estimator.evaluate(input_fn=eval_input_fn, hooks=eval_hooks)

    eval_time_elapsed = time.time() - eval_start_time
    eval_time_wo_overhead = eval_hooks[-1].total_time

    time_list = eval_hooks[-1].time_list
    time_list.sort()
    num_sentences = (eval_hooks[-1].count - eval_hooks[-1].skipped) * FLAGS.eval_batch_size

    avg = np.mean(time_list)
    cf_50 = max(time_list[:int(len(time_list) * 0.50)])
    cf_90 = max(time_list[:int(len(time_list) * 0.90)])
    cf_95 = max(time_list[:int(len(time_list) * 0.95)])
    cf_99 = max(time_list[:int(len(time_list) * 0.99)])
    cf_100 = max(time_list[:int(len(time_list) * 1)])
    ss_sentences_per_second = num_sentences * 1.0 / eval_time_wo_overhead

    tf.logging.info("-----------------------------")
    tf.logging.info("Total Inference Time = %0.2f for Sentences = %d", eval_time_elapsed,
                    eval_hooks[-1].count * FLAGS.eval_batch_size)
    tf.logging.info("Total Inference Time W/O Overhead = %0.2f for Sentences = %d", eval_time_wo_overhead,
                    (eval_hooks[-1].count - eval_hooks[-1].skipped) * FLAGS.eval_batch_size)
    tf.logging.info("Summary Inference Statistics on EVAL set")
    tf.logging.info("Batch size = %d", FLAGS.eval_batch_size)
    tf.logging.info("Sequence Length = %d", FLAGS.max_seq_length)
    tf.logging.info("Precision = %s", "fp16" if FLAGS.use_fp16 else "fp32")
    tf.logging.info("Latency Confidence Level 50 (ms) = %0.2f", cf_50 * 1000)
    tf.logging.info("Latency Confidence Level 90 (ms) = %0.2f", cf_90 * 1000)
    tf.logging.info("Latency Confidence Level 95 (ms) = %0.2f", cf_95 * 1000)
    tf.logging.info("Latency Confidence Level 99 (ms) = %0.2f", cf_99 * 1000)
    tf.logging.info("Latency Confidence Level 100 (ms) = %0.2f", cf_100 * 1000)
    tf.logging.info("Latency Average (ms) = %0.2f", avg * 1000)
    tf.logging.info("Throughput Average (sentences/sec) = %0.2f", ss_sentences_per_second)
    tf.logging.info("-----------------------------")


    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict and master_process:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        batch_size=FLAGS.predict_batch_size,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    predict_hooks = [LogEvalRunHook(FLAGS.predict_batch_size)]
    predict_start_time = time.time()

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        for prediction in estimator.predict(input_fn=predict_input_fn, hooks=predict_hooks,
                                            yield_single_examples=False):
            output_line = "\t".join(
                str(class_probability) for class_probability in prediction) + "\n"
            writer.write(output_line)


    predict_time_elapsed = time.time() - predict_start_time
    predict_time_wo_overhead = predict_hooks[-1].total_time

    time_list = predict_hooks[-1].time_list
    time_list.sort()
    num_sentences = (predict_hooks[-1].count - predict_hooks[-1].skipped) * FLAGS.predict_batch_size

    avg = np.mean(time_list)
    cf_50 = max(time_list[:int(len(time_list) * 0.50)])
    cf_90 = max(time_list[:int(len(time_list) * 0.90)])
    cf_95 = max(time_list[:int(len(time_list) * 0.95)])
    cf_99 = max(time_list[:int(len(time_list) * 0.99)])
    cf_100 = max(time_list[:int(len(time_list) * 1)])
    ss_sentences_per_second = num_sentences * 1.0 / predict_time_wo_overhead

    tf.logging.info("-----------------------------")
    tf.logging.info("Total Inference Time = %0.2f for Sentences = %d", predict_time_elapsed,
                    predict_hooks[-1].count * FLAGS.predict_batch_size)
    tf.logging.info("Total Inference Time W/O Overhead = %0.2f for Sentences = %d", predict_time_wo_overhead,
                    (predict_hooks[-1].count - predict_hooks[-1].skipped) * FLAGS.predict_batch_size)

    tf.logging.info("Summary Inference Statistics on TEST SET")
    tf.logging.info("Batch size = %d", FLAGS.predict_batch_size)
    tf.logging.info("Sequence Length = %d", FLAGS.max_seq_length)
    tf.logging.info("Precision = %s", "fp16" if FLAGS.use_fp16 else "fp32")
    tf.logging.info("Latency Confidence Level 50 (ms) = %0.2f", cf_50 * 1000)
    tf.logging.info("Latency Confidence Level 90 (ms) = %0.2f", cf_90 * 1000)
    tf.logging.info("Latency Confidence Level 95 (ms) = %0.2f", cf_95 * 1000)
    tf.logging.info("Latency Confidence Level 99 (ms) = %0.2f", cf_99 * 1000)
    tf.logging.info("Latency Confidence Level 100 (ms) = %0.2f", cf_100 * 1000)
    tf.logging.info("Latency Average (ms) = %0.2f", avg * 1000)
    tf.logging.info("Throughput Average (sentences/sec) = %0.2f", ss_sentences_per_second)
    tf.logging.info("-----------------------------")


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()