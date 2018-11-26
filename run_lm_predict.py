# coding=utf-8
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
"""BERT language model predict."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import modeling
import tokenization
import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("max_predictions_per_seq", 20,
"In this task, it also refers to maximum number of masked tokens per word.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

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

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  def __init__(self, unique_id, text):
    self.unique_id = unique_id
    self.text = text


def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    while True:
      line = tokenization.convert_to_unicode(reader.readline())
      if not line:
        break
      line = line.strip()
      unique_id += 1
      examples.append(
        InputExample(unique_id, line))
      unique_id += 1
  return examples


def model_fn_builder(bert_config, init_checkpoint, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]


    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    masked_lm_example_loss = get_masked_lm_output(
        bert_config, model.get_sequence_output(), model.get_embedding_table(),
        masked_lm_positions, masked_lm_ids)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=masked_lm_example_loss, scaffold_fn=scaffold_fn)  # 输出mask_word的score
    return output_spec

  return model_fn




def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    loss = tf.reshape(per_example_loss, [-1, tf.shape(positions)[1]])
    # TODO: dynamic gather from per_example_loss
  return loss



def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(features, seq_length, max_predictions_per_seq):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_masked_lm_positions = []
  all_masked_lm_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_masked_lm_positions.append(feature.masked_lm_positions)
    all_masked_lm_ids.append(feature.masked_lm_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]
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
        "masked_lm_positions":
            tf.constant(
                all_masked_lm_positions,
                shape=[num_examples, max_predictions_per_seq],
                dtype=tf.int32),
        "masked_lm_ids":
            tf.constant(
                all_masked_lm_ids,
                shape=[num_examples, max_predictions_per_seq],
                dtype=tf.int32)
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn



# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, max_seq_length, tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  all_features = []
  all_tokens = []

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    features, tokens = convert_single_example(ex_index, example,
                                     max_seq_length, tokenizer)
    all_features.extend(features)
    all_tokens.extend(tokens)

  return all_features, all_tokens

tokenizer = tokenization.FullTokenizer(
  vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
MASKED_TOKEN = "[MASK]"
MASKED_ID = tokenizer.convert_tokens_to_ids([MASKED_TOKEN])[0]


def create_masked_lm_prediction(input_ids, mask_position, mask_count=1):
  new_input_ids = list(input_ids)
  masked_lm_labels = []
  masked_lm_positions = list(range(mask_position, mask_position + mask_count))
  for i in masked_lm_positions:
    new_input_ids[i] = MASKED_ID
    masked_lm_labels.append(input_ids[i])
  return new_input_ids, masked_lm_positions, masked_lm_labels


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, segment_ids, input_mask, masked_lm_positions,
               masked_lm_ids):
    self.input_ids = input_ids,
    self.segment_ids = segment_ids,
    self.input_mask = input_mask,
    self.masked_lm_positions = masked_lm_positions,
    self.masked_lm_ids = masked_lm_ids,


def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  tokens = tokenizer.tokenize(example.text)

  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens) > max_seq_length - 2:
    tokens = tokens[0:(max_seq_length - 2)]

  input_tokens = []
  segment_ids = []
  input_tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens:
    input_tokens.append(token)
    segment_ids.append(0)
  input_tokens.append("[SEP]")
  segment_ids.append(0)

  input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("id: %s" % (example.unique_id))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in input_tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

  features = create_sequential_mask(input_tokens, input_ids, input_mask, segment_ids,
                                    FLAGS.max_predictions_per_seq)
  return features, input_tokens


def is_subtoken(x):
  return x.startswith("##")

def create_sequential_mask(input_tokens, input_ids, input_mask, segment_ids,
                           max_predictions_per_seq):
  """Mask each token/word sequentially"""
  features = []
  i = 1
  while i < len(input_tokens) - 1:
    mask_count = 1
    while is_subtoken(input_tokens[i+mask_count]):
      mask_count += 1

    input_ids_new, masked_lm_positions, masked_lm_labels = create_masked_lm_prediction(input_ids, i, mask_count)
    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_labels.append(0)

    feature = InputFeatures(
      input_ids=input_ids_new,
      input_mask=input_mask,
      segment_ids=segment_ids,
      masked_lm_positions=masked_lm_positions,
      masked_lm_ids=masked_lm_labels)
    features.append(feature)
    i += mask_count
  return features


def parse_result(result, all_tokens, output_file=None):
  with tf.gfile.GFile(output_file, "w") as writer:
    tf.logging.info("***** Predict results *****")
    i = 0
    sentences = []
    for word_loss in result:
      # start of a sentence
      if all_tokens[i] == "[CLS]":
        sentence = {}
        tokens = []
        sentence_loss = 0.0
        word_count_per_sent = 0
        i += 1

      # add token
      tokens.append({"token": tokenization.printable_text(all_tokens[i]),
                     "prob": float(np.exp(-word_loss[0])) })
      sentence_loss += word_loss[0]
      word_count_per_sent += 1
      i += 1

      token_count_per_word = 0
      while is_subtoken(all_tokens[i]):
        token_count_per_word += 1
        tokens.append({"token": tokenization.printable_text(all_tokens[i]),
                       "prob": float(np.exp(-word_loss[token_count_per_word]))})
        sentence_loss += word_loss[token_count_per_word]
        i += 1

      # end of a sentence
      if all_tokens[i] == "[SEP]":
        sentence["tokens"] = tokens
        sentence["ppl"] = float(np.exp(sentence_loss / word_count_per_sent))
        sentences.append(sentence)
        i += 1

    if output_file is not None:
      tf.logging.info("Saving results to %s" % output_file)
      writer.write(json.dumps(sentences, indent=2, ensure_ascii=False))

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))


  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.predict_batch_size)


  predict_examples = read_examples(FLAGS.input_file)
  features, all_tokens = convert_examples_to_features(predict_examples,
                                          FLAGS.max_seq_length, tokenizer)

  tf.logging.info("***** Running prediction*****")
  tf.logging.info("  Num examples = %d", len(predict_examples))
  tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

  if FLAGS.use_tpu:
    # Warning: According to tpu_estimator.py Prediction on TPU is an
    # experimental feature and hence not supported here
    raise ValueError("Prediction in TPU not supported")

  predict_input_fn = input_fn_builder(
      features=features,
      seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq)

  result = estimator.predict(input_fn=predict_input_fn)
  output_predict_file = os.path.join(FLAGS.output_dir, "test_results.json")
  parse_result(result, all_tokens, output_predict_file)

if __name__ == "__main__":
  tf.app.run()
