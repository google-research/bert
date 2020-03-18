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
"""run text infilling for pre-trained BERT without fine-tuning"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import modeling
import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 64,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")

flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

flags.DEFINE_integer("batch_size", 1, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None, "If using a TPU, the address of the master.")

flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True since it is much faster.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from metadata.")


class InputExample(object):

    def __init__(self, unique_id, text, mask_idx):
        self.unique_id = unique_id
        self.text = text
        self.mask_idx = mask_idx


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, target_idx):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.target_idx = target_idx


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    batch_size, seq_length, width = modeling.get_shape_list(sequence_tensor, expected_rank=3)

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = [feature.unique_id for feature in features]
    all_input_ids = [feature.input_ids for feature in features]
    all_input_mask = [feature.input_mask for feature in features]
    all_input_type_ids = [feature.input_type_ids for feature in features]
    all_target_ids = [feature.target_idx for feature in features]

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids": tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids": tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "input_mask": tf.constant(all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
            "input_type_ids": tf.constant(all_input_type_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "target_idx": tf.constant(all_target_ids, shape=[num_examples], dtype=tf.int32),
        })
        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, use_tpu, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]
        target_idx = features['target_idx']
        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % mode)

        input_tensor = model.get_sequence_output()  # [batch_size, seq_len, hidden_size]
        input_tensor = tf.squeeze(input_tensor, axis=0)
        output_weights = model.get_embedding_table()
        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable("output_bias", shape=[bert_config.vocab_size],
                                          initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            _, predicted_index = tf.nn.top_k(log_probs, k=5)
            predicted_index = tf.expand_dims(predicted_index, 0)
        predictions = {"unique_id": unique_ids, "predicted_index": predicted_index, "target_idx": target_idx}

        tvars = tf.trainable_variables()
        scaffold_fn = None
        assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            if var.name not in initialized_variable_names:
                init_string = ", *NOT_INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        valid_tokens = tokens[0: seq_length - 2]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        target_token = ''
        for i, token in enumerate(valid_tokens):
            if i == example.mask_idx:
                tokens.append('[MASK]')
                target_token = token
            else:
                tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % example.unique_id)
            tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
            tf.logging.info("target_token: %s" % target_token)

        features.append(InputFeatures(unique_id=example.unique_id, tokens=tokens, input_ids=input_ids,
                                      input_mask=input_mask, input_type_ids=input_type_ids, target_idx=example.mask_idx))
    return features


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    with tf.gfile.GFile(input_file, "r") as reader:
        for unique_id, line in enumerate(reader):
            line = tokenization.convert_to_unicode(line.strip())
            text, mask_idx = line.split()
            examples.append(InputExample(unique_id=unique_id, text=text, mask_idx=int(mask_idx)))
    return examples


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        tpu_config=tf.contrib.tpu.TPUConfig(num_shards=FLAGS.num_tpu_cores, per_host_input_for_training=is_per_host))

    examples = read_examples(FLAGS.input_file)
    features = convert_examples_to_features(examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(bert_config=bert_config,
                                init_checkpoint=FLAGS.init_checkpoint,
                                use_tpu=FLAGS.use_tpu,
                                use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.batch_size,
        predict_batch_size=FLAGS.batch_size)

    input_fn = input_fn_builder(features=features, seq_length=FLAGS.max_seq_length)

    for result in estimator.predict(input_fn, yield_single_examples=True):
        top_tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in result['predicted_index']]
        unique_id = result['unique_id']
        target_idx = result['target_idx']
        prediction = {'top_tokens': top_tokens, 'unique_id': unique_id, 'target_idx': target_idx}
        print(prediction)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    tf.app.run()
