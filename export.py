from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tqdm import tqdm

import modeling
import tokenization
from run_classifier import flags, FLAGS, InputExample, DataProcessor, create_model

flags.DEFINE_string(
    "model_dir", None,
    "The input data dir. Should contain the .ckpt files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "serving_model_save_path", None,
    "The input serving_model_save_path. Should be used to contain the .pt files (or other data files) "
    "for the task.")


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        # tf.logging.info("*** Features ***")
        # for name in sorted(features.keys()):
        #     tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        output_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=probabilities
        )
        return output_spec

    return model_fn


def serving_input_receiver_fn():
    input_ids = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name='segment_ids')
    label_ids = tf.placeholder(dtype=tf.int64, shape=[None, ], name='unique_ids')

    receive_tensors = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids,
                       'label_ids': label_ids}
    features = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, "label_ids": label_ids}
    return tf.estimator.export.ServingInputReceiver(features, receive_tensors)


class MyProcessor(DataProcessor):

    def get_test_examples(self, data_dir):
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_pred_examples(self, data_dir):
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "pred.tsv")), "pred")

    def get_labels(self):
        """See base class."""
        return ["-1", "0", "1"]

    def create_examples(self, lines, set_type, file_base=True):
        """Creates examples for the training and dev sets. each line is label+\t+text_a+\t+text_b """
        examples = []
        for (i, line) in tqdm(enumerate(lines)):

            if file_base:
                if i == 0:
                    continue

            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            if set_type == "test" or set_type == "pred":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text, label=label))
        return examples


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "setiment": MyProcessor,
    }

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()

    label_list = processor.get_labels()

    run_config = tf.contrib.tpu.RunConfig(model_dir=FLAGS.model_dir)

    num_train_steps = None
    num_warmup_steps = None

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(use_tpu=FLAGS.use_tpu,
                                            model_fn=model_fn,
                                            config=run_config,
                                            predict_batch_size=FLAGS.predict_batch_size,
                                            export_to_tpu=False)

    estimator.export_savedmodel(FLAGS.serving_model_save_path, serving_input_receiver_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("serving_model_save_path")
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    tf.app.run()
