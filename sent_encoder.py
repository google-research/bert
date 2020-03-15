from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tokenization
import numpy as np
import modeling
import time


maximum_seq_length = 64
sentence_embedding = True
#init_checkpoint = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
init_checkpoint = 'aimind/bert_tf.ckpt'

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputProcessor():
        
    def get_test_examples(self,text1,text2):
        data_df1 = text1
        data_df2 = text2
        return self._create_examples(data_df1,data_df2, "test")
        

    
    def _create_examples(self, df1,df2):
        """Creates examples for the training and dev sets."""
        examples = []
        text_a = df1
        text_b = df2
        examples.append(
            InputExample(text_a=text_a,text_b=text_b))
        return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

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

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features

def extract_features(text):
    inputs = [text]
    tokenizer = tokenization.FullTokenizer(vocab_file='./uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
    bert_config = modeling.BertConfig.from_json_file('./uncased_L-12_H-768_A-12/bert_config.json')
    input_examples = []
    for index, example in enumerate(inputs):
        input_examples.append(
            InputExample(index, example, text_b=None, label=0))

    input_features = convert_examples_to_features(
        input_examples, [0, 1], maximum_seq_length, tokenizer)

    
    input_ids = list(map(lambda x: x.input_ids, input_features))
    input_mask = list(map(lambda x: x.input_mask, input_features))
    segment_ids = list(map(lambda x: x.segment_ids, input_features))

    input_ids = tf.convert_to_tensor(input_ids)
    input_mask = tf.convert_to_tensor(segment_ids)
    segment_ids = tf.convert_to_tensor(input_mask)
    
    model = modeling.BertModel(config=bert_config, is_training=True,input_ids=input_ids, input_mask=input_mask, token_type_ids=segment_ids)

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    if sentence_embedding:
        output_embeddings = model.get_pooled_output()
    else:
        output_embeddings = model.get_sequence_output()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(output_embeddings)
        return (result)
      
if __name__ == "__main__":
    text = "etherlabs is a collaborative service for machine learning"
    sent_emb = extract_features(text)
    print (sent_emb[0])
