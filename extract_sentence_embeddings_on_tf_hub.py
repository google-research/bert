"""This file outputs a representations for every token in
the input sequence and a pooled representation of the entire input.


Example usage(for getting sentence embeddings) -

$python bert_sentence.py \
> --input_file=path/to/input_data.txt \
> --output_file=path/to/output_vectors \
> --bert_hub_module=https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1 \
> --maximum_seq_length=128 \
> --sentence_embedding=True

Output will be saved in 'output_vectors.npy' as numpy.ndarray.
Load sentence embeddings as -

>>import numpy as np
>>sentence_embeddings = np.load('path/to/output_vectors.npy')

NOTE: input file should contain '\n' separated sentences. For eg.
$cat path/to/input_data.txt
hello how are you
what is your name

output vectors will have same index as input sentences
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from run_classifier_with_tfhub import create_tokenizer_from_hub_module
import run_classifier

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("bert_hub_module",
                    "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1",
                    "URL for tf-hub BERT module(for eg. https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1)")
flags.DEFINE_string("input_file", None, "Input file containing file to generate vectors from")

flags.DEFINE_string("output_file", None, "File to save vectors into(.npy)")

flags.DEFINE_integer(
    "maximum_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "sentence_embedding", True,
    " True if sentence embedding required else False")


def main():
    inputs = []
    with open(FLAGS.input_file, 'r') as inputfile:
        for line in inputfile:
            inputs.append(line)

    tokenizer = create_tokenizer_from_hub_module(FLAGS.bert_hub_module)
    input_examples = []
    for index, example in enumerate(inputs):
        input_examples.append(
            run_classifier.InputExample(
                index, example, text_b=None, label=0))

    input_features = run_classifier.convert_examples_to_features(
        input_examples, [0, 1], FLAGS.maximum_seq_length, tokenizer)

    input_ids = list(map(lambda x: x.input_ids, input_features))
    input_mask = list(map(lambda x: x.input_mask, input_features))
    segment_ids = list(map(lambda x: x.segment_ids, input_features))
    bert_module = hub.Module(FLAGS.bert_hub_module, trainable=True)
    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)
    bert_outputs = bert_module(bert_inputs, signature="tokens", as_dict=True)
    if FLAGS.sentence_embedding:
        output_embeddings = bert_outputs["pooled_output"]
    else:
        output_embeddings = bert_outputs["sequence_output"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(output_embeddings)
        np.save(FLAGS.output_file.split(".")[0], result)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    main()
