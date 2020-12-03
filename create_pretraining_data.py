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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import, division, print_function

import collections
import random

import tensorflow as tf

import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("output_file", None, "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case",
    True,
    "Whether to lower case the input text. Should be True for uncased " "models and False for cased models.",
)

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer(
    "max_predictions_per_seq",
    20,
    "Maximum number of masked LM predictions per sequence.",
)

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor",
    1,
    "Number of times to duplicate the input data (with different masks).",
)

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob",
    0.0,
    "Probability of creating sequences which are shorter than the " "maximum length.",
)

flags.DEFINE_bool("turn_sep", True, "make turn seperation token")

flags.DEFINE_bool("is_multi", False, "make instance only have more than 3 turns")

flags.DEFINE_bool("for_multi", False, "make instance Session B has only 1 sentence")

flags.DEFINE_float("pair_prob", 0.0, "pair probability")

flags.DEFINE_bool("example_save", True, "save example or not")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(
        self,
        tokens,
        segment_ids,
        masked_lm_positions,
        masked_lm_labels,
        is_random_next,
        turn_ids=None,
    ):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        if FLAGS.turn_sep:
            self.turn_ids = turn_ids

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "turn_ids: %s\n" % (" ".join([str(x) for x in self.turn_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length, max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    f_example = None
    if FLAGS.example_save:
        name_split = output_files[0].split("/")
        example_file_path = "/".join(name_split[:-2])
        example_file_path += "/examples/" + name_split[-1] + ".txt"
        tf.logging.info(f"Example_file_path : {example_file_path}")
        f_example = open(example_file_path, "w")

    writer_index = 0

    total_written = 0
    negative_count = 0

    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        if FLAGS.turn_sep:
            turn_ids = list(instance.turn_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            if FLAGS.turn_sep:
                turn_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if FLAGS.turn_sep:
            assert len(turn_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])
        if FLAGS.turn_sep:
            features["turn_ids"] = create_int_feature(turn_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1
        negative_count += next_sentence_label

        if inst_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in instance.tokens]))

            if FLAGS.example_save:
                f_example.write("tokens: %s" % " ".join([tokenization.printable_text(x) for x in instance.tokens]))
                f_example.write("\n")

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info("%s: %s" % (feature_name, " ".join([str(x) for x in values])))

                if FLAGS.example_save:
                    f_example.write("%s: %s" % (feature_name, " ".join([str(x) for x in values])))
                    f_example.write("\n")

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)
    tf.logging.info("Wrote %d Positive Examples", total_written - negative_count)
    tf.logging.info("Wrote %d Negative Examples", negative_count)

    if FLAGS.example_save:
        f_example.write("Wrote %d total instances" % int(total_written))
        f_example.write("\n")
        f_example.write("Wrote %d Positive Examples" % (total_written - negative_count))
        f_example.write("\n")
        f_example.write("Wrote %d Negative Examples" % int(negative_count))
        f_example.write("\n")
        f_example.close()


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(
    input_files,
    tokenizer,
    max_seq_length,
    dupe_factor,
    short_seq_prob,
    masked_lm_prob,
    max_predictions_per_seq,
    rng,
    turn_sep,
    for_multi,
):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents,
                    document_index,
                    max_seq_length,
                    short_seq_prob,
                    masked_lm_prob,
                    max_predictions_per_seq,
                    vocab_words,
                    rng,
                    turn_sep,
                    for_multi,
                )
            )

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
    all_documents,
    document_index,
    max_seq_length,
    short_seq_prob,
    masked_lm_prob,
    max_predictions_per_seq,
    vocab_words,
    rng,
    turn_sep,
    for_multi,
):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:

            if turn_sep:
                turn_ids = []
                current_turn = 0

            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                    if turn_sep:
                        turn_length = len(current_chunk[j]) + 1
                        tokens_a.append("[SEPT]")
                        turn_ids.extend([current_turn] * turn_length)
                        current_turn = 1 if current_turn == 0 else 0

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    while True:
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    # random_start = rng.randint(0, len(random_document) - 1)
                    random_start = 0

                    segment_b_length = len(random_document)
                    if for_multi:
                        segment_b_length = random_start + 1

                    for j in range(random_start, segment_b_length):
                        tokens_b.extend(random_document[j])

                        if turn_sep:
                            turn_length = len(random_document[j]) + 1
                            tokens_b.append("[SEPT]")
                            turn_ids.extend([current_turn] * turn_length)
                            current_turn = 1 if current_turn == 0 else 0

                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    segment_b_length = len(current_chunk)
                    if for_multi:
                        segment_b_length = a_end + 1

                    is_random_next = False
                    for j in range(a_end, segment_b_length):
                        tokens_b.extend(current_chunk[j])

                        if turn_sep:
                            turn_length = len(current_chunk[j]) + 1
                            tokens_b.append("[SEPT]")
                            turn_ids.extend([current_turn] * turn_length)
                            current_turn = 1 if current_turn == 0 else 0

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, turn_ids)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)

                if turn_sep:
                    if turn_ids[0] == 1:
                        for k, turn_id in enumerate(turn_ids):
                            turn_ids[k] = 1 if turn_id == 0 else 0
                    turn_ids = [turn_ids[0]] + turn_ids

                a_idx = 1
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                    a_idx += 1

                if turn_sep:
                    turn_ids = turn_ids[:a_idx] + [turn_ids[a_idx - 1]] + turn_ids[a_idx:]

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)
                turn_ids.append(turn_ids[-1])

                (
                    tokens,
                    masked_lm_positions,
                    masked_lm_labels,
                ) = create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)

                if turn_sep:
                    instance = TrainingInstance(
                        tokens=tokens,
                        segment_ids=segment_ids,
                        turn_ids=turn_ids,
                        is_random_next=is_random_next,
                        masked_lm_positions=masked_lm_positions,
                        masked_lm_labels=masked_lm_labels,
                    )
                else:
                    instance = TrainingInstance(
                        tokens=tokens,
                        segment_ids=segment_ids,
                        is_random_next=is_random_next,
                        masked_lm_positions=masked_lm_positions,
                        masked_lm_labels=masked_lm_labels,
                    )
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]" or token == "[SEPT]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, turn_ids):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if len(tokens_a) > len(tokens_b):
            assert len(tokens_a) >= 1
            del tokens_a[0]
            del turn_ids[0]
        else:
            assert len(tokens_b) >= 1
            tokens_b.pop()
            turn_ids.pop()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
        input_files,
        tokenizer,
        FLAGS.max_seq_length,
        FLAGS.dupe_factor,
        FLAGS.short_seq_prob,
        FLAGS.masked_lm_prob,
        FLAGS.max_predictions_per_seq,
        rng,
    )

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(
        instances,
        tokenizer,
        FLAGS.max_seq_length,
        FLAGS.max_predictions_per_seq,
        output_files,
    )


# for multi processing
def one_process(
    input_file,
    output_file,
    vocab,
    max_seq_length=128,
    do_lower_case=False,
    dupe_factor=1,
    short_seq_prob=0.0,
    masked_lm_prob=0.15,
    max_predictions_per_seq=20,
    random_seed=12345,
    turn_sep=FLAGS.turn_sep,
    for_multi=FLAGS.for_multi,
):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=do_lower_case)

    input_files = []
    for input_pattern in input_file:
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    rng = random.Random(random_seed)
    instances = create_training_instances(
        input_files,
        tokenizer,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        rng,
        turn_sep,
        for_multi,
    )

    output_files = output_file
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, max_seq_length, max_predictions_per_seq, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
