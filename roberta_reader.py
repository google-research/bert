import random

import numpy as np
import tensorflow as tf
import collections
import tokenization
import random
import pygtrie

WORD_CNT = -1


def parse_file(filename):
    res = []
    with open(filename, 'r', encoding='utf-8') as f:
        doc = []
        for line in f.readline():
            line = tokenization.convert_to_unicode(line)
            tokens = line.split(" ")
            if len(tokens) < 1:
                continue
            else:
                if WORD_CNT < 0:
                    res.append(tokens)
                else:
                    doc.append(tokens)
                    if len(doc) > WORD_CNT:
                        res.append(doc)
                        doc = []
        # end of file, what ever remaining.
        res.append(doc)
    return res


def tf_parse_file(filename):
    return tf.py_func(parse_file, [filename], [tf.string])


class GreedyTokenizer:
    """
    This tokenzier use the longest match to segmented token into smaller pieces so that
    bert model can consume. The implementation is using trie to build the np ndarray
    input for pipeline to digest for efficiency.
    """
    assert sys.version.startswith('3')

    def __init__(self, vocab_file):
        self.unk_idx = -1
        vocab = collections.defaultdict(lambda: [-1, -1])
        index = 0
        with open(vocab_file, "r") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                if token == "[UNK]":
                    self.unk_idx = index
                lidx = 0
                if token.startswith("##"):
                    token = token[2:]
                    lidx = 1
                vocab[token][lidx] = index
                index += 1
        assert self.unk_idx != -1
        self.trie = pygtrie.CharTrie()
        for key in vocab.keys():
            self.trie[key] = tuple(vocab[key])
        self.vocab_size = len(vocab.keys())

    def _tokenize_one(self, token):
        res = []
        start = 0
        lidx = 0
        while start < len(token):
            tmp = token[start:]
            key, value = self.trie.longest_prefix(tmp)
            if key == None:
                start += 1
                lidx = 1
                res.append(self.unk_idx)
            else:
                res.append(value[lidx])
                start += len(key)
                lidx = 1
        return res

    def tokenize(self, text):
        split_tokens = []
        for token in text:
            self._tokenize_one(token, split_tokens)
        return split_tokens


class IdMasker:
    """
    This class uses the trie to further split the tokens into word piece and returns
    the id for it so that it can be used for down stream pipeline. It does whole word
    masking along the way.
    """
    def __init__(self, args):
        print("vocab file:" + args.vocab_file)
        self.tokenizer = GreedyTokenizer(vocab_file=args.vocab_file)
        self.max_seq_length = args.max_seq_length
        self.mask_prob = args.mask_prob
        self.max_mask_length = int(args.mask_prob * self.max_seq_length)
        self.index_of_mask = 4
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, tf_text_batch):
        text_batch = tf_text_batch.numpy()

        """Encode the input based on wordpiece into features need by later pipeline"""
        input_idss = tf.zeros((len(text_batch), self.max_seq_length), dtype=np.int32)
        input_masks = np.zeros((len(text_batch), self.max_seq_length), dtype=np.int32)
        segment_idss = np.zeros((len(text_batch), self.max_seq_length), dtype=np.int32)
        mask_pos = np.zeros((len(text_batch), self.max_mask_length), dtype=np.int32)
        mask_label = np.zeros((len(text_batch), self.max_mask_length), dtype=np.int32)
        mask_weight = np.zeros((len(text_batch), self.max_mask_length), dtype=np.float32)
        seq_length = np.zeros((len(text_batch)), dtype=np.int32)

        for batch_idx in range(len(text_batch)):
            tokenss = self.tokenizer.tokenize(text_batch[batch_idx])

            maskpos = []
            maskids = []

            # CLS = 2
            input_idss[batch_idx][0] = 2
            input_masks[batch_idx][0] = 1
            segment_idss[batch_idx][0] = 0
            pos_idx = 1

            # first sentence.
            for tokens in tokenss:
                # we do sampling here for whole word masking.
                # first decide whether we want to mask here, based random selection and make sure
                # it does not go overboard: beyond max_mask_length
                mask_word = False
                if random.random() < 0.15 and len(maskids) + len(tokens) > self.max_mask_length:
                    mask_word = True
                for token in tokens:
                    input_idss[batch_idx][pos_idx] = token
                    input_masks[batch_idx][pos_idx] = 1
                    segment_idss[batch_idx][pos_idx] = 0

                    # handle if the whole word to be masked.
                    if mask_word:
                        # 80% of the time, replace with [MASK]
                        if random.random() < 0.8:
                            maskidx = self.index_of_mask
                        else:
                            if random.random() < 0.5:
                                maskidx = token
                            else:
                                maskidx = random.randint(0, self.vocab_size - 5) + 5

                        maskpos.append(pos_idx)
                        maskids.append(token)
                        input_idss[batch_idx][pos_idx] = maskidx

                    pos_idx += 1

            # SEP = 3
            input_idss[batch_idx][pos_idx] = 3
            segment_idss[batch_idx][pos_idx] = 0
            input_masks[batch_idx][pos_idx] = 1
            pos_idx += 1

            seq_length[batch_idx] = pos_idx

            for i in range(len(maskpos)):
                mask_pos[batch_idx][i] = maskpos[i]
                mask_label[batch_idx][i] = maskids[i]
                mask_weight[batch_idx][i] = 1.0

        return input_idss, input_masks, segment_id, mask_pos, mask_label, mask_weight, seq_length, -1


class RobertaTextReader:
    """
    Use this to generate the whole word masked training example for later pretraining.
    """
    def __init__(self, config):
        # fix the seed
        if args.random_seed != -1:
            random.seed(args.random_seed)
        self.epochs = args.epochs
        self.idmask = IdMasker(config)
        self.io_threads = args.io_threads
        self.num_gpus = args.num_gpus
        self.pad = tf.constant('')
        self.max_seq_length = args.max_seq_length


    def __call__(self, input_files, batch_size, is_training):

        def tf_idmask(text_batch):
            return tf.py_func(self.idmask, [text_batch], [])

        files = tf.data.Dataset.from_tensor_slices(input_files)
        dataset = files.map(tf_parse_file, num_parallel_calls=self.io_threads)
        dataset = dataset.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))

        if is_training:
            dataset = dataset.repeat(config.epochs)
            dataset = dataset.shuffle(buffer_size=100)

        # this is not correct.
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([self.max_seq_length]),
            padding_values=self.pad,
            drop_remainder=True,
        )

        dataset = dataset.map(tf_idmask, num_parallel_calls=self.io_threads)
        self.dataset = dataset.prefetch(self.num_gpus)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, default='/mnt/cephfs/lukai/nlu/bert/data/pre/3',
                        help='an integer for the accumulator')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='an integer for the labels')
    args = parser.parse_args()
    args.io_threads = 16
    args.max_seq_length = 128
    args.random_seed = 12345
    args.masked_prob = 0.15
    args.num_gpus = 1
    args.batch_size = 2
    args.word_cnt = -1
    args.epochs = 3

    input_files = args.input_files.split(',')
    reader = RobertaTextReader(input_files, args, args.batch_size, True)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(reader.table.init)
        sess.run(reader.iterator.initializer)
        iterator = reader.iterator
        features = iterator.get_next()

        output_tokens = features['input_ids']
        input_mask = features['input_mask']
        segment_id = features['segment_ids']
        mask_pos = features['masked_lm_positions']
        mask_label = features['masked_lm_ids']
        mask_weight = features['masked_lm_weights']
        next_sent_label = features['next_sentence_labels']
        seq  = features['seq']

        while True:
            try:
                out, mask, segment, pos, label, weight, next_label = sess.run(
                    [output_tokens, input_mask, segment_id, mask_pos, mask_label, mask_weight, next_sent_label])
                print('-----------output tokens-------------')
                print(seq)
                print(out)
                print(reader.i2w[out])
                print('------------input mask---------------')
                print(mask)
                print('------------segment id---------------')
                print(segment)
                print('-----------mask positions-------------')
                print(pos)
                print('-----------mask labels-----------------')
                print(reader.i2w[label])
                print('-----------mask weight-----------------')
                print(weight)
                print('------------next sentence label--------')
                print(next_label)
                print('\n\n')
            except tf.errors.OutOfRangeError:
                print('end of sequence')
                break
