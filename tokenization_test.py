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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import tokenization
import six
import tensorflow as tf


class TokenizationTest(tf.test.TestCase):

  def test_full_tokenizer(self):
    vocab_tokens = [
        "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
        "##ing", ","
    ]
    with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
      if six.PY2:
        vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
      else:
        vocab_writer.write("".join(
            [x + "\n" for x in vocab_tokens]).encode("utf-8"))

      vocab_file = vocab_writer.name

    tokenizer = tokenization.FullTokenizer(vocab_file)
    os.unlink(vocab_file)

    tokens = tokenizer.tokenize(u"UNwant\u00E9d,running")
    self.assertAllEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])

    self.assertAllEqual(
        tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

  def test_chinese(self):
    tokenizer = tokenization.BasicTokenizer()

    self.assertAllEqual(
        tokenizer.tokenize(u"ah\u535A\u63A8zz"),
        [u"ah", u"\u535A", u"\u63A8", u"zz"])

  def test_basic_tokenizer_lower(self):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=True)

    self.assertAllEqual(
        tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
        ["hello", "!", "how", "are", "you", "?"])
    self.assertAllEqual(tokenizer.tokenize(u"H\u00E9llo"), ["hello"])

  def test_basic_tokenizer_no_lower(self):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

    self.assertAllEqual(
        tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
        ["HeLLo", "!", "how", "Are", "yoU", "?"])

  def test_wordpiece_tokenizer(self):
    vocab_tokens = [
        "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
        "##ing"
    ]

    vocab = {}
    for (i, token) in enumerate(vocab_tokens):
      vocab[token] = i
    tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

    self.assertAllEqual(tokenizer.tokenize(""), [])

    self.assertAllEqual(
        tokenizer.tokenize("unwanted running"),
        ["un", "##want", "##ed", "runn", "##ing"])

    self.assertAllEqual(
        tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

 def test_is_whitespace(self):
    whitespace_chars = [u" ", u"\t", u"\r", u"\n", u"\u00A0"]
    non_whitespace_chars = [u"A", u"-"]

    for char in whitespace_chars:
        self.assertTrue(tokenization._is_whitespace(char))

    for char in non_whitespace_chars:
        self.assertFalse(tokenization._is_whitespace(char))

def test_is_control(self):
    control_chars = [u"\u0005"]
    non_control_chars = [u"A", u" ", u"\t", u"\r", u"\U0001F4A9"]

    for char in control_chars:
        self.assertTrue(tokenization._is_control(char))

    for char in non_control_chars:
        self.assertFalse(tokenization._is_control(char))

def test_is_punctuation(self):
    punctuation_chars = [u"-", u"$", u"`", u"."]
    non_punctuation_chars = [u"A", u" "]

    for char in punctuation_chars:
        self.assertTrue(tokenization._is_punctuation(char))

    for char in non_punctuation_chars:
        self.assertFalse(tokenization._is_punctuation(char))


if __name__ == "__main__":
  tf.test.main()
