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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf
import jieba
import re
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

flags.DEFINE_bool("non_chinese", False,"manually set this to True if you are not doing chinese pre-train task.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

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

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
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
        strings=reader.readline()
        strings=strings.replace("   "," ").replace("  "," ") # 如果有两个或三个空格，替换为一个空格
        line = tokenization.convert_to_unicode(strings)
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
        create_instances_from_document_albert( # change to albert style for sentence order prediction(SOP), 2019-08-28, brightmart
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  rng.shuffle(instances)
  return instances

def get_new_segment(segment):  # 新增的方法 ####
    """
    输入一句话，返回一句经过处理的话: 为了支持中文全称mask，将被分开的词，将上特殊标记("#")，使得后续处理模块，能够知道哪些字是属于同一个词的。
    :param segment: 一句话. e.g.  ['悬', '灸', '技', '术', '培', '训', '专', '家', '教', '你', '艾', '灸', '降', '血', '糖', '，', '为', '爸', '妈', '收', '好', '了', '！']
    :return: 一句处理过的话 e.g.    ['悬', '##灸', '技', '术', '培', '训', '专', '##家', '教', '你', '艾', '##灸', '降', '##血', '##糖', '，', '为', '爸', '##妈', '收', '##好', '了', '！']
    """
    seq_cws = jieba.lcut("".join(segment)) # 分词
    seq_cws_dict = {x: 1 for x in seq_cws} # 分词后的词加入到词典dict
    new_segment = []
    i = 0
    while i < len(segment): # 从句子的第一个字开始处理，知道处理完整个句子
      if len(re.findall('[\u4E00-\u9FA5]', segment[i])) == 0:  # 如果找不到中文的，原文加进去即不用特殊处理。
        new_segment.append(segment[i])
        i += 1
        continue

      has_add = False
      for length in range(3, 0, -1):
        if i + length > len(segment):
          continue
        if ''.join(segment[i:i + length]) in seq_cws_dict:
          new_segment.append(segment[i])
          for l in range(1, length):
            new_segment.append('##' + segment[i + l])
          i += length
          has_add = True
          break
      if not has_add:
        new_segment.append(segment[i])
        i += 1
    # print("get_new_segment.wwm.get_new_segment:",new_segment)
    return new_segment

def create_instances_from_document_albert(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document.
     This method is changed to create sentence-order prediction (SOP) followed by idea from paper of ALBERT, 2019-08-28, brightmart
  """
  document = all_documents[document_index] # 得到一个文档

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
  if rng.random() < short_seq_prob: # 有一定的比例，如10%的概率，我们使用比较短的序列长度，以缓解预训练的长序列和调优阶段（可能的）短序列的不一致情况
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  # 设法使用实际的句子，而不是任意的截断句子，从而更好的构造句子连贯性预测的任务
  instances = []
  current_chunk = [] # 当前处理的文本段，包含多个句子
  current_length = 0
  i = 0
  # print("###document:",document) # 一个document可以是一整篇文章、新闻、词条等. document:[['是', '爷', '们', '，', '就', '得', '给', '媳', '妇', '幸', '福'], ['关', '注', '【', '晨', '曦', '教', '育', '】', '，', '获', '取', '育', '儿', '的', '智', '慧', '，', '与', '孩', '子', '一', '同', '成', '长', '！'], ['方', '法', ':', '打', '开', '微', '信', '→', '添', '加', '朋', '友', '→', '搜', '号', '→', '##he', '##bc', '##x', '##jy', '##→', '关', '注', '!', '我', '是', '一', '个', '爷', '们', '，', '孝', '顺', '是', '做', '人', '的', '第', '一', '准', '则', '。'], ['甭', '管', '小', '时', '候', '怎', '么', '跟', '家', '长', '犯', '混', '蛋', '，', '长', '大', '了', '，', '就', '底', '报', '答', '父', '母', '，', '以', '后', '我', '媳', '妇', '也', '必', '须', '孝', '顺', '。'], ['我', '是', '一', '个', '爷', '们', '，', '可', '以', '花', '心', '，', '可', '以', '好', '玩', '。'], ['但', '我', '一', '定', '会', '找', '一', '个', '管', '的', '住', '我', '的', '女', '人', '，', '和', '我', '一', '起', '生', '活', '。'], ['28', '岁', '以', '前', '在', '怎', '么', '玩', '都', '行', '，', '但', '我', '最', '后', '一', '定', '会', '找', '一', '个', '勤', '俭', '持', '家', '的', '女', '人', '。'], ['我', '是', '一', '爷', '们', '，', '我', '不', '会', '让', '自', '己', '的', '女', '人', '受', '一', '点', '委', '屈', '，', '每', '次', '把', '她', '抱', '在', '怀', '里', '，', '看', '她', '洋', '溢', '着', '幸', '福', '的', '脸', '，', '我', '都', '会', '引', '以', '为', '傲', '，', '这', '特', '么', '就', '是', '我', '的', '女', '人', '。'], ['我', '是', '一', '爷', '们', '，', '干', '什', '么', '也', '不', '能', '忘', '了', '自', '己', '媳', '妇', '，', '就', '算', '和', '哥', '们', '一', '起', '喝', '酒', '，', '喝', '到', '很', '晚', '，', '也', '要', '提', '前', '打', '电', '话', '告', '诉', '她', '，', '让', '她', '早', '点', '休', '息', '。'], ['我', '是', '一', '爷', '们', '，', '我', '媳', '妇', '绝', '对', '不', '能', '抽', '烟', '，', '喝', '酒', '还', '勉', '强', '过', '得', '去', '，', '不', '过', '该', '喝', '的', '时', '候', '喝', '，', '不', '该', '喝', '的', '时', '候', '，', '少', '扯', '纳', '极', '薄', '蛋', '。'], ['我', '是', '一', '爷', '们', '，', '我', '媳', '妇', '必', '须', '听', '我', '话', '，', '在', '人', '前', '一', '定', '要', '给', '我', '面', '子', '，', '回', '家', '了', '咱', '什', '么', '都', '好', '说', '。'], ['我', '是', '一', '爷', '们', '，', '就', '算', '难', '的', '吃', '不', '上', '饭', '了', '，', '都', '不', '张', '口', '跟', '媳', '妇', '要', '一', '分', '钱', '。'], ['我', '是', '一', '爷', '们', '，', '不', '管', '上', '学', '还', '是', '上', '班', '，', '我', '都', '会', '送', '媳', '妇', '回', '家', '。'], ['我', '是', '一', '爷', '们', '，', '交', '往', '不', '到', '1', '年', '，', '绝', '对', '不', '会', '和', '媳', '妇', '提', '过', '分', '的', '要', '求', '，', '我', '会', '尊', '重', '她', '。'], ['我', '是', '一', '爷', '们', '，', '游', '戏', '永', '远', '比', '不', '上', '我', '媳', '妇', '重', '要', '，', '只', '要', '媳', '妇', '发', '话', '，', '我', '绝', '对', '唯', '命', '是', '从', '。'], ['我', '是', '一', '爷', '们', '，', '上', 'q', '绝', '对', '是', '为', '了', '等', '媳', '妇', '，', '所', '有', '暧', '昧', '的', '心', '情', '只', '为', '她', '一', '个', '女', '人', '而', '写', '，', '我', '不', '一', '定', '会', '经', '常', '写', '日', '志', '，', '可', '是', '我', '会', '告', '诉', '全', '世', '界', '，', '我', '很', '爱', '她', '。'], ['我', '是', '一', '爷', '们', '，', '不', '一', '定', '要', '经', '常', '制', '造', '浪', '漫', '、', '偶', '尔', '过', '个', '节', '日', '也', '要', '送', '束', '玫', '瑰', '花', '给', '媳', '妇', '抱', '回', '家', '。'], ['我', '是', '一', '爷', '们', '，', '手', '机', '会', '24', '小', '时', '为', '她', '开', '机', '，', '让', '她', '半', '夜', '痛', '经', '的', '时', '候', '，', '做', '恶', '梦', '的', '时', '候', '，', '随', '时', '可', '以', '联', '系', '到', '我', '。'], ['我', '是', '一', '爷', '们', '，', '我', '会', '经', '常', '带', '媳', '妇', '出', '去', '玩', '，', '她', '不', '一', '定', '要', '和', '我', '所', '有', '的', '哥', '们', '都', '认', '识', '，', '但', '见', '面', '能', '说', '的', '上', '话', '就', '行', '。'], ['我', '是', '一', '爷', '们', '，', '我', '会', '和', '媳', '妇', '的', '姐', '妹', '哥', '们', '搞', '好', '关', '系', '，', '让', '她', '们', '相', '信', '我', '一', '定', '可', '以', '给', '我', '媳', '妇', '幸', '福', '。'], ['我', '是', '一', '爷', '们', '，', '吵', '架', '后', '、', '也', '要', '主', '动', '打', '电', '话', '关', '心', '她', '，', '咱', '是', '一', '爷', '们', '，', '给', '媳', '妇', '服', '个', '软', '，', '道', '个', '歉', '怎', '么', '了', '？'], ['我', '是', '一', '爷', '们', '，', '绝', '对', '不', '会', '嫌', '弃', '自', '己', '媳', '妇', '，', '拿', '她', '和', '别', '人', '比', '，', '说', '她', '这', '不', '如', '人', '家', '，', '纳', '不', '如', '人', '家', '的', '。'], ['我', '是', '一', '爷', '们', '，', '陪', '媳', '妇', '逛', '街', '时', '，', '碰', '见', '熟', '人', '，', '无', '论', '我', '媳', '妇', '长', '的', '好', '看', '与', '否', '，', '我', '都', '会', '大', '方', '的', '介', '绍', '。'], ['谁', '让', '咱', '爷', '们', '就', '好', '这', '口', '呢', '。'], ['我', '是', '一', '爷', '们', '，', '我', '想', '我', '会', '给', '我', '媳', '妇', '最', '好', '的', '幸', '福', '。'], ['【', '我', '们', '重', '在', '分', '享', '。'], ['所', '有', '文', '字', '和', '美', '图', '，', '来', '自', '网', '络', '，', '晨', '欣', '教', '育', '整', '理', '。'], ['对', '原', '文', '作', '者', '，', '表', '示', '敬', '意', '。'], ['】', '关', '注', '晨', '曦', '教', '育', '[UNK]', '[UNK]', '晨', '曦', '教', '育', '（', '微', '信', '号', '：', 'he', '##bc', '##x', '##jy', '）', '。'], ['打', '开', '微', '信', '，', '扫', '描', '二', '维', '码', '，', '关', '注', '[UNK]', '晨', '曦', '教', '育', '[UNK]', '，', '获', '取', '更', '多', '育', '儿', '资', '源', '。'], ['点', '击', '下', '面', '订', '阅', '按', '钮', '订', '阅', '，', '会', '有', '更', '多', '惊', '喜', '哦', '！']]
  while i < len(document): # 从文档的第一个位置开始，按个往下看
    segment = document[i] # segment是列表，代表的是按字分开的一个完整句子，如 segment=['我', '是', '一', '爷', '们', '，', '我', '想', '我', '会', '给', '我', '媳', '妇', '最', '好', '的', '幸', '福', '。']
    if FLAGS.non_chinese==False: # if non chinese is False, that means it is chinese, then do something to make chinese whole word mask works.
      segment = get_new_segment(segment)  # whole word mask for chinese: 结合分词的中文的whole mask设置即在需要的地方加上“##”

    current_chunk.append(segment) # 将一个独立的句子加入到当前的文本块中
    current_length += len(segment) # 累计到为止位置接触到句子的总长度
    if i == len(document) - 1 or current_length >= target_seq_length:
      # 如果累计的序列长度达到了目标的长度，或当前走到了文档结尾==>构造并添加到“A[SEP]B“中的A和B中；
      if current_chunk: # 如果当前块不为空
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2: # 当前块，如果包含超过两个句子，取当前块的一部分作为“A[SEP]B“中的A部分
          a_end = rng.randint(1, len(current_chunk) - 1)
        # 将当前文本段中选取出来的前半部分，赋值给A即tokens_a
        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        # 构造“A[SEP]B“中的B部分(有一部分是正常的当前文档中的后半部;在原BERT的实现中一部分是随机的从另一个文档中选取的，）
        tokens_b = []
        for j in range(a_end, len(current_chunk)):
          tokens_b.extend(current_chunk[j])

        # 有百分之50%的概率交换一下tokens_a和tokens_b的位置
        # print("tokens_a length1:",len(tokens_a))
        # print("tokens_b length1:",len(tokens_b)) # len(tokens_b) = 0

        if len(tokens_a) == 0 or len(tokens_b) == 0: i += 1; continue
        if rng.random() < 0.5: # 交换一下tokens_a和tokens_b
          is_random_next=True
          temp=tokens_a
          tokens_a=tokens_b
          tokens_b=temp
        else:
          is_random_next=False

        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        # 把tokens_a & tokens_b加入到按照bert的风格，即以[CLS]tokens_a[SEP]tokens_b[SEP]的形式，结合到一起，作为最终的tokens; 也带上segment_ids，前面部分segment_ids的值是0，后面部分的值是1.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        # 创建masked LM的任务的数据 Creates the predictions for the masked LM objective
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance( # 创建训练实例的对象
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = [] # 清空当前块
      current_length = 0 # 重置当前文本块的长度
    i += 1 # 接着文档中的内容往后看

  return instances


def create_instances_from_document_original( # THIS IS ORIGINAL BERT STYLE FOR CREATE DATA OF MLM AND NEXT SENTENCE PREDICTION TASK
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index] # 得到一个文档

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
  if rng.random() < short_seq_prob: # 有一定的比例，如10%的概率，我们使用比较短的序列长度，以缓解预训练的长序列和调优阶段（可能的）短序列的不一致情况
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  # 设法使用实际的句子，而不是任意的截断句子，从而更好的构造句子连贯性预测的任务
  instances = []
  current_chunk = [] # 当前处理的文本段，包含多个句子
  current_length = 0
  i = 0
  # print("###document:",document) # 一个document可以是一整篇文章、新闻、一个词条等. document:[['是', '爷', '们', '，', '就', '得', '给', '媳', '妇', '幸', '福'], ['关', '注', '【', '晨', '曦', '教', '育', '】', '，', '获', '取', '育', '儿', '的', '智', '慧', '，', '与', '孩', '子', '一', '同', '成', '长', '！'], ['方', '法', ':', '打', '开', '微', '信', '→', '添', '加', '朋', '友', '→', '搜', '号', '→', '##he', '##bc', '##x', '##jy', '##→', '关', '注', '!', '我', '是', '一', '个', '爷', '们', '，', '孝', '顺', '是', '做', '人', '的', '第', '一', '准', '则', '。'], ['甭', '管', '小', '时', '候', '怎', '么', '跟', '家', '长', '犯', '混', '蛋', '，', '长', '大', '了', '，', '就', '底', '报', '答', '父', '母', '，', '以', '后', '我', '媳', '妇', '也', '必', '须', '孝', '顺', '。'], ['我', '是', '一', '个', '爷', '们', '，', '可', '以', '花', '心', '，', '可', '以', '好', '玩', '。'], ['但', '我', '一', '定', '会', '找', '一', '个', '管', '的', '住', '我', '的', '女', '人', '，', '和', '我', '一', '起', '生', '活', '。'], ['28', '岁', '以', '前', '在', '怎', '么', '玩', '都', '行', '，', '但', '我', '最', '后', '一', '定', '会', '找', '一', '个', '勤', '俭', '持', '家', '的', '女', '人', '。'], ['我', '是', '一', '爷', '们', '，', '我', '不', '会', '让', '自', '己', '的', '女', '人', '受', '一', '点', '委', '屈', '，', '每', '次', '把', '她', '抱', '在', '怀', '里', '，', '看', '她', '洋', '溢', '着', '幸', '福', '的', '脸', '，', '我', '都', '会', '引', '以', '为', '傲', '，', '这', '特', '么', '就', '是', '我', '的', '女', '人', '。'], ['我', '是', '一', '爷', '们', '，', '干', '什', '么', '也', '不', '能', '忘', '了', '自', '己', '媳', '妇', '，', '就', '算', '和', '哥', '们', '一', '起', '喝', '酒', '，', '喝', '到', '很', '晚', '，', '也', '要', '提', '前', '打', '电', '话', '告', '诉', '她', '，', '让', '她', '早', '点', '休', '息', '。'], ['我', '是', '一', '爷', '们', '，', '我', '媳', '妇', '绝', '对', '不', '能', '抽', '烟', '，', '喝', '酒', '还', '勉', '强', '过', '得', '去', '，', '不', '过', '该', '喝', '的', '时', '候', '喝', '，', '不', '该', '喝', '的', '时', '候', '，', '少', '扯', '纳', '极', '薄', '蛋', '。'], ['我', '是', '一', '爷', '们', '，', '我', '媳', '妇', '必', '须', '听', '我', '话', '，', '在', '人', '前', '一', '定', '要', '给', '我', '面', '子', '，', '回', '家', '了', '咱', '什', '么', '都', '好', '说', '。'], ['我', '是', '一', '爷', '们', '，', '就', '算', '难', '的', '吃', '不', '上', '饭', '了', '，', '都', '不', '张', '口', '跟', '媳', '妇', '要', '一', '分', '钱', '。'], ['我', '是', '一', '爷', '们', '，', '不', '管', '上', '学', '还', '是', '上', '班', '，', '我', '都', '会', '送', '媳', '妇', '回', '家', '。'], ['我', '是', '一', '爷', '们', '，', '交', '往', '不', '到', '1', '年', '，', '绝', '对', '不', '会', '和', '媳', '妇', '提', '过', '分', '的', '要', '求', '，', '我', '会', '尊', '重', '她', '。'], ['我', '是', '一', '爷', '们', '，', '游', '戏', '永', '远', '比', '不', '上', '我', '媳', '妇', '重', '要', '，', '只', '要', '媳', '妇', '发', '话', '，', '我', '绝', '对', '唯', '命', '是', '从', '。'], ['我', '是', '一', '爷', '们', '，', '上', 'q', '绝', '对', '是', '为', '了', '等', '媳', '妇', '，', '所', '有', '暧', '昧', '的', '心', '情', '只', '为', '她', '一', '个', '女', '人', '而', '写', '，', '我', '不', '一', '定', '会', '经', '常', '写', '日', '志', '，', '可', '是', '我', '会', '告', '诉', '全', '世', '界', '，', '我', '很', '爱', '她', '。'], ['我', '是', '一', '爷', '们', '，', '不', '一', '定', '要', '经', '常', '制', '造', '浪', '漫', '、', '偶', '尔', '过', '个', '节', '日', '也', '要', '送', '束', '玫', '瑰', '花', '给', '媳', '妇', '抱', '回', '家', '。'], ['我', '是', '一', '爷', '们', '，', '手', '机', '会', '24', '小', '时', '为', '她', '开', '机', '，', '让', '她', '半', '夜', '痛', '经', '的', '时', '候', '，', '做', '恶', '梦', '的', '时', '候', '，', '随', '时', '可', '以', '联', '系', '到', '我', '。'], ['我', '是', '一', '爷', '们', '，', '我', '会', '经', '常', '带', '媳', '妇', '出', '去', '玩', '，', '她', '不', '一', '定', '要', '和', '我', '所', '有', '的', '哥', '们', '都', '认', '识', '，', '但', '见', '面', '能', '说', '的', '上', '话', '就', '行', '。'], ['我', '是', '一', '爷', '们', '，', '我', '会', '和', '媳', '妇', '的', '姐', '妹', '哥', '们', '搞', '好', '关', '系', '，', '让', '她', '们', '相', '信', '我', '一', '定', '可', '以', '给', '我', '媳', '妇', '幸', '福', '。'], ['我', '是', '一', '爷', '们', '，', '吵', '架', '后', '、', '也', '要', '主', '动', '打', '电', '话', '关', '心', '她', '，', '咱', '是', '一', '爷', '们', '，', '给', '媳', '妇', '服', '个', '软', '，', '道', '个', '歉', '怎', '么', '了', '？'], ['我', '是', '一', '爷', '们', '，', '绝', '对', '不', '会', '嫌', '弃', '自', '己', '媳', '妇', '，', '拿', '她', '和', '别', '人', '比', '，', '说', '她', '这', '不', '如', '人', '家', '，', '纳', '不', '如', '人', '家', '的', '。'], ['我', '是', '一', '爷', '们', '，', '陪', '媳', '妇', '逛', '街', '时', '，', '碰', '见', '熟', '人', '，', '无', '论', '我', '媳', '妇', '长', '的', '好', '看', '与', '否', '，', '我', '都', '会', '大', '方', '的', '介', '绍', '。'], ['谁', '让', '咱', '爷', '们', '就', '好', '这', '口', '呢', '。'], ['我', '是', '一', '爷', '们', '，', '我', '想', '我', '会', '给', '我', '媳', '妇', '最', '好', '的', '幸', '福', '。'], ['【', '我', '们', '重', '在', '分', '享', '。'], ['所', '有', '文', '字', '和', '美', '图', '，', '来', '自', '网', '络', '，', '晨', '欣', '教', '育', '整', '理', '。'], ['对', '原', '文', '作', '者', '，', '表', '示', '敬', '意', '。'], ['】', '关', '注', '晨', '曦', '教', '育', '[UNK]', '[UNK]', '晨', '曦', '教', '育', '（', '微', '信', '号', '：', 'he', '##bc', '##x', '##jy', '）', '。'], ['打', '开', '微', '信', '，', '扫', '描', '二', '维', '码', '，', '关', '注', '[UNK]', '晨', '曦', '教', '育', '[UNK]', '，', '获', '取', '更', '多', '育', '儿', '资', '源', '。'], ['点', '击', '下', '面', '订', '阅', '按', '钮', '订', '阅', '，', '会', '有', '更', '多', '惊', '喜', '哦', '！']]
  while i < len(document): # 从文档的第一个位置开始，按个往下看
    segment = document[i] # segment是列表，代表的是按字分开的一个完整句子，如 segment=['我', '是', '一', '爷', '们', '，', '我', '想', '我', '会', '给', '我', '媳', '妇', '最', '好', '的', '幸', '福', '。']
    # print("###i:",i,";segment:",segment)
    current_chunk.append(segment) # 将一个独立的句子加入到当前的文本块中
    current_length += len(segment) # 累计到为止位置接触到句子的总长度
    if i == len(document) - 1 or current_length >= target_seq_length: # 如果累计的序列长度达到了目标的长度==>构造并添加到“A[SEP]B“中的A和B中。
      if current_chunk: # 如果当前块不为空
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2: # 当前块，如果包含超过两个句子，怎取当前块的一部分作为“A[SEP]B“中的A部分
          a_end = rng.randint(1, len(current_chunk) - 1)
        # 将当前文本段中选取出来的前半部分，赋值给A即tokens_a
        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        # 构造“A[SEP]B“中的B部分(原本的B有一部分是随机的从另一个文档中选取的，有一部分是正常的当前文档中的后半部）
        tokens_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5: # 有50%的概率，是从其他文档中随机的选取一个文档，并得到这个文档的后半版本作为B即tokens_b
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          random_document_index=0
          for _ in range(10): # 随机的选出一个与当前的文档不一样的文档的索引
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index] # 选出这个文档
          random_start = rng.randint(0, len(random_document) - 1) # 从这个文档选出一个段落的开始位置
          for j in range(random_start, len(random_document)): # 从这个文档的开始位置到结束，作为我们的“A[SEP]B“中的B即tokens_b
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste. 这里是为了防止文本的浪费的一个小技巧
          num_unused_segments = len(current_chunk) - a_end # e.g. 550-200=350
          i -= num_unused_segments # i=i-num_unused_segments, e.g. i=400, num_unused_segments=350, 那么 i=i-num_unused_segments=400-350=50
        # Actual next
        else: # 有另外50%的几乎，从当前文本块（长度为max_sequence_length）中的后段中填充到tokens_b即“A[SEP]B“中的B。
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        # 把tokens_a & tokens_b加入到按照bert的风格，即以[CLS]tokens_a[SEP]tokens_b[SEP]的形式，结合到一起，作为最终的tokens; 也带上segment_ids，前面部分segment_ids的值是0，后面部分的值是1.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        # 创建masked LM的任务的数据 Creates the predictions for the masked LM objective
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance( # 创建训练实例的对象
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = [] # 清空当前块
      current_length = 0 # 重置当前文本块的长度
    i += 1 # 接着文档中的内容往后看

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
            token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  if FLAGS.non_chinese==False: # if non chinese is False, that means it is chinese, then try to remove "##" which is added previously
    output_tokens = [t[2:] if len(re.findall('##[\u4E00-\u9FA5]', t)) > 0 else t for t in tokens]  # 去掉"##"
  else: # english and other language, which is not chinese
    output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          if FLAGS.non_chinese == False: # if non chinese is False, that means it is chinese, then try to remove "##" which is added previously
            masked_token = tokens[index][2:] if len(re.findall('##[\u4E00-\u9FA5]', tokens[index])) > 0 else tokens[index]  # 去掉"##"
          else:
            masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  # tf.logging.info('%s' % (tokens))
  # tf.logging.info('%s' % (output_tokens))
  return (output_tokens, masked_lm_positions, masked_lm_labels)

def create_masked_lm_predictions_original(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
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
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()