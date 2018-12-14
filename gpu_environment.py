# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import fp16_utils
import tensorflow as tf

FLAGS = tf.flags.FLAGS

custom_getter = fp16_utils.float32_variable_storage_getter if FLAGS.use_fp16 else None
compute_type = tf.float16 if FLAGS.use_fp16 else tf.float32
