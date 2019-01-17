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

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import init_ops
import numpy
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn

def fused_layer_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               begin_norm_axis=1,
               begin_params_axis=-1,
               scope=None,
               use_fused_batch_norm=False):
  with tf.variable_scope(
      scope, 'LayerNorm', [inputs], reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.shape
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if begin_norm_axis < 0:
      begin_norm_axis = inputs_rank + begin_norm_axis
    if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
      raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                       'must be < rank(inputs) (%d)' %
                       (begin_params_axis, begin_norm_axis, inputs_rank))
    params_shape = inputs_shape[begin_params_axis:]
    if not params_shape.is_fully_defined():
      raise ValueError(
          'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
          (inputs.name, begin_params_axis, inputs_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta = variables.model_variable(
          'beta',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.zeros_initializer(),
          collections=beta_collections,
          trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(
          variables_collections, 'gamma')
      gamma = variables.model_variable(
          'gamma',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.ones_initializer(),
          collections=gamma_collections,
          trainable=trainable)
    if use_fused_batch_norm:
      # get static TensorShape if fully defined,
      # otherwise retrieve shape tensor
      norm_shape = inputs.shape[begin_norm_axis:]
      if norm_shape.is_fully_defined():
        bn_shape = [1, -1, 1, numpy.prod(norm_shape.as_list())]
      else:
        norm_shape = tf.shape(inputs)[begin_norm_axis:]
        bn_shape = [1, -1, 1, tf.reduce_prod(norm_shape)]
      if inputs.get_shape().is_fully_defined():
        outputs_shape = inputs.get_shape()
      else:
        outputs_shape = tf.shape(inputs)
      inputs = array_ops.reshape(inputs, bn_shape)
      if inputs.get_shape().is_fully_defined():
        # static inputs TensorShape fully defined after reshape.
        ones = array_ops.ones(inputs.get_shape()[1], dtype=dtypes.float32)
        zeros = array_ops.zeros(inputs.get_shape()[1], dtype=dtypes.float32)
      else:
        # static inputs TensorShape NOT fully defined after reshape.
        # must use dynamic shape, which means these input tensors
        # have to be created at runtime, which causes a slowdown.
        scale_shape = tf.shape(inputs)[1]
        ones = array_ops.ones(scale_shape, dtype=dtypes.float32)
        zeros = array_ops.zeros(scale_shape, dtype=dtypes.float32)
      outputs, mean, variance = nn.fused_batch_norm(
          inputs,
          ones, zeros,
          epsilon=1e-4,
          data_format="NCHW")
      outputs = array_ops.reshape(outputs, outputs_shape)
      if center and scale:
        outputs = outputs * gamma + beta
      elif center:
        outputs = outputs + beta
      elif scale:
        outputs = outputs * gamma
    else:
      # Calculate the moments on the last axis (layer activations).
      norm_axes = list(range(begin_norm_axis, inputs_rank))
      mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)
      # Compute layer normalization using the batch_normalization function.
      variance_epsilon = 1e-4
      outputs = nn.batch_normalization(
          inputs,
          mean,
          variance,
          offset=beta,
          scale=gamma,
          variance_epsilon=variance_epsilon)
      outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

