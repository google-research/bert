from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf

def get_shape_list(tensor, expected_rank=None, name=None):
	"""Returns a list of the shape of tensor, preferring static dimensions.

	Args:
		tensor: A tf.Tensor object to find the shape of.
		expected_rank: (optional) int. The expected rank of `tensor`. If this is
			specified and the `tensor` has a different rank, and exception will be
			thrown.
		name: Optional name of the tensor for the error message.

	Returns:
		A list of dimensions of the shape of tensor. All static dimensions will
		be returned as python integers, and dynamic dimensions will be returned
		as tf.Tensor scalars.
	"""
	if name is None:
		name = tensor.name

	if expected_rank is not None:
		assert_rank(tensor, expected_rank, name)

	shape = tensor.shape.as_list()

	non_static_indexes = []
	for (index, dim) in enumerate(shape):
		if dim is None:
			non_static_indexes.append(index)

	if not non_static_indexes:
		return shape

	dyn_shape = tf.shape(tensor)
	for index in non_static_indexes:
		shape[index] = dyn_shape[index]
	return shape

def reshape_to_matrix(input_tensor):
	"""Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
	ndims = input_tensor.shape.ndims
	if ndims < 2:
		raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
										 (input_tensor.shape))
	if ndims == 2:
		return input_tensor

	width = input_tensor.shape[-1]
	output_tensor = tf.reshape(input_tensor, [-1, width])
	return output_tensor

def reshape_from_matrix(output_tensor, orig_shape_list):
	"""Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
	if len(orig_shape_list) == 2:
		return output_tensor

	output_shape = get_shape_list(output_tensor)

	orig_dims = orig_shape_list[0:-1]
	width = output_shape[-1]

	return tf.reshape(output_tensor, orig_dims + [width])

def assert_rank(tensor, expected_rank, name=None):
	"""Raises an exception if the tensor rank is not of the expected rank.

	Args:
		tensor: A tf.Tensor to check the rank of.
		expected_rank: Python integer or list of integers, expected rank.
		name: Optional name of the tensor for the error message.

	Raises:
		ValueError: If the expected shape doesn't match the actual shape.
	"""
	if name is None:
		name = tensor.name

	expected_rank_dict = {}
	if isinstance(expected_rank, six.integer_types):
		expected_rank_dict[expected_rank] = True
	else:
		for x in expected_rank:
			expected_rank_dict[x] = True

	actual_rank = tensor.shape.ndims
	if actual_rank not in expected_rank_dict:
		scope_name = tf.get_variable_scope().name
		raise ValueError(
				"For the tensor `%s` in scope `%s`, the actual rank "
				"`%d` (shape = %s) is not equal to the expected rank `%s`" %
				(name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def gather_indexes(sequence_tensor, positions):
	"""Gathers the vectors at the specific positions over a minibatch."""
	sequence_shape = get_shape_list(sequence_tensor, expected_rank=3)
	batch_size = sequence_shape[0]
	seq_length = sequence_shape[1]
	width = sequence_shape[2]

	flat_offsets = tf.reshape(
			tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
	flat_positions = tf.reshape(positions + flat_offsets, [-1])
	flat_sequence_tensor = tf.reshape(sequence_tensor,
																		[batch_size * seq_length, width])
	output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
	return output_tensor

# add sequence mask for:
# 1. random shuffle lm modeling---xlnet with random shuffled input
# 2. left2right and right2left language modeling
# 3. conditional generation
def generate_seq2seq_mask(attention_mask, mask_sequence, seq_type, **kargs):
	if seq_type == 'seq2seq':
		if mask_sequence is not None:
			seq_shape = get_shape_list(mask_sequence, expected_rank=2)
			seq_len = seq_shape[1]
			ones = tf.ones((1, seq_len, seq_len))
			a_mask = tf.matrix_band_part(ones, -1, 0)
			s_ex12 = tf.expand_dims(tf.expand_dims(mask_sequence, 1), 2)
			s_ex13 = tf.expand_dims(tf.expand_dims(mask_sequence, 1), 3)
			a_mask = (1 - s_ex13) * (1 - s_ex12) + s_ex13 * a_mask
			# generate mask of batch x seq_len x seq_len
			a_mask = tf.reshape(a_mask, (-1, seq_len, seq_len))
			out_mask = attention_mask * a_mask
		else:
			ones = tf.ones_like(attention_mask[:1])
			mask = (tf.matrix_band_part(ones, -1, 0))
			out_mask = attention_mask * mask
	else:
		out_mask = attention_mask

	return out_mask

