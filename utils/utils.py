# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf
import time

# report latency and throughput during eval
class LogEvalRunHook(tf.train.SessionRunHook):
  def __init__(self, global_batch_size, hvd_rank=-1):
    self.global_batch_size = global_batch_size
    self.hvd_rank = hvd_rank
    self.total_time = 0.0
    self.count = 0
    self.skipped = 0
    self.time_list = []

  def before_run(self, run_context):
    self.t0 = time.time()

  def after_run(self, run_context, run_values):
    elapsed_secs = time.time() - self.t0
    self.count += 1

    # Removing first 2 (arbitrary) number of startup iterations from perf evaluations
    if self.count <= 2:
      print("Skipping time record for ", self.count, " due to overhead")
      self.skipped += 1
    else:
      self.time_list.append(elapsed_secs)
      self.total_time += elapsed_secs

# report throughput during training
class LogTrainRunHook(tf.train.SessionRunHook):
  def __init__(self, global_batch_size, hvd_rank=-1, save_checkpoints_steps=1000):
    self.global_batch_size = global_batch_size
    self.hvd_rank = hvd_rank
    self.save_checkpoints_steps = save_checkpoints_steps

    self.total_time = 0.0
    self.count = 0 # Holds number of iterations, including skipped iterations for fp16 loss scaling

  def after_create_session(self, session, coord):
    self.init_global_step = session.run(tf.train.get_global_step())

  def before_run(self, run_context):
    self.t0 = time.time()
    return tf.train.SessionRunArgs(
        fetches=['step_update:0'])

  def after_run(self, run_context, run_values):
    elapsed_secs = time.time() - self.t0
    self.global_step = run_values.results[0]
    self.count += 1

    # Removing first step + first two steps after every checkpoint save
    if (self.global_step - self.init_global_step) % self.save_checkpoints_steps <= 1:
      print("Skipping time record for ", self.global_step, " due to checkpoint-saving/warmup overhead")
    else:
      self.total_time += elapsed_secs

  def end(self, session):
    num_global_steps = self.global_step - self.init_global_step

    self.skipped = (num_global_steps // self.save_checkpoints_steps) * 2 + \
                   min(2, num_global_steps % self.save_checkpoints_steps) - 1