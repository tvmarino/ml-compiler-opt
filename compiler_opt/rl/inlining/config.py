# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inlining Training config."""

import gin
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step


# pylint: disable=g-complex-comprehension
@gin.configurable()
def get_inlining_signature_spec():
  """Returns (time_step_spec, action_spec) for LLVM inlining."""
  observation_spec = dict(
      (key, tf.TensorSpec(dtype=tf.int64, shape=(), name=key)) for key in (
          'caller_basic_block_count',
          'caller_conditionally_executed_blocks',
          'caller_users',
          'callee_basic_block_count',
          'callee_conditionally_executed_blocks',
          'callee_users',
          'nr_ctant_params',
          'node_count',
          'edge_count',
          'callsite_height',
          'cost_estimate',
          # inlining_default is not used as feature in training.
          'inlining_default'))
  reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
  time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)
  action_spec = tensor_spec.BoundedTensorSpec(
      dtype=tf.int64, shape=(), name='inlining_decision', minimum=0, maximum=1)

  return time_step_spec, action_spec
