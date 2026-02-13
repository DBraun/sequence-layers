# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for NNX attention layers."""

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.jax import types as jax_types
from sequence_layers.nnx import attention
from sequence_layers.nnx import test_utils


class DotProductSelfAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0. Steppable.
      (1, 2, 3, 0, False),
      (1, 2, 3, 0, True),
      (3, 5, 3, 0, False),
      (3, 5, 3, 0, True),
      # max_past_horizon > 0, max_future_horizon > 0. Steppable.
      (3, 5, 3, 2, False),
      (3, 5, 3, 2, True),
      (3, 5, 3, 5, False),
      (3, 5, 3, 5, True),
      # max_past_horizon == -1, max_future_horizon > 0. Not steppable.
      (3, 5, -1, 2, False),
      # max_past_horizon > 0, max_future_horizon == -1. Not steppable.
      (3, 5, 3, -1, False),
      # max_past_horizon == -1, max_future_horizon == -1. Not steppable.
      (3, 5, -1, -1, False),
  )
  def test_basic(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
      random_mask,
  ):
    channels = 7
    batch_size = 2
    l = attention.DotProductSelfAttention(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        precision=jax.lax.Precision.HIGHEST,
        per_dim_scale=True,
        in_features=channels,
        rngs=nnx.Rngs(0),
    )

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape((channels,)),
        (num_heads, units_per_head),
    )
    self.assertEqual(
        l.supports_step,
        max_past_horizon >= 0 and max_future_horizon >= 0,
    )
    self.assertEqual(l.input_latency, max(0, max_future_horizon))

    for time in [1, 2, 3, 11, 12]:
      with self.subTest(f'time{time}'):
        x = random_sequence(
            batch_size, time, channels, random_mask=random_mask,
        )
        self.verify_contract(
            l, x, grad_atol=1e-5, grad_rtol=1e-5,
        )

  def test_with_bias(self):
    l = attention.DotProductSelfAttention(
        num_heads=2,
        units_per_head=4,
        max_past_horizon=5,
        max_future_horizon=0,
        use_bias=True,
        in_features=3,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 7, 3)
    self.verify_contract(l, x, grad_atol=1e-5, grad_rtol=1e-5)

  def test_gqa(self):
    l = attention.DotProductSelfAttention(
        num_heads=4,
        units_per_head=3,
        num_kv_heads=2,
        max_past_horizon=5,
        max_future_horizon=0,
        in_features=5,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 7, 5)
    self.verify_contract(l, x, grad_atol=1e-5, grad_rtol=1e-5)

  def test_soft_cap(self):
    l = attention.DotProductSelfAttention(
        num_heads=2,
        units_per_head=4,
        max_past_horizon=5,
        max_future_horizon=0,
        attention_logits_soft_cap=50.0,
        in_features=3,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 7, 3)
    self.verify_contract(l, x, grad_atol=1e-5, grad_rtol=1e-5)

  def test_zero_fully_masked(self):
    l = attention.DotProductSelfAttention(
        num_heads=2,
        units_per_head=4,
        max_past_horizon=5,
        max_future_horizon=0,
        zero_fully_masked=True,
        in_features=3,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 7, 3, random_mask=True)
    self.verify_contract(l, x, grad_atol=1e-5, grad_rtol=1e-5)

  def test_invalid_max_horizons(self):
    with self.assertRaises(ValueError):
      attention.DotProductSelfAttention(
          num_heads=2,
          units_per_head=4,
          max_past_horizon=0,
          max_future_horizon=0,
          in_features=3,
          rngs=nnx.Rngs(0),
      )

  def test_invalid_gqa_heads(self):
    with self.assertRaises(ValueError):
      attention.DotProductSelfAttention(
          num_heads=3,
          units_per_head=4,
          num_kv_heads=2,
          max_past_horizon=5,
          max_future_horizon=0,
          in_features=5,
          rngs=nnx.Rngs(0),
      )

  def test_emits(self):
    l = attention.DotProductSelfAttention(
        num_heads=2,
        units_per_head=4,
        max_past_horizon=5,
        max_future_horizon=0,
        in_features=3,
        rngs=nnx.Rngs(0),
    )
    l.eval()
    x = random_sequence(2, 7, 3)
    y, emits = l.layer_with_emits(x)
    self.assertIsInstance(emits, attention.SelfAttentionEmits)
    self.assertEqual(
        emits.probabilities.values.shape,
        (2, 7, 2, 7),
    )


class DotProductAttentionTest(test_utils.SequenceLayerTest):

  def test_basic(self):
    l = attention.DotProductAttention(
        source_name='source',
        num_heads=2,
        units_per_head=4,
        in_features=5,
        source_features=3,
        precision=jax.lax.Precision.HIGHEST,
        rngs=nnx.Rngs(0),
    )

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(l.get_output_shape((5,)), (2, 4))

    source = random_sequence(2, 10, 3)
    constants = {'source': source}

    for time in [1, 3, 7]:
      with self.subTest(f'time{time}'):
        x = random_sequence(2, time, 5)
        self.verify_contract(
            l, x, constants=constants,
            grad_atol=1e-5, grad_rtol=1e-5,
        )

  def test_with_bias(self):
    l = attention.DotProductAttention(
        source_name='source',
        num_heads=2,
        units_per_head=4,
        use_bias=True,
        in_features=5,
        source_features=3,
        rngs=nnx.Rngs(0),
    )
    source = random_sequence(2, 10, 3)
    constants = {'source': source}
    x = random_sequence(2, 7, 5)
    self.verify_contract(
        l, x, constants=constants,
        grad_atol=1e-5, grad_rtol=1e-5,
    )

  def test_gqa(self):
    l = attention.DotProductAttention(
        source_name='source',
        num_heads=4,
        units_per_head=3,
        num_kv_heads=2,
        in_features=5,
        source_features=6,
        rngs=nnx.Rngs(0),
    )
    source = random_sequence(2, 10, 6)
    constants = {'source': source}
    x = random_sequence(2, 7, 5)
    self.verify_contract(
        l, x, constants=constants,
        grad_atol=1e-5, grad_rtol=1e-5,
    )

  def test_missing_source_raises(self):
    l = attention.DotProductAttention(
        source_name='source',
        num_heads=2,
        units_per_head=4,
        in_features=5,
        source_features=3,
        rngs=nnx.Rngs(0),
    )
    l.eval()
    x = random_sequence(2, 7, 5)
    with self.assertRaises(ValueError):
      l.layer(x)

  def test_emits(self):
    l = attention.DotProductAttention(
        source_name='source',
        num_heads=2,
        units_per_head=4,
        in_features=5,
        source_features=3,
        rngs=nnx.Rngs(0),
    )
    l.eval()
    source = random_sequence(2, 10, 3)
    constants = {'source': source}
    x = random_sequence(2, 7, 5)
    y, emits = l.layer_with_emits(x, constants=constants)
    self.assertIsInstance(emits, attention.CrossAttentionEmits)
    self.assertEqual(
        emits.probabilities.values.shape,
        (2, 7, 2, 10),
    )


if __name__ == '__main__':
  test_utils.main()
