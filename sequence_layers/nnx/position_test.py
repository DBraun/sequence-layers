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
"""Tests for NNX position layers."""

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.jax import types as jax_types
from sequence_layers.nnx import position
from sequence_layers.nnx import test_utils


class AddTimingSignalTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      dict(
          min_timescale=1.0,
          max_timescale=1.0e4,
          trainable_scale=True,
          channel_shape=(3,),
          axes=None,
      ),
      dict(
          min_timescale=1.0,
          max_timescale=1.0e4,
          trainable_scale=False,
          channel_shape=(3,),
          axes=None,
      ),
      dict(
          min_timescale=10.0,
          max_timescale=1.0e5,
          trainable_scale=False,
          channel_shape=(3,),
          axes=0,
      ),
      dict(
          min_timescale=1.0,
          max_timescale=1.0e4,
          trainable_scale=True,
          channel_shape=(5, 9),
          axes=(1,),
      ),
      dict(
          min_timescale=1.0,
          max_timescale=1.0e4,
          trainable_scale=True,
          channel_shape=(5, 9, 3),
          axes=[1, 2],
      ),
      dict(
          min_timescale=1.0,
          max_timescale=1.0e4,
          trainable_scale=True,
          channel_shape=(5, 9),
          axes=(1,),
          only_advance_position_for_valid_timesteps=False,
      ),
  )
  def test_basic(
      self,
      min_timescale,
      max_timescale,
      trainable_scale,
      channel_shape,
      axes,
      only_advance_position_for_valid_timesteps=True,
  ):
    l = position.AddTimingSignal(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        trainable_scale=trainable_scale,
        axes=axes,
        only_advance_position_for_valid_timesteps=only_advance_position_for_valid_timesteps,
        rngs=nnx.Rngs(0) if trainable_scale else None,
    )

    batch_size = 8
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    for time in range(13, 15):
      x = random_sequence(
          batch_size,
          time,
          *channel_shape,
          random_mask=True,
      )
      self.verify_contract(
          l, x, grad_atol=1e-5, grad_rtol=1e-5,
      )

  def test_no_trainable_scale(self):
    l = position.AddTimingSignal(
        trainable_scale=False,
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_trainable_scale(self):
    l = position.AddTimingSignal(
        trainable_scale=True,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_trainable_scale_requires_rngs(self):
    with self.assertRaises(ValueError):
      position.AddTimingSignal(trainable_scale=True)


class ApplyRotaryPositionalEncodingTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      max_wavelength=(1.0e4, 1.0e5),
      channel_shape=((4,), (3, 6)),
      only_advance_position_for_valid_timesteps=(False, True),
  )
  def test_basic(
      self,
      max_wavelength,
      channel_shape,
      only_advance_position_for_valid_timesteps,
  ):
    l = position.ApplyRotaryPositionalEncoding(
        max_wavelength=max_wavelength,
        only_advance_position_for_valid_timesteps=only_advance_position_for_valid_timesteps,
    )

    batch_size = 2
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    for time in range(13, 15):
      x = random_sequence(
          batch_size,
          time,
          *channel_shape,
          random_mask=only_advance_position_for_valid_timesteps,
      )
      self.verify_contract(l, x)

  def test_only_advance_position_for_valid_timesteps(self):
    l = position.ApplyRotaryPositionalEncoding(
        max_wavelength=1.0e5,
        only_advance_position_for_valid_timesteps=True,
    )
    l.eval()

    x = jax_types.Sequence(
        jax.random.normal(jax.random.PRNGKey(1234), (3, 3, 6)),
        jnp.asarray(
            [[False, True, True], [True, False, True], [True, True, False]]
        ),
    ).mask_invalid()

    y = l.layer(x)

    # Verify the layer ignores invalid timesteps by showing the output
    # is equal to processing a sequence without the invalid timesteps.
    self.assertSequencesEqual(
        y[0:1, 1:],
        l.layer(x[0:1, 1:]),
    )
    self.assertSequencesEqual(
        jax_types.Sequence.concatenate_sequences(
            [y[1:2, :1], y[1:2, 2:]]
        ),
        l.layer(
            jax_types.Sequence.concatenate_sequences(
                [x[1:2, :1], x[1:2, 2:]]
            ),
        ),
    )
    self.assertSequencesEqual(
        y[2:3, :-1],
        l.layer(x[2:3, :-1]),
    )

  def test_odd_axis_size_raises(self):
    with self.assertRaises(ValueError):
      l = position.ApplyRotaryPositionalEncoding(
          max_wavelength=1.0e4,
      )
      x = random_sequence(2, 5, 3)
      l.layer(x)


if __name__ == '__main__':
  test_utils.main()
