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
"""Tests for NNX recurrent layers."""

from flax import nnx
import jax.numpy as jnp
from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.nnx import recurrent
from sequence_layers.nnx import test_utils


class LSTMTest(test_utils.SequenceLayerTest):

  def test_lstm(self):
    l = recurrent.LSTM(
        in_features=5,
        units=8,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 7, 5)
    self.verify_contract(l, x)

  def test_lstm_no_bias(self):
    l = recurrent.LSTM(
        in_features=5,
        units=8,
        use_bias=False,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 7, 5)
    self.verify_contract(l, x)

  def test_lstm_output_shape(self):
    l = recurrent.LSTM(
        in_features=5,
        units=8,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(l.get_output_shape((5,)), (8,))


class RGLRUTest(test_utils.SequenceLayerTest):

  def test_rglru_real(self):
    l = recurrent.RGLRU(
        in_features=8,
        units=8,
        num_heads=2,
        only_real=True,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 32, 8)
    self.verify_contract(l, x)

  def test_rglru_complex(self):
    l = recurrent.RGLRU(
        in_features=8,
        units=8,
        num_heads=2,
        only_real=False,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 32, 8)
    self.verify_contract(l, x)

  def test_rglru_linear_native(self):
    l = recurrent.RGLRU(
        in_features=8,
        units=8,
        num_heads=2,
        scan_type='linear_native',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 32, 8)
    self.verify_contract(l, x)

  def test_rglru_associative_native(self):
    l = recurrent.RGLRU(
        in_features=8,
        units=8,
        num_heads=2,
        scan_type='associative_native',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 32, 8)
    self.verify_contract(l, x)

  def test_rglru_output_shape(self):
    l = recurrent.RGLRU(
        in_features=8,
        units=16,
        num_heads=4,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(l.get_output_shape((8,)), (16,))

  def test_rglru_bfloat16(self):
    l = recurrent.RGLRU(
        in_features=8,
        units=8,
        num_heads=2,
        param_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 32, 8)
    self.verify_contract(l, x, atol=5e-2, rtol=5e-2)


if __name__ == '__main__':
  test_utils.main()
