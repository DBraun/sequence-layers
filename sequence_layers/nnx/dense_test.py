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
"""Tests for NNX dense layers."""

from flax import nnx
import jax.numpy as jnp
from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.nnx import dense
from sequence_layers.nnx import test_utils


class DenseTest(test_utils.SequenceLayerTest):

  def test_dense(self):
    l = dense.Dense(in_features=5, features=3, rngs=nnx.Rngs(0))
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_dense_no_bias(self):
    l = dense.Dense(
        in_features=5, features=3, use_bias=False, rngs=nnx.Rngs(0)
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_dense_with_activation(self):
    l = dense.Dense(
        in_features=5, features=3, activation=nnx.relu, rngs=nnx.Rngs(0)
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_dense_output_shape(self):
    l = dense.Dense(in_features=5, features=3, rngs=nnx.Rngs(0))
    self.assertEqual(l.get_output_shape((5,)), (3,))
    self.assertEqual(l.get_output_shape((7, 5)), (7, 3))

  def test_dense_output_dtype(self):
    l = dense.Dense(in_features=5, features=3, rngs=nnx.Rngs(0))
    self.assertEqual(l.get_output_dtype(jnp.float32), jnp.float32)

  def test_dense_bfloat16_compute(self):
    l = dense.Dense(
        in_features=5,
        features=3,
        compute_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(0),
    )
    l.eval()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    self.assertEqual(y.dtype, jnp.bfloat16)


class DenseShapedTest(test_utils.SequenceLayerTest):

  def test_dense_shaped(self):
    l = dense.DenseShaped(
        in_shape=(5,), output_shape=(3,), rngs=nnx.Rngs(0)
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_dense_shaped_multi_dim(self):
    l = dense.DenseShaped(
        in_shape=(3, 4), output_shape=(5, 6), rngs=nnx.Rngs(0)
    )
    x = random_sequence(2, 13, 3, 4)
    self.verify_contract(l, x)

  def test_dense_shaped_no_bias(self):
    l = dense.DenseShaped(
        in_shape=(5,), output_shape=(3,),
        use_bias=False, rngs=nnx.Rngs(0)
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_dense_shaped_output_shape(self):
    l = dense.DenseShaped(
        in_shape=(5,), output_shape=(3, 4), rngs=nnx.Rngs(0)
    )
    self.assertEqual(l.get_output_shape((5,)), (3, 4))

  def test_dense_shaped_with_activation(self):
    l = dense.DenseShaped(
        in_shape=(5,), output_shape=(3,),
        activation=nnx.relu, rngs=nnx.Rngs(0)
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


if __name__ == '__main__':
  test_utils.main()
