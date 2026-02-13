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
import jax
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


class EinsumDenseTest(test_utils.SequenceLayerTest):

  def test_simple_projection(self):
    # ...a,ab->...b  (5,) -> (7,)
    l = dense.EinsumDense(
        equation='...a,ab->...b',
        input_shape=(5,),
        output_shape=(7,),
        kernel_init=nnx.initializers.normal(),
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 3, 5)
    self.assertEqual(l.get_output_shape((5,)), (7,))
    self.assertEqual(l.kernel.shape, (5, 7))
    self.verify_contract(l, x)

  def test_multi_dim_input(self):
    # ...ab,ac->...cb  (5, 7) -> (11, 7)
    l = dense.EinsumDense(
        equation='...ab,ac->...cb',
        input_shape=(5, 7),
        output_shape=(11, 7),
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 3, 5, 7)
    self.assertEqual(l.get_output_shape((5, 7)), (11, 7))
    self.assertEqual(l.kernel.shape, (5, 11))
    self.verify_contract(l, x)

  def test_contraction(self):
    # ...ab,b->...a  (5, 7) -> (5,)
    l = dense.EinsumDense(
        equation='...ab,b->...a',
        input_shape=(5, 7),
        output_shape=(None,),
        kernel_init=nnx.initializers.normal(),
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 3, 5, 7)
    self.assertEqual(l.get_output_shape((5, 7)), (5,))
    self.assertEqual(l.kernel.shape, (7,))
    self.verify_contract(l, x)

  def test_transpose(self):
    # ...ab,ab->...ba  (5, 7) -> (7, 5)
    l = dense.EinsumDense(
        equation='...ab,ab->...ba',
        input_shape=(5, 7),
        output_shape=(None, None),
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 3, 5, 7)
    self.assertEqual(l.get_output_shape((5, 7)), (7, 5))
    self.assertEqual(l.kernel.shape, (5, 7))
    self.verify_contract(l, x)

  def test_expand_dims(self):
    # ...ab,abc->...bac  (5, 7) -> (7, 5, 2)
    l = dense.EinsumDense(
        equation='...ab,abc->...bac',
        input_shape=(5, 7),
        output_shape=(None, None, 2),
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 3, 5, 7)
    self.assertEqual(l.get_output_shape((5, 7)), (7, 5, 2))
    self.assertEqual(l.kernel.shape, (5, 7, 2))
    self.verify_contract(l, x)

  def test_with_bias(self):
    l = dense.EinsumDense(
        equation='...a,ab->...b',
        input_shape=(5,),
        output_shape=(7,),
        bias_axes='b',
        kernel_init=nnx.initializers.normal(),
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 3, 5)
    self.assertEqual(l.kernel.shape, (5, 7))
    self.assertEqual(l.bias.shape, (7,))
    self.verify_contract(l, x)

    # Verify result matches manual einsum + bias.
    l.eval()
    y = l.layer(x)
    y_expected = x.apply_values(
        lambda v: jnp.einsum(
            '...a,ab->...b', v, l.kernel[...]
        ) + l.bias[...]
    ).mask_invalid()
    self.assertSequencesClose(y, y_expected)

  def test_bias_multi_dim(self):
    # ...a,abc->...bc with bias on 'b' only.
    l = dense.EinsumDense(
        equation='...a,abc->...bc',
        input_shape=(5,),
        output_shape=(7, 11),
        bias_axes='b',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 3, 5)
    self.assertEqual(l.kernel.shape, (5, 7, 11))
    # Bias should broadcast over c: shape (7, 1).
    self.assertEqual(l.bias.shape, (7, 1))
    self.verify_contract(l, x)

  def test_with_activation(self):
    l = dense.EinsumDense(
        equation='...a,ab->...b',
        input_shape=(5,),
        output_shape=(7,),
        activation=jax.nn.relu,
        kernel_init=nnx.initializers.normal(),
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 3, 5)
    self.verify_contract(l, x)

  def test_no_bias_preserves_masked(self):
    l = dense.EinsumDense(
        equation='...a,ab->...b',
        input_shape=(5,),
        output_shape=(7,),
        kernel_init=nnx.initializers.normal(),
        rngs=nnx.Rngs(0),
    )
    l.eval()
    x = random_sequence(2, 3, 5)
    y = l.layer(x)
    # With no bias and no activation, output should be MaskedSequence.
    from sequence_layers.jax.types import MaskedSequence
    self.assertIsInstance(y, MaskedSequence)

  def test_compute_dtype(self):
    l = dense.EinsumDense(
        equation='...a,ab->...b',
        input_shape=(5,),
        output_shape=(7,),
        compute_dtype=jnp.bfloat16,
        kernel_init=nnx.initializers.normal(),
        rngs=nnx.Rngs(0),
    )
    l.eval()
    x = random_sequence(2, 3, 5)
    y = l.layer(x)
    self.assertEqual(y.dtype, jnp.bfloat16)


if __name__ == '__main__':
  test_utils.main()
