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
"""Tests for NNX combinators."""

from flax import nnx
import jax.numpy as jnp
from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.nnx import combinators
from sequence_layers.nnx import dense
from sequence_layers.nnx import simple
from sequence_layers.nnx import test_utils


class SerialTest(test_utils.SequenceLayerTest):

  def test_serial_identity(self):
    l = combinators.Serial([simple.Identity(), simple.Identity()])
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_serial_relu_scale(self):
    l = combinators.Serial([simple.Relu(), simple.Scale(scale=2.0)])
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_serial_with_dense(self):
    l = combinators.Serial([
        dense.Dense(in_features=5, features=8, rngs=nnx.Rngs(0)),
        simple.Relu(),
        dense.Dense(in_features=8, features=3, rngs=nnx.Rngs(1)),
    ])
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_serial_output_shape(self):
    l = combinators.Serial([
        dense.Dense(in_features=5, features=8, rngs=nnx.Rngs(0)),
        dense.Dense(in_features=8, features=3, rngs=nnx.Rngs(1)),
    ])
    self.assertEqual(l.get_output_shape((5,)), (3,))

  def test_serial_train_eval(self):
    l = combinators.Serial([
        simple.Dropout(rate=0.5, rngs=nnx.Rngs(dropout=0)),
        simple.Identity(),
    ])
    l.train()
    self.assertFalse(l.deterministic)
    self.assertFalse(l.layers[0].deterministic)
    l.eval()
    self.assertTrue(l.deterministic)
    self.assertTrue(l.layers[0].deterministic)


class ResidualTest(test_utils.SequenceLayerTest):

  def test_residual_identity(self):
    l = combinators.Residual([simple.Identity()])
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_residual_identity_doubles_input(self):
    l = combinators.Residual([simple.Identity()])
    l.eval()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    expected = x.apply_values(lambda v: v * 2)
    self.assertSequencesClose(y, expected)

  def test_residual_with_dense(self):
    # Dense with same output features as input so residual addition works.
    l = combinators.Residual([
        dense.Dense(in_features=5, features=5, rngs=nnx.Rngs(0)),
        simple.Relu(),
    ])
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_residual_with_shortcut(self):
    # Use a Dense shortcut to project from 5 -> 3 to match body output.
    l = combinators.Residual(
        [dense.Dense(in_features=5, features=3, rngs=nnx.Rngs(0))],
        shortcut=dense.Dense(in_features=5, features=3, rngs=nnx.Rngs(1)),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_residual_mismatched_output_ratio_raises(self):
    import fractions

    class FakeDoubleRatio(simple.Identity):
      @property
      def output_ratio(self):
        return fractions.Fraction(2)

    with self.assertRaises(ValueError):
      combinators.Residual(
          [simple.Identity()], shortcut=FakeDoubleRatio()
      )


class ParallelTest(test_utils.SequenceLayerTest):

  def test_parallel_stack(self):
    l = combinators.Parallel(
        [simple.Identity(), simple.Identity()],
        combination=combinators.CombinationMode.STACK,
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_parallel_concat(self):
    l = combinators.Parallel(
        [simple.Identity(), simple.Identity()],
        combination=combinators.CombinationMode.CONCAT,
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_parallel_sum(self):
    l = combinators.Parallel(
        [simple.Identity(), simple.Identity()],
        combination=combinators.CombinationMode.ADD,
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_parallel_with_dense(self):
    l = combinators.Parallel(
        [
            dense.Dense(in_features=5, features=3, rngs=nnx.Rngs(0)),
            dense.Dense(in_features=5, features=3, rngs=nnx.Rngs(1)),
        ],
        combination=combinators.CombinationMode.ADD,
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_parallel_output_shape_stack(self):
    l = combinators.Parallel(
        [simple.Identity(), simple.Identity()],
        combination=combinators.CombinationMode.STACK,
    )
    self.assertEqual(l.get_output_shape((5,)), (2, 5))

  def test_parallel_output_shape_concat(self):
    l = combinators.Parallel(
        [simple.Identity(), simple.Identity()],
        combination=combinators.CombinationMode.CONCAT,
    )
    self.assertEqual(l.get_output_shape((5,)), (10,))

  def test_parallel_mismatched_output_ratio_raises(self):
    import fractions

    class FakeDoubleRatio(simple.Identity):
      @property
      def output_ratio(self):
        return fractions.Fraction(2)

    with self.assertRaises(ValueError):
      combinators.Parallel([simple.Identity(), FakeDoubleRatio()])

  def test_parallel_empty(self):
    l = combinators.Parallel([])
    x = random_sequence(2, 13, 5)
    l.eval()
    y = l.layer(x)
    self.assertSequencesEqual(x, y)


class BidirectionalTest(test_utils.SequenceLayerTest):

  def test_bidirectional_stack(self):
    l = combinators.Bidirectional(
        forward=simple.Identity(),
        backward=simple.Identity(),
        combination=combinators.CombinationMode.STACK,
    )
    l.eval()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 13, 2, 5))

  def test_bidirectional_sum(self):
    l = combinators.Bidirectional(
        forward=simple.Identity(),
        backward=simple.Identity(),
        combination=combinators.CombinationMode.ADD,
    )
    l.eval()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 13, 5))

  def test_bidirectional_concat(self):
    l = combinators.Bidirectional(
        forward=simple.Identity(),
        backward=simple.Identity(),
        combination=combinators.CombinationMode.CONCAT,
    )
    l.eval()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 13, 10))

  def test_bidirectional_with_dense(self):
    l = combinators.Bidirectional(
        forward=dense.Dense(
            in_features=5, features=3, rngs=nnx.Rngs(0)
        ),
        backward=dense.Dense(
            in_features=5, features=3, rngs=nnx.Rngs(1)
        ),
        combination=combinators.CombinationMode.ADD,
    )
    l.eval()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 13, 3))

  def test_bidirectional_does_not_support_step(self):
    l = combinators.Bidirectional(
        forward=simple.Identity(),
        backward=simple.Identity(),
    )
    self.assertFalse(l.supports_step)

  def test_bidirectional_mismatched_output_ratio_raises(self):
    import fractions

    class FakeDoubleRatio(simple.Identity):
      @property
      def output_ratio(self):
        return fractions.Fraction(2)

    with self.assertRaises(ValueError):
      combinators.Bidirectional(
          forward=simple.Identity(),
          backward=FakeDoubleRatio(),
      )


class BlockwiseTest(test_utils.SequenceLayerTest):

  def test_blockwise(self):
    child = dense.Dense(in_features=5, features=5, rngs=nnx.Rngs(0))
    l = combinators.Blockwise(child_layer=child, block_size=4)
    x = random_sequence(2, 12, 5)
    self.verify_contract(l, x)

  def test_blockwise_invalid_block_size(self):
    from sequence_layers.nnx import convolution
    child = convolution.Conv1D(
        in_features=5, filters=5, kernel_size=3, strides=2,
        padding='causal', rngs=nnx.Rngs(0),
    )
    # child.block_size == 2, so block_size=3 is invalid.
    with self.assertRaises(ValueError):
      combinators.Blockwise(child_layer=child, block_size=3)

  def test_blockwise_output_shape(self):
    child = dense.Dense(in_features=5, features=3, rngs=nnx.Rngs(0))
    l = combinators.Blockwise(child_layer=child, block_size=4)
    self.assertEqual(l.get_output_shape((5,)), (3,))
    self.assertEqual(l.block_size, 4)


if __name__ == '__main__':
  test_utils.main()
