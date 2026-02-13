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
"""Tests for NNX simple layers."""

from flax import nnx
import jax
import jax.numpy as jnp
from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.nnx import simple
from sequence_layers.nnx import test_utils


class IdentityTest(test_utils.SequenceLayerTest):

  def test_identity(self):
    l = simple.Identity()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_identity_output_is_input(self):
    l = simple.Identity()
    l.eval()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    self.assertSequencesEqual(x, y)


class ReluTest(test_utils.SequenceLayerTest):

  def test_relu(self):
    l = simple.Relu()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_relu_values(self):
    l = simple.Relu()
    l.eval()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    expected = x.apply_values_masked(lambda v: jnp.maximum(v, 0))
    self.assertSequencesClose(y, expected)


class GeluTest(test_utils.SequenceLayerTest):

  def test_gelu_approximate(self):
    l = simple.Gelu(approximate=True)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_gelu_exact(self):
    l = simple.Gelu(approximate=False)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class ScaleTest(test_utils.SequenceLayerTest):

  def test_scale_scalar(self):
    l = simple.Scale(scale=2.0)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_scale_values(self):
    l = simple.Scale(scale=3.0)
    l.eval()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    expected = x.apply_values_masked(lambda v: v * 3.0)
    self.assertSequencesClose(y, expected)


class MaskInvalidTest(test_utils.SequenceLayerTest):

  def test_mask_invalid(self):
    l = simple.MaskInvalid()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class DropoutTest(test_utils.SequenceLayerTest):

  def test_dropout_eval(self):
    l = simple.Dropout(rate=0.5, rngs=nnx.Rngs(dropout=0))
    x = random_sequence(2, 13, 5)
    # In eval mode, dropout is a no-op.
    self.verify_contract(l, x)

  def test_dropout_train_modifies_values(self):
    l = simple.Dropout(rate=0.5, rngs=nnx.Rngs(dropout=0))
    l.train()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    # With 50% dropout, the output should differ from input.
    # (Probabilistic, but very unlikely to be identical.)
    self.assertFalse(jnp.allclose(x.values, y.values))

  def test_dropout_zero_rate(self):
    l = simple.Dropout(rate=0.0)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_dropout_train_eval_toggle(self):
    l = simple.Dropout(rate=0.5, rngs=nnx.Rngs(dropout=0))
    l.eval()
    self.assertTrue(l.deterministic)
    l.train()
    self.assertFalse(l.deterministic)
    l.eval()
    self.assertTrue(l.deterministic)


class EmbeddingTest(test_utils.SequenceLayerTest):

  def test_embedding(self):
    l = simple.Embedding(
        num_embeddings=10, dimension=8, rngs=nnx.Rngs(0)
    )
    x = random_sequence(2, 13, dtype=jnp.int32)
    self.verify_contract(l, x, test_gradients=False)

  def test_embedding_output_shape(self):
    l = simple.Embedding(
        num_embeddings=10, dimension=8, rngs=nnx.Rngs(0)
    )
    self.assertEqual(l.get_output_shape(()), (8,))
    self.assertEqual(l.get_output_shape((3,)), (3, 8))


class TanhTest(test_utils.SequenceLayerTest):

  def test_tanh(self):
    l = simple.Tanh()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class SigmoidTest(test_utils.SequenceLayerTest):

  def test_sigmoid(self):
    l = simple.Sigmoid()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class SwishTest(test_utils.SequenceLayerTest):

  def test_swish(self):
    l = simple.Swish()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class SoftplusTest(test_utils.SequenceLayerTest):

  def test_softplus(self):
    l = simple.Softplus()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class SoftmaxTest(test_utils.SequenceLayerTest):

  def test_softmax(self):
    l = simple.Softmax()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class LeakyReluTest(test_utils.SequenceLayerTest):

  def test_leaky_relu(self):
    l = simple.LeakyRelu()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_leaky_relu_slope(self):
    l = simple.LeakyRelu(negative_slope=0.2)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class EluTest(test_utils.SequenceLayerTest):

  def test_elu(self):
    l = simple.Elu()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class ExpTest(test_utils.SequenceLayerTest):

  def test_exp(self):
    l = simple.Exp()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class LogTest(test_utils.SequenceLayerTest):

  def test_log(self):
    l = simple.Log()
    # Use positive values for log.
    x = random_sequence(2, 13, 5)
    x = x.apply_values(lambda v: jnp.abs(v) + 0.1)
    self.verify_contract(l, x)


class AbsTest(test_utils.SequenceLayerTest):

  def test_abs(self):
    l = simple.Abs()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x, test_gradients=False)


class PowerTest(test_utils.SequenceLayerTest):

  def test_power(self):
    l = simple.Power(power=2.0)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class CastTest(test_utils.SequenceLayerTest):

  def test_cast(self):
    l = simple.Cast(dtype=jnp.float16)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x, test_gradients=False)

  def test_cast_output_dtype(self):
    l = simple.Cast(dtype=jnp.float16)
    self.assertEqual(l.get_output_dtype(jnp.float32), jnp.float16)


class GatedLinearUnitTest(test_utils.SequenceLayerTest):

  def test_gated_linear_unit(self):
    l = simple.GatedLinearUnit()
    x = random_sequence(2, 13, 10)  # 10 channels -> 5 output channels
    self.verify_contract(l, x)

  def test_gated_linear_unit_output_shape(self):
    l = simple.GatedLinearUnit()
    self.assertEqual(l.get_output_shape((10,)), (5,))


class GatedTanhUnitTest(test_utils.SequenceLayerTest):

  def test_gated_tanh_unit(self):
    l = simple.GatedTanhUnit()
    x = random_sequence(2, 13, 10)
    self.verify_contract(l, x)


class FlattenTest(test_utils.SequenceLayerTest):

  def test_flatten(self):
    l = simple.Flatten()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_flatten_output_shape(self):
    l = simple.Flatten()
    self.assertEqual(l.get_output_shape((3, 4)), (12,))


class SqueezeTest(test_utils.SequenceLayerTest):

  def test_squeeze(self):
    l = simple.Squeeze(axis=-1)
    x = random_sequence(2, 13, 5, 1)
    self.verify_contract(l, x)

  def test_squeeze_output_shape(self):
    l = simple.Squeeze(axis=-1)
    self.assertEqual(l.get_output_shape((5, 1)), (5,))


class ExpandDimsTest(test_utils.SequenceLayerTest):

  def test_expand_dims(self):
    l = simple.ExpandDims(axis=0)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_expand_dims_output_shape(self):
    l = simple.ExpandDims(axis=0)
    self.assertEqual(l.get_output_shape((5,)), (1, 5))


class ReshapeTest(test_utils.SequenceLayerTest):

  def test_reshape(self):
    l = simple.Reshape(output_shape=(2, 5))
    x = random_sequence(2, 13, 10)
    self.verify_contract(l, x)

  def test_reshape_output_shape(self):
    l = simple.Reshape(output_shape=(2, 5))
    self.assertEqual(l.get_output_shape((10,)), (2, 5))


class Downsample1DTest(test_utils.SequenceLayerTest):

  def test_downsample(self):
    l = simple.Downsample1D(rate=2)
    x = random_sequence(2, 14, 5)
    self.verify_contract(l, x)

  def test_downsample_output_ratio(self):
    import fractions
    l = simple.Downsample1D(rate=3)
    self.assertEqual(l.output_ratio, fractions.Fraction(1, 3))
    self.assertEqual(l.block_size, 3)
    self.assertEqual(l.input_latency, 2)


class Upsample1DTest(test_utils.SequenceLayerTest):

  def test_upsample(self):
    l = simple.Upsample1D(rate=2)
    x = random_sequence(2, 7, 5)
    self.verify_contract(l, x)

  def test_upsample_output_ratio(self):
    import fractions
    l = simple.Upsample1D(rate=3)
    self.assertEqual(l.output_ratio, fractions.Fraction(3))


class AddTest(test_utils.SequenceLayerTest):

  def test_add(self):
    l = simple.Add(shift=1.0)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_add_output_shape(self):
    l = simple.Add(shift=1.0)
    self.assertEqual(l.get_output_shape((5,)), (5,))


class MaximumTest(test_utils.SequenceLayerTest):

  def test_maximum(self):
    l = simple.Maximum(maximum=0.0)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class MinimumTest(test_utils.SequenceLayerTest):

  def test_minimum(self):
    l = simple.Minimum(minimum=0.0)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class MeanTest(test_utils.SequenceLayerTest):

  def test_mean(self):
    l = simple.Mean()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_mean_keepdims(self):
    l = simple.Mean(keepdims=True)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_mean_output_shape(self):
    l = simple.Mean(axis=None)
    self.assertEqual(l.get_output_shape((3, 4)), ())
    l2 = simple.Mean(axis=-1)
    self.assertEqual(l2.get_output_shape((3, 4)), (3,))
    l3 = simple.Mean(axis=-1, keepdims=True)
    self.assertEqual(l3.get_output_shape((3, 4)), (3, 1))


class MinReduceTest(test_utils.SequenceLayerTest):

  def test_min(self):
    l = simple.Min()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class MaxReduceTest(test_utils.SequenceLayerTest):

  def test_max(self):
    l = simple.Max()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class SumTest(test_utils.SequenceLayerTest):

  def test_sum(self):
    l = simple.Sum()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class PReluTest(test_utils.SequenceLayerTest):

  def test_prelu(self):
    l = simple.PRelu()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class AffineTest(test_utils.SequenceLayerTest):

  def test_affine(self):
    l = simple.Affine(rngs=nnx.Rngs(0))
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_affine_with_shape(self):
    l = simple.Affine(shape=(5,), rngs=nnx.Rngs(0))
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_affine_no_scale(self):
    l = simple.Affine(use_scale=False, rngs=nnx.Rngs(0))
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_affine_no_bias(self):
    l = simple.Affine(use_bias=False, rngs=nnx.Rngs(0))
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class TransposeTest(test_utils.SequenceLayerTest):

  def test_transpose(self):
    l = simple.Transpose()
    x = random_sequence(2, 13, 3, 4)
    self.verify_contract(l, x)

  def test_transpose_output_shape(self):
    l = simple.Transpose()
    self.assertEqual(l.get_output_shape((3, 4)), (4, 3))

  def test_transpose_explicit_axes(self):
    l = simple.Transpose(axes=(3, 2))
    x = random_sequence(2, 13, 3, 4)
    self.verify_contract(l, x)


class SwapAxesTest(test_utils.SequenceLayerTest):

  def test_swap_axes(self):
    l = simple.SwapAxes(axis1=0, axis2=1)
    x = random_sequence(2, 13, 3, 4)
    self.verify_contract(l, x)

  def test_swap_axes_output_shape(self):
    l = simple.SwapAxes(axis1=0, axis2=1)
    self.assertEqual(l.get_output_shape((3, 4)), (4, 3))


class MoveAxisTest(test_utils.SequenceLayerTest):

  def test_move_axis(self):
    l = simple.MoveAxis(source=0, destination=1)
    x = random_sequence(2, 13, 3, 4)
    self.verify_contract(l, x)

  def test_move_axis_output_shape(self):
    l = simple.MoveAxis(source=0, destination=1)
    self.assertEqual(l.get_output_shape((3, 4)), (4, 3))


class SliceTest(test_utils.SequenceLayerTest):

  def test_slice(self):
    l = simple.Slice(slices=((0, 3, None),))
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_slice_output_shape(self):
    l = simple.Slice(slices=((0, 3, None),))
    self.assertEqual(l.get_output_shape((5,)), (3,))


class OneHotTest(test_utils.SequenceLayerTest):

  def test_one_hot(self):
    l = simple.OneHot(depth=10)
    x = random_sequence(2, 13, dtype=jnp.int32)
    self.verify_contract(l, x, test_gradients=False)

  def test_one_hot_output_shape(self):
    l = simple.OneHot(depth=10)
    self.assertEqual(l.get_output_shape(()), (10,))
    self.assertEqual(l.get_output_shape((3,)), (3, 10))


class ArgmaxTest(test_utils.SequenceLayerTest):

  def test_argmax(self):
    l = simple.Argmax()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x, test_gradients=False)

  def test_argmax_output_shape(self):
    l = simple.Argmax()
    self.assertEqual(l.get_output_shape((5,)), ())


class LambdaTest(test_utils.SequenceLayerTest):

  def test_lambda(self):
    l = simple.Lambda(
        fn=lambda v: v * 2.0,
        expected_input_spec=jax.ShapeDtypeStruct((5,), jnp.float32),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_lambda_sequence_input(self):
    from sequence_layers.jax import types as jax_types

    def my_fn(seq):
      return jax_types.Sequence(seq.values * 2.0, seq.mask)

    l = simple.Lambda(
        fn=my_fn,
        sequence_input=True,
        expected_input_spec=jax.ShapeDtypeStruct((5,), jnp.float32),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class ModTest(test_utils.SequenceLayerTest):

  def test_mod(self):
    l = simple.Mod(divisor=2.0)
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_mod_array(self):
    import numpy as np
    l = simple.Mod(divisor=np.array([2.0, 3.0, 4.0, 5.0, 6.0]))
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class SnakeTest(test_utils.SequenceLayerTest):

  def test_snake(self):
    l = simple.Snake(features=5, separate_beta=True, rngs=nnx.Rngs(0))
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_snake_shared_beta(self):
    l = simple.Snake(
        features=5, separate_beta=False, rngs=nnx.Rngs(0)
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class EmbeddingTransposeTest(test_utils.SequenceLayerTest):

  def test_embedding_transpose(self):
    emb = simple.Embedding(
        num_embeddings=10, dimension=5, rngs=nnx.Rngs(0)
    )
    l = simple.EmbeddingTranspose(embedding=emb, rngs=nnx.Rngs(1))
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_embedding_transpose_no_bias(self):
    emb = simple.Embedding(
        num_embeddings=10, dimension=5, rngs=nnx.Rngs(0)
    )
    l = simple.EmbeddingTranspose(
        embedding=emb, use_bias=False, rngs=nnx.Rngs(1)
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_embedding_transpose_output_shape(self):
    emb = simple.Embedding(
        num_embeddings=10, dimension=5, rngs=nnx.Rngs(0)
    )
    l = simple.EmbeddingTranspose(embedding=emb, rngs=nnx.Rngs(1))
    self.assertEqual(l.get_output_shape((5,)), (10,))


class EmitTest(test_utils.SequenceLayerTest):

  def test_emit(self):
    l = simple.Emit()
    l.eval()
    x = random_sequence(2, 13, 5)
    y, emits = l.layer_with_emits(x)
    self.assertEqual(y.shape, x.shape)


class NamedEmitTest(test_utils.SequenceLayerTest):

  def test_named_emit(self):
    l = simple.NamedEmit(emit_name='my_output')
    l.eval()
    x = random_sequence(2, 13, 5)
    y, emits = l.layer_with_emits(x)
    self.assertEqual(y.shape, x.shape)
    self.assertIn('my_output', emits)


class Upsample2DTest(test_utils.SequenceLayerTest):

  def test_upsample2d(self):
    l = simple.Upsample2D(rate=2)
    l.eval()
    x = random_sequence(2, 7, 4, 5)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 14, 8, 5))

  def test_upsample2d_asymmetric(self):
    l = simple.Upsample2D(rate=(2, 3))
    l.eval()
    x = random_sequence(2, 7, 4, 5)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 14, 12, 5))

  def test_upsample2d_output_shape(self):
    l = simple.Upsample2D(rate=2)
    self.assertEqual(l.get_output_shape((4, 5)), (8, 5))


class EinopsRearrangeTest(test_utils.SequenceLayerTest):

  def test_einops_rearrange(self):
    l = simple.EinopsRearrange(
        pattern='(h w) c -> h w c', axes_lengths={'h': 3}
    )
    x = random_sequence(2, 13, 6, 5)
    self.verify_contract(l, x)

  def test_einops_rearrange_output_shape(self):
    l = simple.EinopsRearrange(
        pattern='(h w) c -> h w c', axes_lengths={'h': 3}
    )
    self.assertEqual(l.get_output_shape((6, 5)), (3, 2, 5))

  def test_einops_rearrange_flatten(self):
    l = simple.EinopsRearrange(pattern='h w c -> (h w c)')
    x = random_sequence(2, 13, 3, 4, 5)
    self.verify_contract(l, x)


if __name__ == '__main__':
  test_utils.main()
