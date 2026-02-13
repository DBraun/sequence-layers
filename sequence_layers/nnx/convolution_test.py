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
"""Tests for NNX convolution layers."""

from absl.testing import parameterized
from flax import nnx
import jax
from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.nnx import convolution
from sequence_layers.nnx import test_utils


class Conv1DTest(test_utils.SequenceLayerTest):

  def test_conv1d_causal(self):
    l = convolution.Conv1D(
        in_features=5,
        filters=8,
        kernel_size=3,
        padding='causal',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_conv1d_causal_valid(self):
    l = convolution.Conv1D(
        in_features=5,
        filters=8,
        kernel_size=3,
        padding='causal_valid',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_conv1d_causal_strided(self):
    l = convolution.Conv1D(
        in_features=5,
        filters=8,
        kernel_size=3,
        strides=2,
        padding='causal',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 14, 5)
    self.verify_contract(l, x)

  def test_conv1d_causal_dilated(self):
    l = convolution.Conv1D(
        in_features=5,
        filters=8,
        kernel_size=3,
        dilation_rate=2,
        padding='causal',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_conv1d_valid(self):
    l = convolution.Conv1D(
        in_features=5,
        filters=8,
        kernel_size=3,
        padding='valid',
        rngs=nnx.Rngs(0),
    )
    l.eval()
    x = random_sequence(2, 13, 5)
    # VALID padding doesn't support step, so just test layer.
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 11, 8))

  def test_conv1d_no_bias(self):
    l = convolution.Conv1D(
        in_features=5,
        filters=8,
        kernel_size=3,
        padding='causal',
        use_bias=False,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_conv1d_with_activation(self):
    l = convolution.Conv1D(
        in_features=5,
        filters=8,
        kernel_size=3,
        padding='causal',
        activation=jax.nn.relu,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_conv1d_kernel_size_1(self):
    l = convolution.Conv1D(
        in_features=5,
        filters=8,
        kernel_size=1,
        padding='causal',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_conv1d_output_shape(self):
    l = convolution.Conv1D(
        in_features=5,
        filters=8,
        kernel_size=3,
        padding='causal',
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(l.get_output_shape((5,)), (8,))

  def test_conv1d_groups(self):
    l = convolution.Conv1D(
        in_features=8,
        filters=8,
        kernel_size=3,
        padding='causal',
        groups=4,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 8)
    self.verify_contract(l, x)

  def test_conv1d_reverse_causal(self):
    l = convolution.Conv1D(
        in_features=5,
        filters=8,
        kernel_size=3,
        padding='reverse_causal',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class DepthwiseConv1DTest(test_utils.SequenceLayerTest):

  def test_depthwise_conv1d_causal(self):
    l = convolution.DepthwiseConv1D(
        in_features=5,
        kernel_size=3,
        padding='causal',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_depthwise_conv1d_causal_valid(self):
    l = convolution.DepthwiseConv1D(
        in_features=5,
        kernel_size=3,
        padding='causal_valid',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_depthwise_conv1d_depth_multiplier(self):
    l = convolution.DepthwiseConv1D(
        in_features=5,
        kernel_size=3,
        depth_multiplier=2,
        padding='causal',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_depthwise_conv1d_output_shape(self):
    l = convolution.DepthwiseConv1D(
        in_features=5,
        kernel_size=3,
        depth_multiplier=3,
        padding='causal',
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(l.get_output_shape((5,)), (15,))


class Conv2DTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      # Causal time padding, various kernel/stride combos.
      (3, 1, 1, 'causal'),
      (3, 2, 1, 'causal'),
      (2, 2, 1, 'causal'),
      # Causal valid time padding.
      (3, 1, 1, 'causal_valid'),
      (3, 2, 1, 'causal_valid'),
      # Reverse causal.
      (3, 1, 1, 'reverse_causal'),
      # With dilation.
      (3, 1, 2, 'causal'),
  )
  def test_conv2d_steppable(
      self, kernel_size, strides, dilation_rate, time_padding
  ):
    l = convolution.Conv2D(
        in_features=3,
        filters=4,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        time_padding=time_padding,
        spatial_padding='same',
        rngs=nnx.Rngs(0),
    )
    # Input: [batch, time, spatial, channels]
    x = random_sequence(2, 14, 7, 3)
    self.verify_contract(l, x, atol=1e-5, rtol=1e-5)

  def test_conv2d_valid(self):
    l = convolution.Conv2D(
        in_features=3,
        filters=4,
        kernel_size=3,
        time_padding='valid',
        spatial_padding='valid',
        rngs=nnx.Rngs(0),
    )
    l.eval()
    x = random_sequence(2, 13, 7, 3)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 11, 5, 4))

  def test_conv2d_output_shape(self):
    l = convolution.Conv2D(
        in_features=3,
        filters=4,
        kernel_size=3,
        time_padding='causal',
        spatial_padding='same',
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(l.get_output_shape((7, 3)), (7, 4))

  def test_conv2d_explicit_spatial_padding(self):
    l = convolution.Conv2D(
        in_features=3,
        filters=4,
        kernel_size=3,
        time_padding='causal',
        spatial_padding=(1, 1),
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 7, 3)
    self.verify_contract(l, x, atol=1e-5, rtol=1e-5)

  def test_conv2d_no_bias(self):
    l = convolution.Conv2D(
        in_features=3,
        filters=4,
        kernel_size=3,
        time_padding='causal',
        use_bias=False,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 7, 3)
    self.verify_contract(l, x, atol=1e-5, rtol=1e-5)


class Conv3DTest(test_utils.SequenceLayerTest):

  def test_conv3d_causal(self):
    l = convolution.Conv3D(
        in_features=3,
        filters=4,
        kernel_size=3,
        time_padding='causal',
        rngs=nnx.Rngs(0),
    )
    # Input: [batch, time, h, w, channels]
    x = random_sequence(2, 13, 5, 5, 3)
    self.verify_contract(l, x, atol=1e-5, rtol=1e-5)

  def test_conv3d_causal_valid(self):
    l = convolution.Conv3D(
        in_features=3,
        filters=4,
        kernel_size=3,
        time_padding='causal_valid',
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5, 5, 3)
    self.verify_contract(l, x, atol=1e-5, rtol=1e-5)

  def test_conv3d_valid(self):
    l = convolution.Conv3D(
        in_features=3,
        filters=4,
        kernel_size=3,
        time_padding='valid',
        spatial_padding=('valid', 'valid'),
        rngs=nnx.Rngs(0),
    )
    l.eval()
    x = random_sequence(2, 13, 5, 5, 3)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 11, 3, 3, 4))

  def test_conv3d_output_shape(self):
    l = convolution.Conv3D(
        in_features=3,
        filters=4,
        kernel_size=3,
        time_padding='causal',
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(l.get_output_shape((5, 5, 3)), (5, 5, 4))


class Conv1DTransposeTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (1, 1),
      (1, 2),
      (2, 1),
      (2, 2),
      (2, 3),
      (3, 2),
      (3, 3),
      (3, 4),
  )
  def test_conv1d_transpose_causal(self, kernel_size, strides):
    l = convolution.Conv1DTranspose(
        in_features=3,
        filters=4,
        kernel_size=kernel_size,
        strides=strides,
        padding='causal',
        rngs=nnx.Rngs(0),
    )
    self.assertTrue(l.supports_step)
    self.assertEqual(l.output_ratio, strides)
    x = random_sequence(2, 13, 3)
    self.verify_contract(l, x, grad_atol=1e-5, grad_rtol=1e-5)

  def test_conv1d_transpose_valid(self):
    l = convolution.Conv1DTranspose(
        in_features=3,
        filters=4,
        kernel_size=3,
        strides=2,
        padding='valid',
        rngs=nnx.Rngs(0),
    )
    l.eval()
    self.assertFalse(l.supports_step)
    x = random_sequence(2, 13, 3)
    y = l.layer(x)
    self.assertEqual(y.values.shape[0], 2)
    self.assertEqual(y.values.shape[2], 4)

  def test_conv1d_transpose_same(self):
    l = convolution.Conv1DTranspose(
        in_features=3,
        filters=4,
        kernel_size=3,
        strides=2,
        padding='same',
        rngs=nnx.Rngs(0),
    )
    l.eval()
    x = random_sequence(2, 13, 3)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 26, 4))

  def test_conv1d_transpose_output_shape(self):
    l = convolution.Conv1DTranspose(
        in_features=3,
        filters=4,
        kernel_size=3,
        padding='causal',
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(l.get_output_shape((3,)), (4,))

  def test_conv1d_transpose_unsupported_padding(self):
    with self.assertRaises(ValueError):
      convolution.Conv1DTranspose(
          in_features=3,
          filters=4,
          kernel_size=3,
          padding='reverse_causal',
          rngs=nnx.Rngs(0),
      )

  def test_conv1d_transpose_no_bias(self):
    l = convolution.Conv1DTranspose(
        in_features=3,
        filters=4,
        kernel_size=3,
        strides=2,
        padding='causal',
        use_bias=False,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 3)
    self.verify_contract(l, x, grad_atol=1e-5, grad_rtol=1e-5)


class Conv2DTransposeTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (3, 1),
      (3, 2),
      (2, 2),
      (2, 3),
  )
  def test_conv2d_transpose_causal(self, kernel_size, strides):
    l = convolution.Conv2DTranspose(
        in_features=3,
        filters=4,
        kernel_size=kernel_size,
        strides=strides,
        time_padding='causal',
        spatial_padding='same',
        rngs=nnx.Rngs(0),
    )
    self.assertTrue(l.supports_step)
    self.assertEqual(l.output_ratio, strides)
    x = random_sequence(2, 13, 7, 3)
    self.verify_contract(l, x, grad_atol=1e-5, grad_rtol=1e-5)

  def test_conv2d_transpose_valid(self):
    l = convolution.Conv2DTranspose(
        in_features=3,
        filters=4,
        kernel_size=3,
        strides=2,
        time_padding='valid',
        spatial_padding='valid',
        rngs=nnx.Rngs(0),
    )
    l.eval()
    self.assertFalse(l.supports_step)
    x = random_sequence(2, 13, 7, 3)
    y = l.layer(x)
    self.assertEqual(y.values.shape[0], 2)
    self.assertEqual(y.values.shape[3], 4)

  def test_conv2d_transpose_output_shape(self):
    l = convolution.Conv2DTranspose(
        in_features=3,
        filters=4,
        kernel_size=3,
        time_padding='causal',
        spatial_padding='same',
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(l.get_output_shape((7, 3)), (7, 4))

  def test_conv2d_transpose_unsupported_padding(self):
    with self.assertRaises(ValueError):
      convolution.Conv2DTranspose(
          in_features=3,
          filters=4,
          kernel_size=3,
          time_padding='reverse_causal',
          rngs=nnx.Rngs(0),
      )


if __name__ == '__main__':
  test_utils.main()
