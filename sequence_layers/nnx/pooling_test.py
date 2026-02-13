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
"""Tests for NNX pooling layers."""

from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.nnx import pooling
from sequence_layers.nnx import test_utils


class MaxPooling1DTest(test_utils.SequenceLayerTest):

  def test_max_pool_causal(self):
    l = pooling.MaxPooling1D(pool_size=3, padding='causal')
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x, test_gradients=False)

  def test_max_pool_causal_strided(self):
    l = pooling.MaxPooling1D(pool_size=3, strides=2, padding='causal')
    x = random_sequence(2, 14, 5)
    self.verify_contract(l, x, test_gradients=False)

  def test_max_pool_valid(self):
    l = pooling.MaxPooling1D(pool_size=3, padding='valid')
    l.eval()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 11, 5))


class MinPooling1DTest(test_utils.SequenceLayerTest):

  def test_min_pool_causal(self):
    l = pooling.MinPooling1D(pool_size=3, padding='causal')
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x, test_gradients=False)


class AveragePooling1DTest(test_utils.SequenceLayerTest):

  def test_avg_pool_causal(self):
    l = pooling.AveragePooling1D(pool_size=3, padding='causal')
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_avg_pool_causal_strided(self):
    l = pooling.AveragePooling1D(pool_size=3, strides=2, padding='causal')
    x = random_sequence(2, 14, 5)
    self.verify_contract(l, x)

  def test_avg_pool_masked_average(self):
    l = pooling.AveragePooling1D(
        pool_size=3, padding='causal', masked_average=True
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_avg_pool_valid(self):
    l = pooling.AveragePooling1D(pool_size=3, padding='valid')
    l.eval()
    x = random_sequence(2, 13, 5)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 11, 5))


class MaxPooling2DTest(test_utils.SequenceLayerTest):

  def test_max_pool_2d_causal(self):
    l = pooling.MaxPooling2D(
        pool_size=3, time_padding='causal', spatial_padding='same'
    )
    x = random_sequence(2, 13, 7, 5)
    self.verify_contract(l, x, test_gradients=False)

  def test_max_pool_2d_causal_strided(self):
    l = pooling.MaxPooling2D(
        pool_size=3, strides=2, time_padding='causal', spatial_padding='same'
    )
    x = random_sequence(2, 14, 7, 5)
    self.verify_contract(l, x, test_gradients=False)

  def test_max_pool_2d_valid(self):
    l = pooling.MaxPooling2D(
        pool_size=3, time_padding='valid', spatial_padding='valid'
    )
    l.eval()
    x = random_sequence(2, 13, 7, 5)
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 11, 5, 5))

  def test_max_pool_2d_output_shape(self):
    l = pooling.MaxPooling2D(
        pool_size=3, time_padding='causal', spatial_padding='same'
    )
    self.assertEqual(l.get_output_shape((7, 5)), (7, 5))


class MinPooling2DTest(test_utils.SequenceLayerTest):

  def test_min_pool_2d_causal(self):
    l = pooling.MinPooling2D(
        pool_size=3, time_padding='causal', spatial_padding='same'
    )
    x = random_sequence(2, 13, 7, 5)
    self.verify_contract(l, x, test_gradients=False)


class AveragePooling2DTest(test_utils.SequenceLayerTest):

  def test_avg_pool_2d_causal(self):
    l = pooling.AveragePooling2D(
        pool_size=3, time_padding='causal', spatial_padding='same'
    )
    x = random_sequence(2, 13, 7, 5)
    self.verify_contract(l, x, atol=1e-5, rtol=1e-5)

  def test_avg_pool_2d_masked_average(self):
    l = pooling.AveragePooling2D(
        pool_size=3, time_padding='causal', spatial_padding='same',
        masked_average=True,
    )
    x = random_sequence(2, 13, 7, 5)
    self.verify_contract(l, x, atol=1e-5, rtol=1e-5)


class MaxPooling3DTest(test_utils.SequenceLayerTest):

  def test_max_pool_3d_causal(self):
    l = pooling.MaxPooling3D(
        pool_size=3, time_padding='causal',
    )
    x = random_sequence(2, 13, 5, 5, 3)
    self.verify_contract(l, x, test_gradients=False)

  def test_max_pool_3d_output_shape(self):
    l = pooling.MaxPooling3D(
        pool_size=3, time_padding='causal',
    )
    self.assertEqual(l.get_output_shape((5, 5, 3)), (5, 5, 3))


class MinPooling3DTest(test_utils.SequenceLayerTest):

  def test_min_pool_3d_causal(self):
    l = pooling.MinPooling3D(
        pool_size=3, time_padding='causal',
    )
    x = random_sequence(2, 13, 5, 5, 3)
    self.verify_contract(l, x, test_gradients=False)


class AveragePooling3DTest(test_utils.SequenceLayerTest):

  def test_avg_pool_3d_causal(self):
    l = pooling.AveragePooling3D(
        pool_size=3, time_padding='causal',
    )
    x = random_sequence(2, 13, 5, 5, 3)
    self.verify_contract(l, x, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  test_utils.main()
