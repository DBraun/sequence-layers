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
"""Tests for NNX normalization layers."""

from flax import nnx
from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.nnx import normalization
from sequence_layers.nnx import test_utils


class LayerNormalizationTest(test_utils.SequenceLayerTest):

  def test_layer_norm(self):
    l = normalization.LayerNormalization(
        features=5,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_layer_norm_no_bias(self):
    l = normalization.LayerNormalization(
        features=5,
        use_bias=False,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_layer_norm_no_scale(self):
    l = normalization.LayerNormalization(
        features=5,
        use_scale=False,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_layer_norm_output_shape(self):
    l = normalization.LayerNormalization(
        features=5,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(l.get_output_shape((5,)), (5,))


class RMSNormalizationTest(test_utils.SequenceLayerTest):

  def test_rms_norm(self):
    l = normalization.RMSNormalization(
        features=5,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)

  def test_rms_norm_no_scale(self):
    l = normalization.RMSNormalization(
        features=5,
        use_scale=False,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class L2NormalizeTest(test_utils.SequenceLayerTest):

  def test_l2_normalize(self):
    l = normalization.L2Normalize()
    x = random_sequence(2, 13, 5)
    self.verify_contract(l, x)


class GroupNormalizationTest(test_utils.SequenceLayerTest):

  def test_group_norm(self):
    l = normalization.GroupNormalization(
        num_groups=2,
        features=4,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 4)
    self.verify_contract(l, x)

  def test_group_norm_no_bias(self):
    l = normalization.GroupNormalization(
        num_groups=2,
        features=4,
        use_bias=False,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 4)
    self.verify_contract(l, x)


class BatchNormalizationTest(test_utils.SequenceLayerTest):

  def test_batch_norm(self):
    l = normalization.BatchNormalization(
        num_features=5,
        rngs=nnx.Rngs(0),
    )
    # Train to populate running stats.
    x = random_sequence(2, 13, 5)
    l.train()
    _ = l.layer(x)
    # Verify contract in eval mode (uses running stats).
    self.verify_contract(l, x)

  def test_batch_norm_no_bias(self):
    l = normalization.BatchNormalization(
        num_features=5,
        use_bias=False,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    l.train()
    _ = l.layer(x)
    self.verify_contract(l, x)

  def test_batch_norm_no_scale(self):
    l = normalization.BatchNormalization(
        num_features=5,
        use_scale=False,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 5)
    l.train()
    _ = l.layer(x)
    self.verify_contract(l, x)

  def test_batch_norm_step_training_raises(self):
    l = normalization.BatchNormalization(
        num_features=5,
        rngs=nnx.Rngs(0),
    )
    l.train()
    x = random_sequence(2, 1, 5)
    state = l.get_initial_state(
        batch_size=2, input_spec=x.channel_spec
    )
    with self.assertRaises(ValueError):
      l.step(x, state)

  def test_batch_norm_output_shape(self):
    l = normalization.BatchNormalization(
        num_features=5,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(l.get_output_shape((5,)), (5,))
    self.assertEqual(l.get_output_shape((7, 5)), (7, 5))

  def test_batch_norm_multi_dim(self):
    l = normalization.BatchNormalization(
        num_features=5,
        axis=-1,
        rngs=nnx.Rngs(0),
    )
    x = random_sequence(2, 13, 7, 5)
    l.train()
    _ = l.layer(x)
    self.verify_contract(l, x)


if __name__ == '__main__':
  test_utils.main()
