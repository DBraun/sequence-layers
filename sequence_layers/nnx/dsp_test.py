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
"""Tests for NNX DSP layers."""

from absl.testing import parameterized
import jax.numpy as jnp
from sequence_layers.jax import signal
from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.nnx import dsp
from sequence_layers.nnx import test_utils


class DelayTest(test_utils.SequenceLayerTest):

  def test_delay_nonnegative(self):
    with self.assertRaises(ValueError):
      dsp.Delay(length=-1)

  @parameterized.product(
      length=(0, 1, 4), delay_layer_output=(True, False)
  )
  def test_delay(self, length, delay_layer_output):
    x = random_sequence(2, 11, 3, 5)
    l = dsp.Delay(
        length=length, delay_layer_output=delay_layer_output
    )
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, length)
    self.assertEqual(
        l.output_latency, 0 if delay_layer_output else length
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape((3, 5)), (3, 5))

    y = self.verify_contract(l, x)
    self.assertEqual(
        y.shape[1], 11 + length if delay_layer_output else 11
    )


class LookaheadTest(test_utils.SequenceLayerTest):

  def test_lookahead_nonnegative(self):
    with self.assertRaises(ValueError):
      dsp.Lookahead(length=-1)

  @parameterized.product(length=(0, 1, 4))
  def test_lookahead(self, length):
    x = random_sequence(2, 11, 3, 5)
    l = dsp.Lookahead(length=length)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(l.output_latency, length)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape((3, 5)), (3, 5))

    y = self.verify_contract(l, x)
    self.assertEqual(y.shape[1], 11 - length)

  def test_lookahead_preserve_length_in_layer(self):
    x = random_sequence(2, 11, 3, 5)
    l = dsp.Lookahead(length=2, preserve_length_in_layer=True)
    y = self.verify_contract(l, x)
    self.assertEqual(y.shape[1], 11)


class WindowTest(test_utils.SequenceLayerTest):

  def test_window_hann(self):
    x = random_sequence(2, 5, 8)
    l = dsp.Window(axis=2, window_fn=signal.hann_window)
    self.verify_contract(l, x)

  def test_window_hamming(self):
    x = random_sequence(2, 5, 8)
    l = dsp.Window(axis=2, window_fn=signal.hamming_window)
    self.verify_contract(l, x)

  def test_window_output_shape(self):
    l = dsp.Window(axis=2, window_fn=signal.hann_window)
    self.assertEqual(l.get_output_shape((8,)), (8,))

  def test_window_negative_axis(self):
    x = random_sequence(2, 5, 3, 4)
    l = dsp.Window(axis=-1, window_fn=signal.hann_window)
    l.eval()
    y = l.layer(x)
    self.assertEqual(y.values.shape, (2, 5, 3, 4))

  @parameterized.parameters(0, 1, -2, -3, 3)
  def test_window_invalid_axis(self, axis):
    x = random_sequence(2, 5, 1)
    l = dsp.Window(axis=axis, window_fn=signal.hamming_window)
    l.eval()
    with self.assertRaises(ValueError):
      l.layer(x)

  def test_window_values(self):
    """Verify window is actually applied to the values."""
    x = random_sequence(2, 3, 8)
    l = dsp.Window(axis=2, window_fn=signal.hann_window)
    l.eval()
    y = l.layer(x)

    window = signal.hann_window(8, dtype=jnp.float32)
    expected_values = x.values * window[jnp.newaxis, jnp.newaxis, :]
    self.assertSequencesClose(
        y,
        x.apply_values(lambda v: v * window[jnp.newaxis, jnp.newaxis, :]),
    )


if __name__ == '__main__':
  test_utils.main()
