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

import fractions
import itertools
import math

from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import signal
from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.jax import types as jax_types
from sequence_layers.nnx import dsp
from sequence_layers.nnx import test_utils
from sequence_layers.nnx import types


def _pad_or_truncate_for_fft(values, padding, axis, length):
  """Numpy version for test reference."""
  input_dim = values.shape[axis]
  if input_dim == length:
    return values
  if input_dim < length:
    pad_amount = length - input_dim
    if padding == 'center':
      pad_left = pad_amount // 2
      pad_right = pad_amount - pad_left
    else:
      pad_left = 0
      pad_right = pad_amount
    paddings = [(0, 0)] * values.ndim
    paddings[axis] = (pad_left, pad_right)
    return np.pad(values, paddings)
  else:
    if padding == 'center':
      trim_left = (input_dim - length) // 2
    else:
      trim_left = 0
    slices = [slice(None)] * values.ndim
    slices[axis] = slice(trim_left, trim_left + length)
    return values[tuple(slices)]


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


class FrameTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      frame_length_frame_step=(
          (1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3),
      ),
      channel_shape=((), (4,), (5, 9)),
      padding=(
          'causal_valid',
          'semicausal',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'same',
          'valid',
          'explicit_semicausal',
          'semicausal_full',
      ),
  )
  def test_frame(self, frame_length_frame_step, channel_shape, padding):
    batch_size = 2
    frame_length, frame_step = frame_length_frame_step
    if padding == 'explicit_semicausal':
      total_pad = frame_length - 1
      overlap = max(0, frame_length - frame_step)
      explicit_padding = (overlap, total_pad - overlap)
    else:
      explicit_padding = padding

    l = dsp.Frame(
        frame_length=frame_length,
        frame_step=frame_step,
        padding=explicit_padding,
    )
    self.assertEqual(
        l.supports_step,
        padding
        in (
            'causal_valid',
            'semicausal',
            'reverse_causal_valid',
            'causal',
            'reverse_causal',
            'explicit_semicausal',
        ),
    )
    self.assertEqual(l.block_size, frame_step)
    self.assertEqual(1 / l.output_ratio, frame_step)

    match padding:
      case 'causal_valid' | 'causal' | 'semicausal':
        expected_input_latency = 0
      case 'reverse_causal_valid' | 'reverse_causal':
        expected_input_latency = frame_length - 1
      case 'explicit_semicausal':
        expected_input_latency = (
            (frame_length - 1)
            - max(0, frame_length - frame_step)
        )
      case 'semicausal_full':
        expected_input_latency = frame_step - 1
      case _:
        expected_input_latency = 0

    self.assertEqual(l.input_latency, expected_input_latency)
    self.assertEqual(
        l.output_latency,
        expected_input_latency // frame_step,
    )

    x = random_sequence(batch_size, 1, *channel_shape)
    self.assertEqual(
        l.get_output_shape(channel_shape or ()),
        (frame_length,) + channel_shape,
    )

    for time in range(
        20 * l.block_size - 1, 20 * l.block_size + 2
    ):
      x = random_sequence(
          batch_size, time, *channel_shape,
          low_length=time // 2,
      )
      self.verify_contract(l, x)

  def test_frame_invalid_length(self):
    with self.assertRaises(ValueError):
      dsp.Frame(frame_length=0, frame_step=1)

  def test_frame_invalid_step(self):
    with self.assertRaises(ValueError):
      dsp.Frame(frame_length=1, frame_step=0)


class OverlapAddTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      frame_length_frame_step=(
          (1, 1), (2, 1), (2, 2), (3, 2),
      ),
      inner_shape=((), (3,), (5, 9)),
      padding=('causal', 'valid', 'semicausal_full'),
  )
  def test_overlap_add(
      self, frame_length_frame_step, inner_shape, padding
  ):
    frame_length, frame_step = frame_length_frame_step
    b, t = 2, 34
    x = random_sequence(
        b, t, frame_length, *inner_shape
    )
    l = dsp.OverlapAdd(
        frame_length=frame_length,
        frame_step=frame_step,
        padding=padding,
    )
    self.assertEqual(
        l.supports_step, padding == 'causal'
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, frame_step)
    self.assertEqual(
        l.get_output_shape(
            (frame_length,) + inner_shape
        ),
        inner_shape,
    )
    self.verify_contract(l, x)

  @parameterized.parameters(
      (1, 1), (2, 1), (2, 2), (3, 2)
  )
  def test_frame_overlap_add_perfect(
      self, frame_length, frame_step
  ):
    b, t = 2, 35
    x = random_sequence(b, t)
    forward = dsp.Frame(
        frame_length=frame_length,
        frame_step=frame_step,
        padding='semicausal_full',
    )
    backward = dsp.OverlapAdd(
        frame_length=frame_length,
        frame_step=frame_step,
        padding='semicausal_full',
    )
    forward.eval()
    backward.eval()

    y = forward.layer(x)
    z = backward.layer(y)

    self.assertLessEqual(x.shape[1], z.shape[1])
    self.assertTrue(jnp.all(z.lengths() >= x.lengths()))
    np.testing.assert_array_equal(
        z.mask[:, x.shape[1]:],
        jnp.zeros(
            (z.shape[0], z.shape[1] - x.shape[1]),
            dtype=jnp.bool_,
        ),
    )

    z_values = z.values[:, :x.shape[1]]
    z_mask = z.mask[:, :x.shape[1]]
    difference_mask = jnp.logical_xor(x.mask, z_mask)
    self.assertTrue(
        jnp.all(z_values[difference_mask] == 0)
    )

  def test_overlap_add_invalid_length(self):
    with self.assertRaises(ValueError):
      dsp.OverlapAdd(frame_length=0, frame_step=1)

  def test_overlap_add_invalid_step(self):
    with self.assertRaises(ValueError):
      dsp.OverlapAdd(frame_length=1, frame_step=0)

  def test_overlap_add_length_less_than_step(self):
    with self.assertRaises(ValueError):
      dsp.OverlapAdd(frame_length=2, frame_step=3)

  def test_overlap_add_unsupported_padding(self):
    with self.assertRaises(ValueError):
      dsp.OverlapAdd(
          frame_length=2, frame_step=1,
          padding='reverse_causal',
      )


class FFTTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      itertools.product(
          (((2, 3, 32), -1), ((2, 3, 5, 32), -2)),
          (31, 32, 33),
          ('center', 'right'),
      )
  )
  def test_fft(self, shape_axis, fft_length, padding):
    shape, axis = shape_axis
    x = random_sequence(
        *shape, dtype=jnp.complex64, low_length=1
    )
    l = dsp.FFT(
        fft_length=fft_length, axis=axis,
        padding=padding,
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    channel_shape = list(shape[2:])
    channel_shape[axis] = fft_length
    self.assertEqual(
        l.get_output_shape(tuple(shape[2:])),
        tuple(channel_shape),
    )
    # Complex outputs: skip gradient test.
    y = self.verify_contract(l, x, test_gradients=False)

    def apply_fft(values):
      values = _pad_or_truncate_for_fft(
          values, padding, axis, fft_length
      )
      return np.fft.fft(values, n=fft_length, axis=axis)

    y_expected = x.apply_values(apply_fft).mask_invalid()
    self.assertSequencesClose(
        y, y_expected, atol=1e-5, rtol=1e-5
    )


class IFFTTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      itertools.product(
          (((2, 3, 32), -1), ((2, 3, 5, 32), -2)),
          (31, 32, 33, None),
          ('center', 'right'),
      )
  )
  def test_ifft(self, shape_axis, frame_length, padding):
    shape, axis = shape_axis
    fft_length = shape[axis]
    x = random_sequence(
        *shape, dtype=jnp.complex64
    )
    l = dsp.IFFT(
        fft_length=fft_length,
        frame_length=frame_length,
        axis=axis, padding=padding,
    )
    if frame_length is None:
      frame_length = fft_length
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    channel_shape = list(shape[2:])
    channel_shape[axis] = frame_length
    self.assertEqual(
        l.get_output_shape(tuple(shape[2:])),
        tuple(channel_shape),
    )
    # Complex outputs: skip gradient test.
    y = self.verify_contract(l, x, test_gradients=False)

    def apply_fft(values):
      values = np.fft.ifft(
          values, n=fft_length, axis=axis
      )
      return _pad_or_truncate_for_fft(
          values, padding, axis, frame_length
      )

    y_expected = x.apply_values(apply_fft).mask_invalid()
    self.assertSequencesClose(
        y, y_expected, atol=1e-5, rtol=1e-5
    )


class RFFTTest(test_utils.SequenceLayerTest):

  def _run_rfft_test(
      self, shape_axis, fft_length, padding, dtype
  ):
    shape, axis = shape_axis
    x = random_sequence(*shape, dtype=dtype)
    l = dsp.RFFT(
        fft_length=fft_length, axis=axis,
        padding=padding,
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    channel_shape = list(shape[2:])
    channel_shape[axis] = fft_length // 2 + 1
    self.assertEqual(
        l.get_output_shape(tuple(shape[2:])),
        tuple(channel_shape),
    )
    # Real→complex output: skip gradient test.
    y = self.verify_contract(l, x, test_gradients=False)

    def apply_fft(values):
      values = _pad_or_truncate_for_fft(
          values, padding, axis, fft_length
      )
      return np.fft.rfft(
          values, n=fft_length, axis=axis
      )

    y_expected = x.apply_values(apply_fft).mask_invalid()
    self.assertSequencesClose(
        y, y_expected, atol=1e-5, rtol=1e-5
    )

  @parameterized.parameters(
      itertools.product(
          (((2, 3, 32), -1), ((2, 3, 5, 32), -2)),
          (31, 32, 33),
          ('center', 'right'),
      )
  )
  def test_rfft(self, shape_axis, fft_length, padding):
    self._run_rfft_test(
        shape_axis, fft_length, padding,
        dtype=jnp.float32,
    )

  def test_rfft_bfloat16(self):
    # bfloat16→complex: skip gradient test.
    shape_axis = ((2, 3, 32), -1)
    shape, axis = shape_axis
    x = random_sequence(*shape, dtype=jnp.bfloat16)
    l = dsp.RFFT(
        fft_length=31, axis=-1, padding='center',
    )
    self.verify_contract(l, x, test_gradients=False)


class IRFFTTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      itertools.product(
          (((2, 3, 17), -1), ((2, 3, 5, 17), -2)),
          (31, 32, 33, None),
          (32, None),
          ('center', 'right'),
      )
  )
  def test_irfft(
      self, shape_axis, frame_length, fft_length, padding
  ):
    shape, axis = shape_axis
    x = random_sequence(
        *shape, dtype=jnp.complex64
    )
    l = dsp.IRFFT(
        fft_length=fft_length,
        frame_length=frame_length,
        axis=axis, padding=padding,
    )
    if fft_length is None:
      fft_length = 2 * (shape[axis] - 1)
    if frame_length is None:
      frame_length = fft_length

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    channel_shape = list(shape[2:])
    channel_shape[axis] = frame_length
    self.assertEqual(
        l.get_output_shape(tuple(shape[2:])),
        tuple(channel_shape),
    )
    # Complex→real: skip gradient test.
    y = self.verify_contract(l, x, test_gradients=False)

    def apply_fft(values):
      values = np.fft.irfft(
          values, n=fft_length, axis=axis
      )
      return _pad_or_truncate_for_fft(
          values, padding, axis, frame_length
      )

    y_expected = x.apply_values(apply_fft).mask_invalid()
    self.assertSequencesClose(
        y, y_expected, atol=1e-5, rtol=1e-5
    )


class FFTInverseTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      itertools.product(
          (
              ((2, 3, 31), -1),
              ((2, 3, 5, 32), -1),
              ((2, 3, 5, 33), -2),
          ),
          (31, 32, 33),
          ('rfft', 'fft'),
          ('center', 'right'),
      )
  )
  def test_fft_inverse(
      self, shape_axis, fft_length, mode, padding
  ):
    shape, axis = shape_axis
    if mode == 'rfft':
      dtype = jnp.float32
      forward_cls = dsp.RFFT
      backward_cls = dsp.IRFFT
    else:
      dtype = jnp.complex64
      forward_cls = dsp.FFT
      backward_cls = dsp.IFFT

    x = random_sequence(*shape, dtype=dtype)
    frame_length = x.shape[axis]

    forward = forward_cls(
        fft_length=fft_length, axis=axis,
        padding=padding,
    )
    backward = backward_cls(
        fft_length=fft_length,
        frame_length=frame_length,
        axis=axis, padding=padding,
    )
    forward.eval()
    backward.eval()

    y_a = forward.layer(x)
    y_ba = backward.layer(y_a)
    y_aba = forward.layer(y_ba)
    y_baba = backward.layer(y_aba)

    # A B A = A
    self.assertSequencesClose(
        y_a, y_aba, atol=1e-5, rtol=1e-3
    )
    # B A B = B
    self.assertSequencesClose(
        y_ba, y_baba, atol=1e-5, rtol=1e-3
    )


class STFTTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      itertools.product(
          (True, False),
          (1, 2, 3, 4),
          (1, 2),
          (2, 3),
          (
              'causal_valid',
              'valid',
              'same',
              'reverse_causal_valid',
              'causal',
              'reverse_causal',
          ),
          ('center', 'right'),
      )
  )
  def test_stft(
      self,
      output_magnitude,
      frame_length,
      frame_step,
      fft_length,
      time_padding,
      fft_padding,
  ):
    batch_size, time = 2, 20
    x = random_sequence(batch_size, time)
    l = dsp.STFT(
        output_magnitude=output_magnitude,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        time_padding=time_padding,
        fft_padding=fft_padding,
    )
    self.assertEqual(l.block_size, frame_step)
    self.assertEqual(
        l.supports_step,
        time_padding in (
            'causal_valid',
            'reverse_causal_valid',
            'causal',
            'reverse_causal',
        ),
    )
    self.assertEqual(1 / l.output_ratio, frame_step)
    match time_padding:
      case 'causal_valid' | 'causal':
        expected_input_latency = 0
      case 'reverse_causal_valid' | 'reverse_causal':
        expected_input_latency = frame_length - 1
      case 'semicausal':
        expected_input_latency = (
            (frame_length - 1)
            - max(0, frame_length - frame_step)
        )
      case _:
        expected_input_latency = 0
    self.assertEqual(
        l.input_latency, expected_input_latency
    )
    self.assertEqual(
        l.output_latency,
        expected_input_latency // frame_step,
    )
    self.assertEqual(
        l.get_output_shape(()),
        (fft_length // 2 + 1,),
    )
    # Complex output when output_magnitude=False.
    skip_grad = not output_magnitude
    self.verify_contract(
        l, x, test_gradients=not skip_grad
    )

  @parameterized.product(
      channel_shape=((1,), (2,), (2, 3))
  )
  def test_multichannel(self, channel_shape):
    batch_size, time = 2, 20
    x = random_sequence(
        batch_size, time,
        low_length=time // 2, *channel_shape
    )
    l = dsp.STFT(
        output_magnitude=True,
        frame_length=8,
        frame_step=3,
        fft_length=8,
        time_padding='causal',
        fft_padding='right',
    )
    self.assertEqual(l.block_size, 3)
    self.assertTrue(l.supports_step)
    self.verify_contract(l, x)


class InverseSTFTTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      itertools.product(
          (1, 2, 3, 4),
          (1, 2),
          (2, 3),
          ('causal', 'valid'),
          ('center', 'right'),
      )
  )
  def test_inverse_stft(
      self,
      frame_length,
      frame_step,
      fft_length,
      time_padding,
      fft_padding,
  ):
    if frame_length < frame_step:
      self.skipTest('frame_length < frame_step')
    batch_size, time = 2, 20
    x = random_sequence(
        batch_size, time, fft_length // 2 + 1,
        dtype=jnp.complex64,
    )
    l = dsp.InverseSTFT(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=signal.inverse_stft_window_fn(
            frame_step, signal.hann_window
        ),
        time_padding=time_padding,
        fft_padding=fft_padding,
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, frame_step)
    self.assertEqual(
        l.supports_step, time_padding == 'causal'
    )
    self.assertEqual(
        l.get_output_shape((fft_length // 2 + 1,)), ()
    )
    # Complex input: skip gradient test.
    self.verify_contract(l, x, test_gradients=False)

  @parameterized.product(
      channel_shape=((1,), (2,), (2, 3))
  )
  def test_multichannel(self, channel_shape):
    batch_size, time = 2, 20
    frame_length, frame_step, fft_length = 8, 3, 8
    x = random_sequence(
        batch_size, time, fft_length // 2 + 1,
        *channel_shape,
        low_length=time // 2, dtype=jnp.complex64,
    )
    l = dsp.InverseSTFT(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=signal.inverse_stft_window_fn(
            frame_step, signal.hann_window
        ),
        time_padding='causal',
        fft_padding='right',
    )
    self.assertEqual(l.block_size, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.output_ratio, frame_step)
    # Complex input: skip gradient test.
    self.verify_contract(l, x, test_gradients=False)


class STFTPerfectReconstructionTest(
    test_utils.SequenceLayerTest
):

  @parameterized.parameters(
      itertools.product(
          (
              (32, 16, 32),
              (32, 8, 32),
              (32, 8, 64),
              (50, 15, 64),
          ),
          (signal.hann_window, signal.hamming_window),
          ('center', 'right'),
      )
  )
  def test_stft_perfect_reconstruction(
      self,
      length_step_fft,
      window_fn,
      fft_padding,
  ):
    frame_length, frame_step, fft_length = (
        length_step_fft
    )
    batch_size = 2
    overlap = math.ceil(frame_length / frame_step)
    time = 2 * overlap * frame_length + 3

    time_padding = (
        types.PaddingMode.SEMICAUSAL_FULL.value
    )

    x = random_sequence(
        batch_size, time, dtype=jnp.float32
    )
    forward = dsp.STFT(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=signal.hann_window,
        time_padding=time_padding,
        fft_padding=fft_padding,
    )
    backward = dsp.InverseSTFT(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=signal.inverse_stft_window_fn(
            frame_step, signal.hann_window
        ),
        time_padding=time_padding,
        fft_padding=fft_padding,
    )
    forward.eval()
    backward.eval()

    y = forward.layer(x)
    x_hat = backward.layer(y)

    size = x.shape[1]
    self.assertLess(size, x_hat.shape[1])

    mask_and = jnp.logical_and(
        x.mask, x_hat.mask[:, :size]
    )
    np.testing.assert_allclose(
        x.values * mask_and,
        x_hat.values[:, :size] * mask_and,
        atol=1e-5, rtol=1e-5,
    )

    mask_xor = jnp.logical_xor(
        jnp.pad(
            x.mask,
            ((0, 0), (0, x_hat.shape[1] - size)),
        ),
        x_hat.mask,
    )
    x_hat = x_hat.mask_invalid()
    self.assertTrue(
        jnp.all(abs(x_hat.values[mask_xor]) < 1e-6)
    )


class LinearToMelSpectrogramTest(
    test_utils.SequenceLayerTest
):

  def test_linear_to_mel_spectrogram(self):
    batch_size, time, num_spectrogram_bins = 2, 3, 5
    x = random_sequence(
        batch_size, time, num_spectrogram_bins
    )
    l = dsp.LinearToMelSpectrogram(
        num_mel_bins=8,
        sample_rate=400,
        lower_edge_hertz=1.0,
        upper_edge_hertz=200.0,
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape((num_spectrogram_bins,)),
        (8,),
    )
    self.verify_contract(l, x)


if __name__ == '__main__':
  test_utils.main()
