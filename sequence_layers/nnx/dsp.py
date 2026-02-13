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
"""NNX DSP layers."""

import abc
import fractions
import math
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import convolution
from sequence_layers.jax import signal
from sequence_layers.jax import types as jax_types
from sequence_layers.nnx import types


class Delay(types.PreservesShape, types.PreservesType, types.SequenceLayer):
  """A layer that delays its input by ``length`` timesteps.

  In contrast to :class:`Lookahead`, which drops ``length`` timesteps from
  the start of the sequence, :class:`Delay` inserts ``length`` invalid
  timesteps at the start of the sequence.
  """

  def __init__(
      self,
      *,
      length: int,
      delay_layer_output: bool = True,
  ):
    super().__init__()
    if length < 0:
      raise ValueError(
          f'Expected nonnegative delay length. Got: {length}'
      )
    self._length = length
    self._delay_layer_output = delay_layer_output

  @property
  def input_latency(self) -> int:
    return self._length

  @property
  def output_latency(self) -> int:
    if self._delay_layer_output:
      return 0
    else:
      return self._length

  @property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    if self._delay_layer_output:
      return {0: (-self._length, -self._length)}
    else:
      return {0: (0, 0)}

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    if not self._length:
      return ()
    return types.Sequence(
        jnp.zeros(
            (batch_size, self._length) + input_spec.shape,
            dtype=input_spec.dtype,
        ),
        jnp.zeros(
            (batch_size, self._length), types.MASK_DTYPE
        ),
    )

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if not self._length:
      return x, state

    state = state.concatenate(x)
    y_values, state_values = jnp.split(
        state.values, [x.shape[1]], axis=1
    )
    y_mask, state_mask = jnp.split(
        state.mask, [x.shape[1]], axis=1
    )
    y = types.Sequence(y_values, y_mask)
    state = types.Sequence(state_values, state_mask)
    return y, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if self._delay_layer_output:
      return x.pad_time(self._length, 0, valid=False)
    else:
      return x


class Lookahead(
    types.PreservesShape, types.PreservesType, types.SequenceLayer
):
  """A layer that drops the first ``length`` timesteps from its input.

  In contrast to :class:`Delay`, which inserts ``length`` invalid timesteps
  at the start, :class:`Lookahead` drops ``length`` timesteps from the
  start of the sequence.
  """

  def __init__(
      self,
      *,
      length: int,
      preserve_length_in_layer: bool = False,
  ):
    super().__init__()
    if length < 0:
      raise ValueError(
          f'Expected nonnegative lookahead length. Got: {length}'
      )
    self._length = length
    self._preserve_length_in_layer = preserve_length_in_layer

  @property
  def input_latency(self) -> int:
    return 0

  @property
  def output_latency(self) -> int:
    return self._length

  @property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    return {0: (self._length, self._length)}

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    if not self._length:
      return ()
    return jnp.full(
        (batch_size,),
        jnp.array(self._length + 1, jnp.int32),
    )

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if not self._length:
      return x, state
    assert x.shape[1] > 0

    increments = jnp.cumsum(x.mask, axis=1)
    countdown = jnp.maximum(
        0, state[:, jnp.newaxis] - increments
    )
    mask = jnp.logical_and(x.mask, countdown == 0)
    y = types.Sequence(x.values, mask)
    state = countdown[:, -1]
    return y, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if not self._length:
      return x

    x = x[:, self._length:]
    if self._preserve_length_in_layer:
      return x.pad_time(0, self._length, valid=False)
    else:
      return x


class Window(
    types.PreservesShape, types.PreservesType, types.Stateless
):
  """Applies a window function along a channel axis.

  Used in STFT/InverseSTFT pipelines.
  """

  def __init__(
      self,
      *,
      axis: int,
      window_fn: Callable[..., jax.Array | np.ndarray] = (
          signal.hann_window
      ),
  ):
    super().__init__()
    self._axis = axis
    self._window_fn = window_fn

  def _get_axis(self, x: types.Sequence) -> int:
    axis = self._axis
    if axis < 0:
      axis = x.ndim + axis

    if axis in (0, 1):
      raise ValueError(
          'The axis cannot be 0 (batch) or 1 (time)'
          f' (got {self._axis}).'
      )

    if axis >= x.ndim:
      raise ValueError(
          f'Axis {self._axis} larger than inputs ndim'
          f' ({x.ndim})'
      )

    return axis

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    axis = self._get_axis(x)

    window = self._window_fn(x.shape[axis], dtype=x.dtype)
    shape_broadcast = [jnp.newaxis] * x.ndim
    shape_broadcast[axis] = slice(None, None)

    return x.apply_values(
        lambda v: v * window[tuple(shape_broadcast)]
    )


class Frame(types.PreservesType, types.SequenceLayer):
  """Produce a sequence of overlapping frames of the input sequence.

  Converts a [batch, time, ...channels] input into
  [batch, num_frames, frame_length, ...channels] output.
  """

  def __init__(
      self,
      *,
      frame_length: int,
      frame_step: int,
      padding: tuple[int, int] | jax_types.PaddingModeString = (
          types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ),
      explicit_padding_is_same_like: bool = False,
  ):
    super().__init__()
    if frame_length <= 0:
      raise ValueError(
          f'frame_length must be positive, got: {frame_length}'
      )
    if frame_step <= 0:
      raise ValueError(
          f'frame_step must be positive, got: {frame_step}'
      )

    if isinstance(padding, str):
      padding = jax_types.validate_padding(padding)
    elif sum(padding) != frame_length - 1:
      raise NotImplementedError(
          f'{padding=} must sum to {frame_length - 1=}'
      )

    self._frame_length = frame_length
    self._frame_step = frame_step
    self._padding = padding
    self._explicit_padding_is_same_like = explicit_padding_is_same_like

  @property
  def supports_step(self) -> bool:
    if isinstance(self._padding, str):
      return self._padding in (
          types.PaddingMode.CAUSAL_VALID.value,
          types.PaddingMode.REVERSE_CAUSAL_VALID.value,
          types.PaddingMode.CAUSAL.value,
          types.PaddingMode.REVERSE_CAUSAL.value,
          types.PaddingMode.SEMICAUSAL.value,
      )
    else:
      past_pad, future_pad = self._padding
      return past_pad + future_pad == self._frame_length - 1

  @property
  def block_size(self) -> int:
    return self._frame_step

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self._frame_step)

  @property
  def input_latency(self) -> int:
    if not isinstance(self._padding, str):
      past_pad, _ = self._padding
      return max(0, (self._frame_length - 1) - past_pad)
    match self._padding:
      case (
          types.PaddingMode.CAUSAL_VALID.value
          | types.PaddingMode.CAUSAL.value
          | types.PaddingMode.SEMICAUSAL.value
      ):
        return 0
      case (
          types.PaddingMode.REVERSE_CAUSAL_VALID.value
          | types.PaddingMode.REVERSE_CAUSAL.value
      ):
        return self._frame_length - 1
      case types.PaddingMode.SEMICAUSAL_FULL.value:
        return self._frame_step - 1
      case _:
        return 0

  @property
  def output_latency(self) -> int:
    match self._padding:
      case types.PaddingMode.SEMICAUSAL_FULL.value:
        return 0
      case _:
        return super().output_latency

  @property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    return convolution.conv_receptive_field_per_step(
        self._frame_length, self._frame_step, 1, self._padding
    )

  @property
  def _buffer_width(self) -> int:
    if not isinstance(self._padding, str):
      past_pad, future_pad = self._padding
      assert past_pad + future_pad == self._frame_length - 1
      return (
          self._frame_length
          - 1
          - (future_pad % self._frame_step)
      )

    match self._padding:
      case types.PaddingMode.SEMICAUSAL.value:
        return max(
            self._frame_length - self._frame_step, 0
        )
      case (
          types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ):
        return (
            (self._frame_length - 1)
            // self._frame_step
            * self._frame_step
        )
      case (
          types.PaddingMode.CAUSAL.value
          | types.PaddingMode.CAUSAL_VALID.value
      ):
        return self._frame_length - 1
      case _:
        raise NotImplementedError(
            f'Unsupported padding mode: {self._padding}'
        )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return (self._frame_length,) + tuple(input_shape)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    if not (buffer_width := self._buffer_width):
      return ()

    match self._padding:
      case (
          types.PaddingMode.CAUSAL_VALID.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
          | types.PaddingMode.SEMICAUSAL_FULL.value
      ):
        mask = jnp.ones(
            (batch_size, buffer_width), dtype=types.MASK_DTYPE
        )
      case (
          types.PaddingMode.CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.SEMICAUSAL.value
      ):
        mask = jnp.zeros(
            (batch_size, buffer_width), dtype=types.MASK_DTYPE
        )
      case (unused_pad_left, unused_pad_right):
        if self._explicit_padding_is_same_like:
          mask = jnp.zeros(
              (batch_size, buffer_width),
              dtype=types.MASK_DTYPE,
          )
        else:
          mask = jnp.ones(
              (batch_size, buffer_width),
              dtype=types.MASK_DTYPE,
          )
      case _:
        raise ValueError(
            'Stepwise processing is not supported with'
            f' padding: {self._padding}'
        )

    return types.MaskedSequence(
        jnp.zeros(
            (batch_size, buffer_width) + input_spec.shape,
            dtype=input_spec.dtype,
        ),
        mask,
    )

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if self._frame_length > 1:
      x = x.mask_invalid()

    if buffer_width := self._buffer_width:
      state = state.concatenate(x)
    else:
      state = x

    values = signal.frame(
        state.values,
        frame_length=self._frame_length,
        frame_step=self._frame_step,
        pad_mode=types.PaddingMode.VALID.value,
        axis=1,
    )
    mask = convolution.compute_conv_mask(
        state.mask,
        kernel_size=self._frame_length,
        stride=self._frame_step,
        dilation_rate=1,
        padding=self._padding,
        is_step=True,
    )

    if buffer_width:
      state = state[:, -buffer_width:]
    else:
      state = ()

    return types.Sequence(values, mask), state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if self._frame_length > 1:
      x = x.mask_invalid()

    values = signal.frame(
        x.values,
        frame_length=self._frame_length,
        frame_step=self._frame_step,
        pad_mode=self._padding,
        axis=1,
    )
    mask = convolution.compute_conv_mask(
        x.mask,
        kernel_size=self._frame_length,
        stride=self._frame_step,
        dilation_rate=1,
        padding=self._padding,
        is_step=False,
    )

    result_type = (
        types.Sequence if self._frame_length > 1 else type(x)
    )
    return result_type(values, mask)


class OverlapAdd(types.PreservesType, types.SequenceLayer):
  """Overlap-adds windows of [b, t, frame_length, ...].

  For a [b, ti, frame_length, ...] input, the resulting sequence has
  shape [b, to, ...], where:

    to = (ti - 1) * frame_step + frame_length

  Since step() can only produce frame_step samples at a time, layer()
  trims the final ``frame_length - frame_step`` timesteps to match.
  """

  def __init__(
      self,
      *,
      frame_length: int,
      frame_step: int,
      padding: jax_types.PaddingModeString = (
          types.PaddingMode.VALID.value
      ),
  ):
    super().__init__()
    if frame_length <= 0:
      raise ValueError(
          f'frame_length must be positive, got: {frame_length}'
      )
    if frame_step <= 0:
      raise ValueError(
          f'frame_step must be positive, got: {frame_step}'
      )
    if frame_length < frame_step:
      raise ValueError(
          'frame_length must be at least frame_step.'
      )

    padding = jax_types.validate_padding(padding)
    if padding not in (
        types.PaddingMode.CAUSAL.value,
        types.PaddingMode.VALID.value,
        types.PaddingMode.SEMICAUSAL_FULL.value,
    ):
      raise ValueError(
          f'Unsupported padding mode: {padding}'
      )

    self._frame_length = frame_length
    self._frame_step = frame_step
    self._padding = padding

  @property
  def supports_step(self) -> bool:
    return self._padding == types.PaddingMode.CAUSAL.value

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self._frame_step)

  @property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    if (
        self._padding
        == types.PaddingMode.SEMICAUSAL_FULL.value
    ):
      start = 0
      end = (
          math.ceil(
              float(self._frame_length) / self._frame_step
          )
          - 1
      )
      return {0: (start, end)}
    return convolution.transpose_conv_receptive_field_per_step(
        self._frame_length,
        self._frame_step,
        1,
        self._padding,
    )

  def _validate_input_shape(
      self, input_shape: types.ShapeLike
  ) -> None:
    if (
        not input_shape
        or input_shape[0] != self._frame_length
    ):
      raise ValueError(
          'OverlapAdd expects input of shape'
          f' (frame_length, ...), got: {input_shape=}'
      )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    self._validate_input_shape(input_shape)
    return tuple(input_shape[1:])

  @property
  def _buffer_width(self) -> int:
    return max(
        0, self._frame_length - self._frame_step
    )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    self._validate_input_shape(input_spec.shape)
    if buffer_width := self._buffer_width:
      output_spec = self.get_output_spec(
          input_spec, constants=constants
      )
      return jnp.zeros(
          (batch_size, buffer_width) + output_spec.shape,
          dtype=output_spec.dtype,
      )
    else:
      return ()

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    self._validate_input_shape(x.channel_spec.shape)

    if self._frame_length > 1:
      x = x.mask_invalid()

    # Transpose [num_frames, frame_length] to end.
    if x.ndim > 3:
      values = jnp.moveaxis(
          x.values, (1, 2), (-2, -1)
      )
    else:
      values = x.values
    values = signal.overlap_and_add(
        values, self._frame_step
    )
    if x.ndim > 3:
      values = jnp.moveaxis(values, -1, 1)

    mask = convolution.compute_conv_transpose_mask(
        x.mask,
        self._frame_length,
        self._frame_step,
        dilation_rate=1,
        padding=self._padding,
    )

    trim = max(
        self._frame_length - self._frame_step, 0
    )
    match self._padding:
      case types.PaddingMode.CAUSAL.value:
        if trim:
          values = values[:, :-trim]

      case types.PaddingMode.SEMICAUSAL_FULL.value:
        if trim:
          values = values[:, trim:]
          mask = mask[:, trim:]
        size = min(values.shape[1], mask.shape[1])
        return types.Sequence(
            values[:, :size], mask[:, :size]
        )

      case _:
        pass

    return types.Sequence(values, mask)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if self._frame_length > 1:
      x = x.mask_invalid()

    # Transpose [num_frames, frame_length] to end.
    if x.ndim > 3:
      values = jnp.moveaxis(
          x.values, (1, 2), (-2, -1)
      )
    else:
      values = x.values
    values = signal.overlap_and_add(
        values, self._frame_step
    )
    if x.ndim > 3:
      values = jnp.moveaxis(values, -1, 1)

    mask = convolution.compute_conv_transpose_mask(
        x.mask,
        self._frame_length,
        self._frame_step,
        dilation_rate=1,
        padding=self._padding,
    )

    if self._buffer_width:
      time = x.shape[1]

      # Pad state to the length of the overlap-add output.
      paddings = [(0, 0)] * state.ndim
      paddings[1] = (
          0,
          values.shape[1] - self._buffer_width,
      )
      state = jnp.pad(state, paddings)

      # Add previous overlap into values.
      values = values + state

      output_samples = self._frame_step * time
      values, state = jnp.split(
          values, [output_samples], axis=1
      )

    return types.Sequence(values, mask), state


# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------

FFTPaddingString = Literal['center', 'right']
_DEFAULT_FFT_PADDING: FFTPaddingString = 'right'


def _validate_and_normalize_axis(
    axis: int,
    input_shape: types.Shape,
) -> int:
  """Normalizes user-provided axis; rejects batch/time."""
  if axis < 0:
    axis += len(input_shape)
  if axis < 0 or axis > len(input_shape) - 1:
    raise ValueError(
        f'Axis out of range {axis=} with {input_shape=}.'
    )
  if axis in (0, 1):
    raise ValueError(
        'Computing FFTs over the batch or time dimension '
        f'is not allowed. Got: {axis}'
    )
  return axis


def _pad_or_truncate_for_fft(
    x: types.Sequence,
    normalized_axis: int,
    required_input_length: int,
    padding: str,
) -> types.Sequence:
  """Pads or truncates the sequence for an FFT/IFFT."""
  assert 1 < normalized_axis < x.ndim
  input_dim = x.shape[normalized_axis]
  if input_dim == required_input_length:
    return x
  if input_dim < required_input_length:
    pad_amount = required_input_length - input_dim
    if padding == 'center':
      pad_left = pad_amount // 2
      pad_right = pad_amount - pad_left
    else:
      assert padding == 'right'
      pad_left = 0
      pad_right = pad_amount
    paddings = [(0, 0)] * x.ndim
    paddings[normalized_axis] = (pad_left, pad_right)
    return x.apply_values_masked(jnp.pad, paddings)
  else:
    assert input_dim > required_input_length

    def slice_in_dim(v, start, length):
      return jax.lax.slice_in_dim(
          v,
          start_index=start,
          limit_index=start + length,
          axis=normalized_axis,
      )

    if padding == 'center':
      trim_left = (
          (input_dim - required_input_length) // 2
      )
      return x.apply_values(
          slice_in_dim, trim_left, required_input_length
      )
    else:
      assert padding == 'right'
      return x.apply_values(
          slice_in_dim, 0, required_input_length
      )


def _validate_fft(fft_length: int | None, padding: str):
  if fft_length is not None and fft_length <= 0:
    raise ValueError(
        f'fft_length must be positive, got: {fft_length}'
    )
  if padding not in ('center', 'right'):
    raise ValueError(
        f'padding must be "center" or "right", got: '
        f'{padding}'
    )


# ---------------------------------------------------------------------------
# FFT base class
# ---------------------------------------------------------------------------


class FFTBase(types.Stateless, metaclass=abc.ABCMeta):
  """A base class for shared FFT logic."""

  @property
  @abc.abstractmethod
  def _axis(self) -> int: ...

  @property
  @abc.abstractmethod
  def _fft_length(self) -> int | None: ...

  @property
  @abc.abstractmethod
  def _padding(self) -> str: ...

  @abc.abstractmethod
  def _get_fft_fn(
      self,
  ) -> Callable[..., types.Sequence]: ...

  @abc.abstractmethod
  def _get_output_length(self, input_size: int) -> int: ...

  def _get_fft_length_resolved(
      self, input_size: int
  ) -> int:
    return self._fft_length or input_size

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    input_shape = list(input_shape)
    axis = _validate_and_normalize_axis(
        self._axis,
        (None, None) + tuple(input_shape),
    )
    axis -= 2
    input_shape[axis] = self._get_output_length(
        input_shape[axis]
    )
    return tuple(input_shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if x.ndim <= 2:
      raise ValueError(
          'FFT requires an input of rank at least 3.'
      )
    axis = _validate_and_normalize_axis(
        self._axis, x.shape
    )
    fft_fn = self._get_fft_fn()
    return fft_fn(x, axis=axis)


# ---------------------------------------------------------------------------
# FFT / IFFT / RFFT / IRFFT
# ---------------------------------------------------------------------------


class FFT(types.PreservesType, FFTBase):
  """Applies an FFT to a channels dimension."""

  def __init__(
      self,
      *,
      fft_length: int | None = None,
      axis: int = -1,
      padding: FFTPaddingString = _DEFAULT_FFT_PADDING,
  ):
    super().__init__()
    _validate_fft(fft_length, padding)
    self._fft_length_val = fft_length
    self._axis_val = axis
    self._padding_val = padding

  @property
  def _axis(self) -> int:
    return self._axis_val

  @property
  def _fft_length(self) -> int | None:
    return self._fft_length_val

  @property
  def _padding(self) -> str:
    return self._padding_val

  def _get_output_length(self, input_size: int) -> int:
    return self._fft_length_val or input_size

  def _get_fft_fn(self):
    def fft_fn(x, axis):
      required_length = self._get_output_length(
          x.shape[axis]
      )
      x = _pad_or_truncate_for_fft(
          x, axis, required_length, self._padding_val
      )
      return x.apply_values(jnp.fft.fft, axis=axis)

    return fft_fn


class IFFT(types.PreservesType, FFTBase):
  """Applies an IFFT to a channels dimension."""

  def __init__(
      self,
      *,
      fft_length: int | None = None,
      frame_length: int | None = None,
      axis: int = -1,
      padding: FFTPaddingString = _DEFAULT_FFT_PADDING,
  ):
    super().__init__()
    _validate_fft(fft_length, padding)
    self._fft_length_val = fft_length
    self._frame_length = frame_length
    self._axis_val = axis
    self._padding_val = padding

  @property
  def _axis(self) -> int:
    return self._axis_val

  @property
  def _fft_length(self) -> int | None:
    return self._fft_length_val

  @property
  def _padding(self) -> str:
    return self._padding_val

  def _get_output_length(self, input_size: int) -> int:
    return self._frame_length or input_size

  def _get_fft_fn(self):
    def ifft_fn(a, axis):
      a = a.apply_values(jnp.fft.ifft, axis=axis)
      required_length = self._get_output_length(
          a.shape[axis]
      )
      return _pad_or_truncate_for_fft(
          a, axis, required_length, self._padding_val
      )

    return ifft_fn


class RFFT(FFTBase):
  """Applies an RFFT to a channels dimension."""

  def __init__(
      self,
      *,
      fft_length: int | None = None,
      axis: int = -1,
      padding: FFTPaddingString = _DEFAULT_FFT_PADDING,
  ):
    super().__init__()
    _validate_fft(fft_length, padding)
    self._fft_length_val = fft_length
    self._axis_val = axis
    self._padding_val = padding

  @property
  def _axis(self) -> int:
    return self._axis_val

  @property
  def _fft_length(self) -> int | None:
    return self._fft_length_val

  @property
  def _padding(self) -> str:
    return self._padding_val

  def _get_fft_length_resolved(
      self, input_size: int
  ) -> int:
    return self._fft_length_val or input_size

  def _get_output_length(self, input_size: int) -> int:
    return (
        self._get_fft_length_resolved(input_size) // 2
        + 1
    )

  def _get_fft_fn(self):
    def rfft(a, axis=-1):
      fft_length = self._get_fft_length_resolved(
          a.shape[axis]
      )
      a = _pad_or_truncate_for_fft(
          a, axis, fft_length, self._padding_val
      )
      rfft_inner = lambda v: jnp.fft.rfft(
          v, n=fft_length, axis=axis
      )
      if a.dtype == jnp.bfloat16:
        return a.apply_values(
            lambda v: rfft_inner(v.astype(jnp.float32))
        )
      return a.apply_values(rfft_inner)

    return rfft

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    match input_dtype:
      case jnp.bfloat16 | jnp.float16 | jnp.float32:
        return jnp.complex64
      case jnp.float64:
        return jnp.complex128
      case _:
        raise ValueError(
            f'Unsupported input dtype: {input_dtype}'
        )


class IRFFT(FFTBase):
  """Applies an IRFFT to a channels dimension."""

  def __init__(
      self,
      *,
      fft_length: int | None = None,
      frame_length: int | None = None,
      axis: int = -1,
      padding: FFTPaddingString = _DEFAULT_FFT_PADDING,
  ):
    super().__init__()
    _validate_fft(fft_length, padding)
    self._fft_length_val = fft_length
    self._frame_length = frame_length
    self._axis_val = axis
    self._padding_val = padding

  @property
  def _axis(self) -> int:
    return self._axis_val

  @property
  def _fft_length(self) -> int | None:
    return self._fft_length_val

  @property
  def _padding(self) -> str:
    return self._padding_val

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    match input_dtype:
      case jnp.complex64:
        return jnp.float32
      case jnp.complex128:
        return jnp.float64
      case _:
        raise ValueError(
            f'Unsupported input dtype: {input_dtype}'
        )

  def _get_fft_length_resolved(
      self, input_size: int
  ) -> int:
    return self._fft_length_val or (input_size - 1) * 2

  def _get_output_length(self, input_size: int) -> int:
    return (
        self._frame_length
        or self._get_fft_length_resolved(input_size)
    )

  def _get_fft_fn(self):
    def irfft_fn(a, axis=-1):
      fft_length = self._get_fft_length_resolved(
          a.shape[axis]
      )
      a = a.apply_values(
          jnp.fft.irfft, axis=axis, n=fft_length
      )
      required_length = (
          self._frame_length or a.shape[axis]
      )
      return _pad_or_truncate_for_fft(
          a, axis, required_length, self._padding_val
      )

    return irfft_fn


# ---------------------------------------------------------------------------
# STFT / InverseSTFT
# ---------------------------------------------------------------------------


class STFT(types.SequenceLayer):
  """Computes the Short-time Fourier Transform.

  For an input [b, t, ...], produces
  [b, t // frame_step, fft_length // 2 + 1, ...].
  """

  def __init__(
      self,
      *,
      frame_length: int,
      frame_step: int,
      fft_length: int,
      window_fn: (
          Callable[..., jax.Array | np.ndarray] | None
      ) = signal.hann_window,
      time_padding: jax_types.PaddingModeString = (
          types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ),
      fft_padding: FFTPaddingString = _DEFAULT_FFT_PADDING,
      output_magnitude: bool = False,
  ):
    super().__init__()
    self._frame_length = frame_length
    self._frame_step = frame_step
    self._fft_length = fft_length
    self._window_fn = window_fn
    self._time_padding = jax_types.validate_padding(
        time_padding
    )
    self._fft_padding = fft_padding
    self._output_magnitude = output_magnitude

    self._framer = Frame(
        frame_length=frame_length,
        frame_step=frame_step,
        padding=self._time_padding,
    )
    self._fft = RFFT(
        fft_length=fft_length,
        axis=2,
        padding=fft_padding,
    )

  @property
  def supports_step(self) -> bool:
    return (
        self._framer.supports_step
        and self._fft.supports_step
    )

  @property
  def block_size(self) -> int:
    return self._framer.block_size

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self._framer.output_ratio

  @property
  def input_latency(self) -> int:
    return self._framer.input_latency

  @property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    if self._window_fn:
      window = self._window_fn(self._frame_length)
    else:
      window = np.ones(self._frame_length)
    receptive_field = self._framer.receptive_field
    assert receptive_field is not None
    start, end = receptive_field
    rf_list = np.arange(start, end + 1, dtype=np.int32)

    if self._fft_length < self._frame_length:
      if self._fft_padding == 'center':
        trim_left = (
            (self._frame_length - self._fft_length) // 2
        )
        trim_right = (
            self._frame_length
            - self._fft_length
            - trim_left
        )
      elif self._fft_padding == 'right':
        trim_left = 0
        trim_right = (
            self._frame_length - self._fft_length
        )
      else:
        raise ValueError(
            f'Unsupported FFT padding: {self._fft_padding}'
        )
      trim_mask = np.array(
          [0] * trim_left
          + [1]
          * (
              self._frame_length
              - trim_left
              - trim_right
          )
          + [0] * trim_right
      )
      window = window * trim_mask

    rf_list = rf_list[window > 0]
    if rf_list.size > 0:
      start = int(np.nanmin(rf_list))
      end = int(np.nanmax(rf_list))
    else:
      start = np.inf
      end = -np.inf
    return {0: (start, end)}

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    frame_shape = self._framer.get_output_shape(
        input_shape, constants=constants
    )
    return self._fft.get_output_shape(
        frame_shape, constants=constants
    )

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    fft_dtype = self._fft.get_output_dtype(
        input_dtype, constants=constants
    )
    if self._output_magnitude:
      match fft_dtype:
        case jnp.complex64:
          return jnp.float32
        case jnp.complex128:
          return jnp.float64
        case _:
          raise ValueError(
              f'Unsupported FFT output dtype: {fft_dtype}'
          )
    return fft_dtype

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    return self._framer.get_initial_state(
        batch_size, input_spec, constants=constants
    )

  def _apply_window(
      self, x: types.Sequence
  ) -> types.Sequence:
    if self._window_fn:
      window = self._window_fn(
          self._frame_length
      ).reshape(
          (1, 1, self._frame_length)
          + (1,) * (x.ndim - 3)
      )
      window = jnp.asarray(window, x.dtype)
      return x.apply_values_masked(
          lambda v: v * window
      )
    return x

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    framed, state = self._framer.step(
        x, state, constants=constants
    )
    framed = self._apply_window(framed)
    dft, _ = self._fft.step(
        framed, (), constants=constants
    )
    if self._output_magnitude:
      dft = dft.apply_values_masked(jnp.abs)
    return dft, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    framed = self._framer.layer(
        x, constants=constants
    )
    framed = self._apply_window(framed)
    dft = self._fft.layer(
        framed, constants=constants
    )
    if self._output_magnitude:
      dft = dft.apply_values_masked(jnp.abs)
    return dft


class InverseSTFT(types.SequenceLayer):
  """Computes the inverse Short-time Fourier Transform.

  For an input [b, t, fft_length // 2 + 1, ...], produces
  [b, t * frame_step, ...].
  """

  def __init__(
      self,
      *,
      frame_length: int,
      frame_step: int,
      fft_length: int,
      window_fn: (
          Callable[..., jax.Array | np.ndarray] | None
      ) = signal.hann_window,
      time_padding: jax_types.PaddingModeString = (
          types.PaddingMode.CAUSAL.value
      ),
      fft_padding: FFTPaddingString = _DEFAULT_FFT_PADDING,
  ):
    super().__init__()
    self._frame_length = frame_length
    self._frame_step = frame_step
    self._fft_length = fft_length
    self._window_fn = window_fn
    self._time_padding = jax_types.validate_padding(
        time_padding
    )
    self._fft_padding = fft_padding

    self._overlap_add = OverlapAdd(
        frame_length=frame_length,
        frame_step=frame_step,
        padding=self._time_padding,
    )
    self._irfft = IRFFT(
        fft_length=fft_length,
        frame_length=frame_length,
        axis=2,
        padding=fft_padding,
    )

  @property
  def supports_step(self) -> bool:
    return (
        self._overlap_add.supports_step
        and self._irfft.supports_step
    )

  @property
  def block_size(self) -> int:
    return self._overlap_add.block_size

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self._overlap_add.output_ratio

  @property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    frame_length = self._frame_length
    frame_step = self._frame_step
    fft_length = self._fft_length
    fft_padding = self._fft_padding

    explicit_padding = (
        convolution.transpose_conv_explicit_padding(
            frame_length,
            frame_step,
            1,
            self._time_padding,
        )
    )
    shift_left = (
        frame_length - explicit_padding[0] - 1
    )
    if (pad_amount := frame_length - fft_length) > 0:
      if fft_padding == 'right':
        pad_left = 0
        pad_right = pad_amount
      elif fft_padding == 'center':
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
      else:
        raise ValueError(
            f'Unsupported FFT padding: {fft_padding}'
        )
    else:
      pad_left = 0
      pad_right = 0

    if self._window_fn:
      window = self._window_fn(
          frame_length, dtype=jnp.int32
      )
      if pad_amount > 0:
        window = np.array([
            v
            if pad_left <= i < pad_left + fft_length
            else 0
            for i, v in enumerate(window)
        ])
      window_mask = window != 0
    else:
      window_mask = jnp.ones(
          frame_length, dtype=jnp.bool_
      )

    p = (
        -math.ceil(
            float(explicit_padding[0] + 1) / frame_step
        )
        + 1
    )
    f = (
        math.ceil(float(shift_left) / frame_step) - 1
    )
    rf_per_step = {}
    for i in range(frame_step):
      if (explicit_padding[0] + 1) % frame_step == 0 or (
          i
          < ((explicit_padding[0] + 1)) % frame_step
      ):
        p_i = p
      else:
        p_i = p + 1
      f_i = (
          f
          if i < -shift_left % frame_step
          else f + 1
      )

      if pad_amount > 0:
        padding_mask = np.array(
            [0] * pad_left
            + [1] * fft_length
            + [0] * pad_right,
            dtype=jnp.bool_,
        )
      else:
        padding_mask = jnp.ones(
            frame_length, dtype=jnp.bool_
        )
      rf_frame_index = (
          (i + shift_left)
          - np.arange(p_i, f_i + 1) * frame_step
      )
      rf_frame_index_mask = (window_mask & padding_mask)[
          rf_frame_index
      ]
      rf_list = np.arange(
          p_i, f_i + 1, dtype=np.int32
      )
      rf_list = rf_list[rf_frame_index_mask]
      if rf_list.size > 0:
        p_i = int(np.nanmin(rf_list))
        f_i = int(np.nanmax(rf_list))
        rf_per_step[i] = (
            (p_i, f_i) if p_i <= f_i else None
        )
      else:
        rf_per_step[i] = None
    return rf_per_step

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    irfft_shape = list(
        self._irfft.get_output_shape(
            input_shape, constants=constants
        )
    )
    irfft_shape[0] = self._frame_length
    return self._overlap_add.get_output_shape(
        irfft_shape, constants=constants
    )

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return self._irfft.get_output_dtype(
        input_dtype, constants=constants
    )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    irfft_spec = self._irfft.get_output_spec(
        input_spec, constants=constants
    )
    irfft_shape = list(irfft_spec.shape)
    irfft_shape[0] = self._frame_length
    irfft_spec = types.ShapeDType(
        irfft_shape, irfft_spec.dtype
    )
    return self._overlap_add.get_initial_state(
        batch_size, irfft_spec, constants=constants
    )

  def _apply_window(
      self, irfft: types.Sequence
  ) -> types.Sequence:
    fft_length = irfft.shape[2]
    if fft_length > self._frame_length:
      irfft = irfft.apply_values_masked(
          lambda v: v[:, :, : self._frame_length]
      )
    elif fft_length < self._frame_length:
      pad_amount = self._frame_length - fft_length
      match self._fft_padding:
        case 'right':
          pad_left, pad_right = 0, pad_amount
        case 'center':
          pad_left = pad_amount // 2
          pad_right = pad_amount - pad_left
        case _:
          raise ValueError(
              'Unsupported FFT padding: '
              f'{self._fft_padding}'
          )
      paddings = [(0, 0)] * irfft.ndim
      paddings[2] = (pad_left, pad_right)
      irfft = irfft.apply_values_masked(
          jnp.pad, paddings
      )

    if self._window_fn:
      window = self._window_fn(
          self._frame_length, dtype=irfft.dtype
      ).reshape(
          (1, 1, self._frame_length)
          + (1,) * (irfft.ndim - 3)
      )
      window = jnp.asarray(window, irfft.dtype)
      irfft = irfft.apply_values_masked(
          lambda v: v * window
      )
    return irfft

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if x.ndim < 3:
      raise ValueError(
          'Expected [b, t, num_frequency_bins, ...] '
          f'input, but got {x.shape=}'
      )
    irfft, _ = self._irfft.step(
        x, (), constants=constants
    )
    irfft = self._apply_window(irfft)
    ola, state = self._overlap_add.step(
        irfft, state, constants=constants
    )
    return ola, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if x.ndim < 3:
      raise ValueError(
          'Expected [b, t, num_frequency_bins, ...] '
          f'input, but got {x.shape=}'
      )
    irfft = self._irfft.layer(
        x, constants=constants
    )
    irfft = self._apply_window(irfft)
    ola = self._overlap_add.layer(
        irfft, constants=constants
    )
    return ola


# ---------------------------------------------------------------------------
# LinearToMelSpectrogram
# ---------------------------------------------------------------------------


class LinearToMelSpectrogram(
    types.PreservesType, types.Stateless
):
  """Converts linear-scale spectrogram to mel-scale."""

  def __init__(
      self,
      *,
      num_mel_bins: int,
      sample_rate: float,
      lower_edge_hertz: float,
      upper_edge_hertz: float,
  ):
    super().__init__()
    self._num_mel_bins = num_mel_bins
    self._sample_rate = sample_rate
    self._lower_edge_hertz = lower_edge_hertz
    self._upper_edge_hertz = upper_edge_hertz

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if not input_shape:
      raise ValueError(
          'Requires input with at least rank 1, got: '
          f'{input_shape}'
      )
    return tuple(input_shape[:-1]) + (
        self._num_mel_bins,
    )

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    num_spectrogram_bins = x.shape[-1]
    linear_to_mel = jnp.asarray(
        signal.linear_to_mel_weight_matrix(
            num_mel_bins=self._num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=self._sample_rate,
            lower_edge_hertz=self._lower_edge_hertz,
            upper_edge_hertz=self._upper_edge_hertz,
            dtype=np.float64,
        ),
        x.dtype,
    )
    return x.apply_values_masked(
        lambda v: jnp.einsum(
            '...a,ab->...b', v, linear_to_mel
        )
    )
