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

import fractions
import math
from typing import Callable

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
