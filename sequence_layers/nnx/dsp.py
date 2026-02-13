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

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import signal
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
