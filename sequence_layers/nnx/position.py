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
"""NNX position embeddings and timing signals."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import utils
from sequence_layers.nnx import types


class AddTimingSignal(
    types.PreservesType, types.PreservesShape, types.SequenceLayer
):
  """Adds sinusoids at varying frequencies to the input channels dimension."""

  def __init__(
      self,
      *,
      min_timescale: float = 1.0,
      max_timescale: float = 1.0e4,
      trainable_scale: bool = False,
      axes: int | tuple[int, ...] | None = None,
      param_dtype: types.DType = jnp.float32,
      only_advance_position_for_valid_timesteps: bool = True,
      rngs: nnx.Rngs | None = None,
  ):
    super().__init__()
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale
    self._axes = axes
    self._param_dtype = param_dtype
    self.only_advance_position_for_valid_timesteps = (
        only_advance_position_for_valid_timesteps
    )
    if trainable_scale:
      if rngs is None:
        raise ValueError('rngs is required when trainable_scale=True.')
      self.scale = nnx.Param(
          jnp.ones([], dtype=param_dtype)
      )
    else:
      self.scale = None

  def _check_inputs(self, input_spec: types.ShapeDType):
    if input_spec.dtype not in (
        jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64
    ):
      raise ValueError(
          f'{type(self).__name__} requires floating point argument.'
      )

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    if self.only_advance_position_for_valid_timesteps:
      return {0: (-np.inf, 0)}
    else:
      return {0: (0, 0)}

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    self._check_inputs(input_spec)
    if self.only_advance_position_for_valid_timesteps:
      return jnp.full((batch_size, 1), -1, dtype=jnp.int32)
    else:
      return jnp.zeros((batch_size, 1), dtype=jnp.int32)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    self._check_inputs(x.channel_spec)
    time = x.shape[1]
    target_shape = utils.match_shape_along_axes(
        x.channel_shape, axes=self._axes
    )
    if self.only_advance_position_for_valid_timesteps:
      position = state + jnp.cumsum(x.mask.astype(jnp.int32), axis=1)
      state = position[:, -1:]
    else:
      position = state + jnp.arange(time, dtype=jnp.int32)
      state = state + time

    timing_signal = utils.get_timing_signal_1d_pos(
        position,
        np.prod(target_shape),
        min_timescale=self.min_timescale,
        max_timescale=self.max_timescale,
        dtype=self._param_dtype,
    )
    batch_size = x.shape[0]
    timing_signal = jnp.reshape(
        timing_signal, [batch_size, time] + list(target_shape)
    )
    if self.scale is not None:
      timing_signal *= self.scale[...]
    x = x.apply_values(lambda v: v + timing_signal.astype(v.dtype))
    return x, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    self._check_inputs(x.channel_spec)
    target_shape = utils.match_shape_along_axes(
        x.channel_shape, axes=self._axes
    )
    if self.only_advance_position_for_valid_timesteps:
      position = jnp.maximum(
          0, jnp.cumsum(x.mask.astype(jnp.int32), axis=1) - 1
      )
    else:
      position = jnp.arange(
          x.shape[1], dtype=jnp.int32
      )[jnp.newaxis, :]
    timing_signal = utils.get_timing_signal_1d_pos(
        position,
        np.prod(target_shape),
        min_timescale=self.min_timescale,
        max_timescale=self.max_timescale,
        dtype=self._param_dtype,
    )
    timing_signal = jnp.reshape(
        timing_signal, position.shape[:2] + target_shape
    )
    if self.scale is not None:
      timing_signal *= self.scale[...]
    x = x.apply_values(lambda v: v + timing_signal.astype(v.dtype))
    return x


class ApplyRotaryPositionalEncoding(
    types.PreservesType, types.PreservesShape, types.SequenceLayer
):
  """Applies Rotary Positional Encodings (RoPE) to the sequence.

  See https://arxiv.org/abs/2104.09864.
  """

  def __init__(
      self,
      *,
      max_wavelength: float,
      axis: int = -1,
      only_advance_position_for_valid_timesteps: bool = True,
      positions_in_at_least_fp32: bool = True,
  ):
    super().__init__()
    self.max_wavelength = max_wavelength
    self._axis = axis
    self.only_advance_position_for_valid_timesteps = (
        only_advance_position_for_valid_timesteps
    )
    self.positions_in_at_least_fp32 = positions_in_at_least_fp32

  def _check_inputs(self, input_spec: types.ShapeDType):
    if input_spec.dtype not in (
        jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64
    ):
      raise ValueError(
          f'{type(self).__name__} requires floating point argument.'
      )
    input_shape = (None, None) + input_spec.shape
    axis = (
        self._axis + len(input_shape)
        if self._axis < 0
        else self._axis
    )
    if axis <= 1:
      raise ValueError(
          f'{type(self).__name__} axis ({self._axis}) must refer to a'
          f' channels dimension ({input_spec=}).'
      )
    axis_size = input_shape[axis]
    if axis_size % 2 != 0:
      raise ValueError(
          f'{type(self).__name__} requires input_shape[{axis}]'
          f'={axis_size} to be even.'
      )

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    if self.only_advance_position_for_valid_timesteps:
      return {0: (-np.inf, 0)}
    else:
      return {0: (0, 0)}

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    self._check_inputs(input_spec)
    if self.only_advance_position_for_valid_timesteps:
      return jnp.full((batch_size, 1), -1, dtype=jnp.int32)
    else:
      return jnp.zeros((batch_size, 1), dtype=jnp.int32)

  def _apply_rope(
      self, x: jax.Array, positions: jax.Array
  ) -> jax.Array:
    axis = (
        self._axis + x.ndim if self._axis < 0 else self._axis
    )
    assert axis > 1
    channel_ndim = x.ndim - 2
    broadcast_shape = [1] * x.ndim
    axis_dim = x.shape[axis]
    broadcast_shape[axis] = axis_dim // 2
    assert axis_dim % 2 == 0

    freq_exponents = 2.0 * jnp.arange(axis_dim // 2) / axis_dim
    timescale = self.max_wavelength**freq_exponents

    @utils.maybe_in_at_least_fp32(
        self.positions_in_at_least_fp32, restore_dtypes=False,
    )
    def apply_position_embeddings(positions):
      radians = positions.reshape(
          positions.shape + (1,) * channel_ndim
      ) / timescale.reshape(broadcast_shape)
      sin, cos = jnp.sin(radians), jnp.cos(radians)
      x1, x2 = jnp.split(x, 2, axis=axis)
      return jnp.concatenate(
          [x1 * cos - x2 * sin, x2 * cos + x1 * sin],
          axis=axis,
          dtype=x.dtype,
      )

    return apply_position_embeddings(positions)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    self._check_inputs(x.channel_spec)
    x_time = x.shape[1]

    if self.only_advance_position_for_valid_timesteps:
      positions = state + jnp.cumsum(
          x.mask.astype(jnp.int32), axis=1
      )
      state = positions[:, -1:]
    else:
      positions = state + jnp.arange(x_time, dtype=jnp.int32)
      state = state + x_time

    y = x.apply_values(self._apply_rope, positions)
    return y, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    self._check_inputs(x.channel_spec)
    if self.only_advance_position_for_valid_timesteps:
      positions = jnp.maximum(
          0, jnp.cumsum(x.mask.astype(jnp.int32), axis=1) - 1
      )
    else:
      positions = jnp.arange(
          x.shape[1], dtype=jnp.int32
      )[jnp.newaxis, :]
    x = x.apply_values(self._apply_rope, positions)
    return x
