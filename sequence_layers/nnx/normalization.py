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
"""NNX normalization layers."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import utils
from sequence_layers.nnx import types


def _validate_and_normalize_axes(
    axes: int | list[int] | tuple[int, ...],
    input_shape: types.Shape,
) -> tuple[int, ...]:
  """Normalizes axes and checks batch/time are not specified."""
  if isinstance(axes, int):
    axes = (axes,)
  else:
    normalized_axes = set()
    for axis in axes:
      if axis < 0:
        axis += len(input_shape)
      if axis < 0 or axis > len(input_shape) - 1:
        raise ValueError(
            f'Axis out of range {axis=} for {axes=} with {input_shape=}.'
        )
      normalized_axes.add(axis)
    axes = tuple(sorted(normalized_axes))
  for axis in axes:
    if axis in (0, 1):
      raise ValueError(
          'Normalizing over the batch or time dimension is '
          f'not allowed. Got: {axes}'
      )
  return axes


class LayerNormalization(types.PreservesType, types.StatelessPointwise):
  """Applies layer normalization to input sequences."""

  def __init__(
      self,
      *,
      features: int | tuple[int, ...],
      axis: int | tuple[int, ...] = -1,
      epsilon: float = 1e-6,
      use_bias: bool = True,
      use_scale: bool = True,
      reductions_in_at_least_fp32: bool = True,
      param_dtype: types.DType = jnp.float32,
      bias_init=nnx.initializers.zeros_init(),
      scale_init=nnx.initializers.ones_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.epsilon = epsilon
    self.axis = axis
    self._use_bias = use_bias
    self._use_scale = use_scale
    self.reductions_in_at_least_fp32 = reductions_in_at_least_fp32
    self._param_dtype = param_dtype

    if isinstance(features, int):
      features = (features,)
    self._features = features
    reduced_feature_shape = list(features)

    if use_scale:
      self.scale = nnx.Param(
          scale_init(rngs.params(), reduced_feature_shape, param_dtype)
      )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), reduced_feature_shape, param_dtype)
      )

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    reduction_axes = _validate_and_normalize_axes(self.axis, x.shape)

    feature_shape = [1] * x.ndim
    for ax in reduction_axes:
      feature_shape[ax] = x.shape[ax]

    @utils.maybe_in_at_least_fp32(self.reductions_in_at_least_fp32)
    def reductions(values: jax.Array) -> jax.Array:
      mean = jnp.mean(values, axis=reduction_axes, keepdims=True)
      variance = jnp.mean(
          jnp.square(values - mean), axis=reduction_axes, keepdims=True
      )
      normed = (values - mean) * jax.lax.rsqrt(
          variance + self.epsilon
      )
      return normed

    y = reductions(x.values)

    if self._use_scale:
      scale = self.scale[...].astype(y.dtype)
      y *= jnp.reshape(scale, feature_shape)

    if self._use_bias:
      bias = self.bias[...].astype(y.dtype)
      y += jnp.reshape(bias, feature_shape)

    return types.Sequence(y, x.mask)


class RMSNormalization(types.PreservesType, types.StatelessPointwise):
  """A simplified version of LayerNormalization (no mean or offset)."""

  def __init__(
      self,
      *,
      features: int | tuple[int, ...],
      axis: int | tuple[int, ...] = -1,
      epsilon: float = 1e-6,
      use_scale: bool = True,
      reductions_in_at_least_fp32: bool = True,
      param_dtype: types.DType = jnp.float32,
      scale_init=nnx.initializers.ones_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.epsilon = epsilon
    self.axis = axis
    self._use_scale = use_scale
    self.reductions_in_at_least_fp32 = reductions_in_at_least_fp32
    self._param_dtype = param_dtype

    if isinstance(features, int):
      features = (features,)
    self._features = features
    reduced_feature_shape = list(features)

    if use_scale:
      self.scale = nnx.Param(
          scale_init(rngs.params(), reduced_feature_shape, param_dtype)
      )

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    reduction_axes = _validate_and_normalize_axes(
        self.axis, x.values.shape
    )

    feature_shape = [1] * x.values.ndim
    for ax in reduction_axes:
      feature_shape[ax] = x.values.shape[ax]

    @utils.maybe_in_at_least_fp32(self.reductions_in_at_least_fp32)
    def reductions(values: jax.Array) -> jax.Array:
      mean_square = jnp.mean(
          jnp.square(values), axis=reduction_axes, keepdims=True
      )
      normed = values * jax.lax.rsqrt(mean_square + self.epsilon)
      return normed

    y = reductions(x.values)

    if self._use_scale:
      scale = self.scale[...].astype(y.dtype)
      y *= jnp.reshape(scale, feature_shape)

    return types.Sequence(y, x.mask)


class L2Normalize(types.PreservesType, types.StatelessPointwise):
  """L2 normalization over the specified channel axes."""

  def __init__(
      self,
      *,
      axis: int | tuple[int, ...] = -1,
      epsilon: float = 1e-12,
  ):
    super().__init__()
    self.axis = axis
    self.epsilon = epsilon

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    reduction_axes = _validate_and_normalize_axes(self.axis, x.shape)

    @utils.maybe_in_at_least_fp32(True)
    def reductions(values: jax.Array) -> jax.Array:
      squared_sum = jnp.sum(
          jnp.square(values), axis=reduction_axes, keepdims=True
      )
      return values * jax.lax.rsqrt(squared_sum + self.epsilon)

    return types.Sequence(reductions(x.values), x.mask)


class BatchNormalization(types.PreservesType, types.StatelessPointwise):
  """Applies batch normalization to input sequences.

  Step-wise training is not supported, since batch normalization is not causal
  during training (it relies on statistics of future timesteps). When not
  training, it uses running statistics learned during training.
  """

  def __init__(
      self,
      *,
      num_features: int,
      axis: int = -1,
      momentum: float = 0.99,
      epsilon: float = 0.001,
      use_bias: bool = True,
      use_scale: bool = True,
      use_fast_variance: bool = True,
      param_dtype: types.DType = jnp.float32,
      bias_init=nnx.initializers.zeros_init(),
      scale_init=nnx.initializers.ones_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self._axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self._use_bias = use_bias
    self._use_scale = use_scale
    self.use_fast_variance = use_fast_variance
    self._param_dtype = param_dtype

    if use_scale:
      self.scale = nnx.Param(
          scale_init(rngs.params(), (num_features,), param_dtype)
      )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), (num_features,), param_dtype)
      )

    self.mean = nnx.BatchStat(jnp.zeros((num_features,)))
    self.var = nnx.BatchStat(jnp.ones((num_features,)))

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if not self.deterministic:
      raise ValueError(
          'Step-wise training is not supported for BatchNormalization.'
      )
    return self.layer(x, constants=constants), state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    (axis,) = _validate_and_normalize_axes([self._axis], x.shape)

    feature_shape = [1] * x.ndim
    feature_shape[axis] = x.shape[axis]

    use_running_average = self.deterministic
    values = x.values

    if use_running_average:
      mean = self.mean[...]
      var = self.var[...]
    else:
      # Compute masked batch statistics.
      mask = x.expanded_mask()
      reduction_axes = [i for i in range(values.ndim) if i != axis]
      masked_values = values * mask
      count = jnp.sum(mask, axis=reduction_axes)
      count = jnp.maximum(count, 1.0)
      mean = jnp.sum(masked_values, axis=reduction_axes) / count
      if self.use_fast_variance:
        var = (
            jnp.sum(masked_values * masked_values, axis=reduction_axes)
            / count
            - mean * mean
        )
        var = jnp.maximum(var, 0.0)
      else:
        diff = (values - jnp.reshape(mean, feature_shape)) * mask
        var = jnp.sum(diff * diff, axis=reduction_axes) / count

      # Update running statistics.
      self.mean[...] = (
          self.momentum * self.mean[...] + (1 - self.momentum) * mean
      )
      self.var[...] = (
          self.momentum * self.var[...] + (1 - self.momentum) * var
      )

    # Normalize.
    mean = jnp.reshape(mean, feature_shape)
    var = jnp.reshape(var, feature_shape)
    y = (values - mean) * jax.lax.rsqrt(var + self.epsilon)

    if self._use_scale:
      scale = self.scale[...].astype(y.dtype)
      y *= jnp.reshape(scale, feature_shape)
    if self._use_bias:
      bias = self.bias[...].astype(y.dtype)
      y += jnp.reshape(bias, feature_shape)

    return types.Sequence(y, x.mask)


class GroupNormalization(types.PreservesType, types.StatelessPointwise):
  """Applies group normalization to input sequences.

  https://arxiv.org/abs/1803.08494
  """

  def __init__(
      self,
      *,
      num_groups: int,
      features: int,
      axis: int = -1,
      epsilon: float = 1e-6,
      use_bias: bool = True,
      use_scale: bool = True,
      param_dtype: types.DType = jnp.float32,
      bias_init=nnx.initializers.zeros_init(),
      scale_init=nnx.initializers.ones_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    if num_groups <= 0:
      raise ValueError(f'{num_groups=} must be positive.')
    self.num_groups = num_groups
    self._features = features
    self._axis = axis
    self.epsilon = epsilon
    self._use_bias = use_bias
    self._use_scale = use_scale

    reduced_feature_shape = [features]
    if use_scale:
      self.scale = nnx.Param(
          scale_init(rngs.params(), reduced_feature_shape, param_dtype)
      )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), reduced_feature_shape, param_dtype)
      )

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    (axis,) = _validate_and_normalize_axes([self._axis], x.shape)

    axis_size = x.shape[axis]
    if axis_size % self.num_groups != 0:
      raise ValueError(
          f'Input axis={axis} size {axis_size} must be divisible'
          f' by num_groups={self.num_groups}.'
      )

    feature_shape = [1] * x.ndim
    feature_shape[axis] = axis_size

    @utils.maybe_in_at_least_fp32(True)
    def reductions(values: jax.Array) -> jax.Array:
      # Reshape to expose groups.
      group_size = axis_size // self.num_groups
      outer = list(values.shape[:axis])
      inner = list(values.shape[axis + 1:])
      grouped = jnp.reshape(
          values, outer + [self.num_groups, group_size] + inner
      )

      # Reduce over all dims except batch, time, and num_groups.
      expanded_rank = grouped.ndim
      reduction_axes = [
          a for a in range(expanded_rank) if a not in (0, 1, axis)
      ]

      mean = jnp.mean(grouped, axis=reduction_axes, keepdims=True)
      variance = jnp.mean(
          jnp.square(grouped - mean),
          axis=reduction_axes,
          keepdims=True,
      )
      normed = (grouped - mean) * jax.lax.rsqrt(
          variance + self.epsilon
      )
      return jnp.reshape(normed, values.shape)

    y = reductions(x.values)

    if self._use_scale:
      scale = self.scale[...].astype(y.dtype)
      y *= jnp.reshape(scale, feature_shape)

    if self._use_bias:
      bias = self.bias[...].astype(y.dtype)
      y += jnp.reshape(bias, feature_shape)

    return types.Sequence(y, x.mask)
