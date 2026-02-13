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
"""NNX convolution layers."""

import abc
import fractions
import functools
from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import convolution as jax_conv
from sequence_layers.jax import utils
from sequence_layers.nnx import types


class BaseConv(types.SequenceLayer, metaclass=abc.ABCMeta):
  """Shared base logic for NNX convolution layers."""

  @property
  @abc.abstractmethod
  def _kernel_size(self) -> tuple[int, ...]:
    pass

  @property
  @abc.abstractmethod
  def _strides(self) -> tuple[int, ...]:
    pass

  @property
  @abc.abstractmethod
  def _dilation_rate(self) -> tuple[int, ...]:
    pass

  @property
  @abc.abstractmethod
  def _paddings(
      self,
  ) -> tuple[types.PaddingMode | tuple[int, int], ...]:
    pass

  @property
  def supports_step(self) -> bool:
    return self._paddings[0] in (
        types.PaddingMode.CAUSAL_VALID.value,
        types.PaddingMode.REVERSE_CAUSAL_VALID.value,
        types.PaddingMode.CAUSAL.value,
        types.PaddingMode.REVERSE_CAUSAL.value,
        types.PaddingMode.SEMICAUSAL.value,
    )

  @property
  def block_size(self) -> int:
    return self._strides[0]

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self._strides[0])

  @property
  def input_latency(self) -> int:
    effective_kernel_size = utils.convolution_effective_kernel_size(
        self._kernel_size[0], self._dilation_rate[0]
    )
    match self._paddings[0]:
      case (
          types.PaddingMode.CAUSAL_VALID.value
          | types.PaddingMode.CAUSAL.value
          | types.PaddingMode.SEMICAUSAL.value
      ):
        return 0
      case (
          types.PaddingMode.REVERSE_CAUSAL_VALID.value
          | types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.SEMICAUSAL_FULL.value
      ):
        return effective_kernel_size - 1
      case _:
        return 0

  @property
  def output_latency(self) -> int:
    match self._paddings[0]:
      case types.PaddingMode.SEMICAUSAL_FULL.value:
        return 0
      case _:
        return super().output_latency

  @functools.cached_property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return jax_conv.conv_receptive_field_per_step(
        self._kernel_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._paddings[0],
    )

  @property
  def _buffer_width(self) -> int:
    effective_kernel_size = utils.convolution_effective_kernel_size(
        self._kernel_size[0], self._dilation_rate[0]
    )
    match self._paddings[0]:
      case types.PaddingMode.SEMICAUSAL.value:
        return max(effective_kernel_size - self._strides[0], 0)
      case (
          types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ):
        return (
            (effective_kernel_size - 1)
            // self._strides[0]
            * self._strides[0]
        )
      case (
          types.PaddingMode.CAUSAL.value
          | types.PaddingMode.CAUSAL_VALID.value
      ):
        return effective_kernel_size - 1
      case _:
        raise NotImplementedError(
            f'Unsupported padding mode: {self._paddings[0]}'
        )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    if not (buffer_width := self._buffer_width):
      return ()
    if len(input_spec.shape) != len(self._kernel_size):
      raise ValueError(
          f'{type(self).__name__} requires a rank 2 input spec, got:'
          f' {input_spec}.'
      )
    return jax_conv.compute_conv_initial_state(
        batch_size,
        input_spec,
        buffer_width,
        self._paddings[0],
    )

  @abc.abstractmethod
  def _conv(
      self,
      x: jax.Array,
      padding: tuple[tuple[int, int], ...],
  ) -> jax.Array:
    """Runs the underlying convolution on raw values."""

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    explicit_paddings = ((0, 0),) + tuple(
        utils.convolution_explicit_padding(
            padding, kernel_size, stride, dilation_rate
        )
        for padding, kernel_size, stride, dilation_rate in zip(
            self._paddings[1:],
            self._kernel_size[1:],
            self._strides[1:],
            self._dilation_rate[1:],
            strict=True,
        )
    )

    effective_kernel_size = utils.convolution_effective_kernel_size(
        self._kernel_size[0], self._dilation_rate[0]
    )
    if effective_kernel_size > 1:
      x = x.mask_invalid()

    if buffer_width := self._buffer_width:
      state = state.concatenate(x)
    else:
      state = x

    values = self._conv(state.values, padding=explicit_paddings)
    mask = jax_conv.compute_conv_mask(
        state.mask,
        self._kernel_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._paddings[0],
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
    if self._kernel_size[0] > 1:
      x = x.mask_invalid()

    explicit_paddings = tuple(
        utils.convolution_explicit_padding(p, k, s, d)
        for p, k, s, d in zip(
            self._paddings,
            self._kernel_size,
            self._strides,
            self._dilation_rate,
            strict=True,
        )
    )
    values = self._conv(x.values, padding=explicit_paddings)
    mask = jax_conv.compute_conv_mask(
        x.mask,
        self._kernel_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._paddings[0],
        is_step=False,
    )
    result_type = (
        types.Sequence
        if self._use_bias or self._kernel_size[0] > 1
        else type(x)
    )
    return result_type(values, mask)

  @property
  @abc.abstractmethod
  def _use_bias(self) -> bool:
    pass


class Conv1D(BaseConv):
  """A 1D strided or dilated convolution layer."""

  def __init__(
      self,
      *,
      in_features: int,
      filters: int,
      kernel_size: int,
      strides: int = 1,
      dilation_rate: int = 1,
      padding: str = 'valid',
      groups: int = 1,
      use_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] | None = None,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      precision=None,
      kernel_init=nnx.initializers.lecun_normal(),
      bias_init=nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    from sequence_layers.jax.types import validate_padding
    self._padding = validate_padding(padding)
    self.in_features = in_features
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.dilation_rate = dilation_rate
    self.groups = groups
    self._use_bias_flag = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.precision = precision

    if in_features % groups != 0:
      raise ValueError(
          f'Input features ({in_features}) must be divisible by groups'
          f' ({groups}).'
      )

    kernel_shape = (kernel_size, in_features // groups, filters)
    self.kernel = nnx.Param(
        kernel_init(rngs.params(), kernel_shape, param_dtype)
    )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), (filters,), param_dtype)
      )

  @property
  def _use_bias(self) -> bool:
    return self._use_bias_flag

  @property
  def _kernel_size(self) -> tuple[int, ...]:
    return (self.kernel_size,)

  @property
  def _strides(self) -> tuple[int, ...]:
    return (self.strides,)

  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    return (self.dilation_rate,)

  @property
  def _paddings(self) -> tuple[str, ...]:
    return (self._padding,)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 1:
      raise ValueError(
          'Conv1D requires rank 3 input got: %s'
          % ([None, None] + list(input_shape))
      )
    return (self.filters,)

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self.compute_dtype
    )

  def _conv(
      self,
      x: jax.Array,
      padding: tuple[tuple[int, int], ...],
  ) -> jax.Array:
    assert len(padding) == 1, padding
    kernel = self.kernel[...]
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self.compute_dtype
    )
    x = x.astype(compute_dtype)
    kernel = kernel.astype(compute_dtype)

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(self.strides,),
        padding=padding,
        lhs_dilation=(1,),
        rhs_dilation=(self.dilation_rate,),
        dimension_numbers=('NHC', 'HIO', 'NHC'),
        feature_group_count=self.groups,
        precision=self.precision,
    )

    if self._use_bias_flag:
      bias = self.bias[...].astype(compute_dtype)
      y = utils.bias_add(y, bias)

    if self.activation:
      y = self.activation(y)

    return y


class DepthwiseConv1D(BaseConv):
  """A 1D depthwise strided or dilated convolution layer."""

  def __init__(
      self,
      *,
      in_features: int,
      kernel_size: int,
      strides: int = 1,
      depth_multiplier: int = 1,
      dilation_rate: int = 1,
      padding: str = 'valid',
      use_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] | None = None,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      precision=None,
      kernel_init=nnx.initializers.lecun_normal(),
      bias_init=nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    from sequence_layers.jax.types import validate_padding
    self._padding = validate_padding(padding)
    self.in_features = in_features
    self.kernel_size = kernel_size
    self.strides = strides
    self.depth_multiplier = depth_multiplier
    self.dilation_rate = dilation_rate
    self._use_bias_flag = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.precision = precision

    out_features = in_features * depth_multiplier
    kernel_shape = (kernel_size, 1, out_features)
    self.kernel = nnx.Param(
        kernel_init(rngs.params(), kernel_shape, param_dtype)
    )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), (out_features,), param_dtype)
      )

  @property
  def _use_bias(self) -> bool:
    return self._use_bias_flag

  @property
  def _kernel_size(self) -> tuple[int, ...]:
    return (self.kernel_size,)

  @property
  def _strides(self) -> tuple[int, ...]:
    return (self.strides,)

  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    return (self.dilation_rate,)

  @property
  def _paddings(self) -> tuple[str, ...]:
    return (self._padding,)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 1:
      raise ValueError(
          'DepthwiseConv1D requires rank 3 input got: %s'
          % ([None, None] + list(input_shape))
      )
    return (input_shape[0] * self.depth_multiplier,)

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self.compute_dtype
    )

  def _conv(
      self,
      x: jax.Array,
      padding: tuple[tuple[int, int], ...],
  ) -> jax.Array:
    assert len(padding) == 1, padding
    in_features = jnp.shape(x)[-1]
    kernel = self.kernel[...]
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self.compute_dtype
    )
    x = x.astype(compute_dtype)
    kernel = kernel.astype(compute_dtype)

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(self.strides,),
        padding=padding,
        lhs_dilation=(1,),
        rhs_dilation=(self.dilation_rate,),
        dimension_numbers=('NHC', 'HIO', 'NHC'),
        feature_group_count=in_features,
        precision=self.precision,
    )

    if self._use_bias_flag:
      bias = self.bias[...].astype(compute_dtype)
      y = utils.bias_add(y, bias)

    if self.activation:
      y = self.activation(y)

    return y


class Conv2D(BaseConv):
  """A 2D strided or dilated convolution layer."""

  def __init__(
      self,
      *,
      in_features: int,
      filters: int,
      kernel_size: int | tuple[int, int],
      strides: int | tuple[int, int] = 1,
      dilation_rate: int | tuple[int, int] = 1,
      time_padding: str = 'valid',
      spatial_padding: str | tuple[int, int] = 'same',
      groups: int = 1,
      use_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] | None = None,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      precision=None,
      kernel_init=nnx.initializers.lecun_normal(),
      bias_init=nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    from sequence_layers.jax.types import validate_padding
    from sequence_layers.jax.types import validate_explicit_padding

    self._time_padding = validate_padding(time_padding)
    if isinstance(spatial_padding, str):
      self._spatial_padding = validate_padding(spatial_padding)
    else:
      self._spatial_padding = validate_explicit_padding(spatial_padding)

    self._kernel_size_val = utils.normalize_2tuple(kernel_size)
    self._strides_val = utils.normalize_2tuple(strides)
    self._dilation_rate_val = utils.normalize_2tuple(dilation_rate)

    self.in_features = in_features
    self.filters = filters
    self.groups = groups
    self._use_bias_flag = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.precision = precision

    if in_features % groups != 0:
      raise ValueError(
          f'Input features ({in_features}) must be divisible by groups'
          f' ({groups}).'
      )

    kernel_shape = self._kernel_size_val + (
        in_features // groups,
        filters,
    )
    self.kernel = nnx.Param(
        kernel_init(rngs.params(), kernel_shape, param_dtype)
    )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), (filters,), param_dtype)
      )

  @property
  def _use_bias(self) -> bool:
    return self._use_bias_flag

  @property
  def _kernel_size(self) -> tuple[int, ...]:
    return self._kernel_size_val

  @property
  def _strides(self) -> tuple[int, ...]:
    return self._strides_val

  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    return self._dilation_rate_val

  @property
  def _paddings(self) -> tuple[str | tuple[int, int], ...]:
    return (self._time_padding, self._spatial_padding)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 2:
      raise ValueError(
          'Conv2D requires rank 4 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    spatial_output_size = utils.convolution_padding_output_size(
        input_shape[0],
        self._spatial_padding,
        kernel_size=self._kernel_size_val[1],
        stride=self._strides_val[1],
        dilation_rate=self._dilation_rate_val[1],
    )
    return (spatial_output_size, self.filters)

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self.compute_dtype
    )

  def _conv(
      self,
      x: jax.Array,
      padding: tuple[tuple[int, int], ...],
  ) -> jax.Array:
    assert len(padding) == 2, padding
    kernel = self.kernel[...]
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self.compute_dtype
    )
    x = x.astype(compute_dtype)
    kernel = kernel.astype(compute_dtype)

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=self._strides_val,
        padding=padding,
        lhs_dilation=(1, 1),
        rhs_dilation=self._dilation_rate_val,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=self.groups,
        precision=self.precision,
    )

    if self._use_bias_flag:
      bias = self.bias[...].astype(compute_dtype)
      y = utils.bias_add(y, bias)

    if self.activation:
      y = self.activation(y)

    return y


class Conv3D(BaseConv):
  """A 3D strided or dilated convolution layer."""

  def __init__(
      self,
      *,
      in_features: int,
      filters: int,
      kernel_size: int | tuple[int, int, int],
      strides: int | tuple[int, int, int] = 1,
      dilation_rate: int | tuple[int, int, int] = 1,
      time_padding: str = 'valid',
      spatial_padding: tuple[
          str | tuple[int, int],
          str | tuple[int, int],
      ] = ('same', 'same'),
      groups: int = 1,
      use_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] | None = None,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      precision=None,
      kernel_init=nnx.initializers.lecun_normal(),
      bias_init=nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    from sequence_layers.jax.types import validate_padding

    self._time_padding = validate_padding(time_padding)

    if len(spatial_padding) != 2:
      raise ValueError(
          'Conv3D expects 2 spatial padding modes got:'
          f' {spatial_padding}'
      )
    self._spatial_padding = tuple(
        validate_padding(s) if isinstance(s, str) else s
        for s in spatial_padding
    )

    self._kernel_size_val = utils.normalize_3tuple(kernel_size)
    self._strides_val = utils.normalize_3tuple(strides)
    self._dilation_rate_val = utils.normalize_3tuple(dilation_rate)

    self.in_features = in_features
    self.filters = filters
    self.groups = groups
    self._use_bias_flag = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.precision = precision

    if in_features % groups != 0:
      raise ValueError(
          f'Input features ({in_features}) must be divisible by groups'
          f' ({groups}).'
      )

    kernel_shape = self._kernel_size_val + (
        in_features // groups,
        filters,
    )
    self.kernel = nnx.Param(
        kernel_init(rngs.params(), kernel_shape, param_dtype)
    )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), (filters,), param_dtype)
      )

  @property
  def _use_bias(self) -> bool:
    return self._use_bias_flag

  @property
  def _kernel_size(self) -> tuple[int, ...]:
    return self._kernel_size_val

  @property
  def _strides(self) -> tuple[int, ...]:
    return self._strides_val

  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    return self._dilation_rate_val

  @property
  def _paddings(self) -> tuple[str | tuple[int, int], ...]:
    return (self._time_padding,) + self._spatial_padding

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 3:
      raise ValueError(
          'Conv3D requires rank 5 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    spatial_output_sizes = [
        utils.convolution_padding_output_size(
            input_shape[i],
            self._spatial_padding[i],
            kernel_size=self._kernel_size_val[1 + i],
            stride=self._strides_val[1 + i],
            dilation_rate=self._dilation_rate_val[1 + i],
        )
        for i in range(2)
    ]
    return tuple(spatial_output_sizes) + (self.filters,)

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self.compute_dtype
    )

  def _conv(
      self,
      x: jax.Array,
      padding: tuple[tuple[int, int], ...],
  ) -> jax.Array:
    assert len(padding) == 3, padding
    kernel = self.kernel[...]
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self.compute_dtype
    )
    x = x.astype(compute_dtype)
    kernel = kernel.astype(compute_dtype)

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=self._strides_val,
        padding=padding,
        lhs_dilation=(1, 1, 1),
        rhs_dilation=self._dilation_rate_val,
        dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'),
        feature_group_count=self.groups,
        precision=self.precision,
    )

    if self._use_bias_flag:
      bias = self.bias[...].astype(compute_dtype)
      y = utils.bias_add(y, bias)

    if self.activation:
      y = self.activation(y)

    return y


class Conv1DTranspose(types.SequenceLayer):
  """A 1D transpose convolution layer."""

  def __init__(
      self,
      *,
      in_features: int,
      filters: int,
      kernel_size: int,
      strides: int = 1,
      dilation_rate: int = 1,
      padding: str = 'valid',
      groups: int = 1,
      use_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] | None = None,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      precision=None,
      kernel_init=nnx.initializers.lecun_normal(),
      bias_init=nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    from sequence_layers.jax.types import validate_padding
    self._padding = validate_padding(padding)

    if self._padding in (
        types.PaddingMode.REVERSE_CAUSAL.value,
        types.PaddingMode.REVERSE_CAUSAL_VALID.value,
        types.PaddingMode.CAUSAL_VALID.value,
    ):
      raise ValueError(f'Unsupported padding mode: {self._padding}')

    self.in_features = in_features
    self.filters = filters
    self._kernel_size_val = kernel_size
    self._strides_val = strides
    self._dilation_rate_val = dilation_rate
    self.groups = groups
    self._use_bias_flag = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.precision = precision

    if in_features % groups != 0:
      raise ValueError(
          f'Input features ({in_features}) must be divisible by groups'
          f' ({groups}).'
      )

    kernel_shape = (kernel_size, in_features // groups, filters)
    self.kernel = nnx.Param(
        kernel_init(rngs.params(), kernel_shape, param_dtype)
    )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), (filters,), param_dtype)
      )

  @property
  def supports_step(self) -> bool:
    return self._padding == types.PaddingMode.CAUSAL.value

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self._strides_val, 1)

  @functools.cached_property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return jax_conv.transpose_conv_receptive_field_per_step(
        self._kernel_size_val,
        self._strides_val,
        self._dilation_rate_val,
        self._padding,
    )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return (self.filters,)

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self.compute_dtype
    )

  @property
  def _buffer_width(self) -> int:
    effective_kernel_size = utils.convolution_effective_kernel_size(
        self._kernel_size_val, self._dilation_rate_val
    )
    return max(0, effective_kernel_size - self._strides_val)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    if buffer_width := self._buffer_width:
      output_spec = self.get_output_spec(input_spec, constants=constants)
      return jnp.zeros(
          (batch_size, buffer_width) + output_spec.shape,
          dtype=output_spec.dtype,
      )
    else:
      return ()

  def _transpose_conv(
      self,
      x: jax.Array,
      explicit_padding: tuple[int, int],
  ) -> tuple[jax.Array, jax.Array | None]:
    kernel = self.kernel[...]
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self.compute_dtype
    )
    x = x.astype(compute_dtype)
    kernel = kernel.astype(compute_dtype)

    bias = None
    if self._use_bias_flag:
      bias = self.bias[...].astype(compute_dtype)

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1,),
        padding=(explicit_padding,),
        lhs_dilation=(self._strides_val,),
        rhs_dilation=(self._dilation_rate_val,),
        dimension_numbers=('NHC', 'HIO', 'NHC'),
        feature_group_count=self.groups,
        batch_group_count=1,
        precision=self.precision,
    )

    return y, bias

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if self._kernel_size_val > 1:
      x = x.mask_invalid()

    explicit_padding = jax_conv.transpose_conv_explicit_padding(
        self._kernel_size_val,
        self._strides_val,
        self._dilation_rate_val,
        self._padding,
    )

    values, bias = self._transpose_conv(x.values, explicit_padding)
    mask = jax_conv.compute_conv_transpose_mask(
        x.mask,
        self._kernel_size_val,
        self._strides_val,
        self._dilation_rate_val,
        self._padding,
    )

    if bias is not None:
      values = utils.bias_add(values, bias)

    if self.activation:
      values = self.activation(values)

    return types.Sequence(values, mask)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if self._kernel_size_val > 1:
      x = x.mask_invalid()

    explicit_padding = jax_conv.transpose_conv_explicit_padding(
        self._kernel_size_val,
        self._strides_val,
        self._dilation_rate_val,
        types.PaddingMode.VALID.value,
    )

    values, bias = self._transpose_conv(x.values, explicit_padding)
    mask = jax_conv.compute_conv_transpose_mask(
        x.mask,
        self._kernel_size_val,
        self._strides_val,
        self._dilation_rate_val,
        self._padding,
    )

    if self._buffer_width:
      time = x.shape[1]

      state = jnp.pad(
          state,
          [[0, 0], [0, values.shape[1] - self._buffer_width], [0, 0]],
      )

      values = values + state

      output_samples = self._strides_val * time

      values, state = jnp.split(values, [output_samples], axis=1)

    if bias is not None:
      values = utils.bias_add(values, bias)

    if self.activation:
      values = self.activation(values)

    return types.Sequence(values, mask), state


class Conv2DTranspose(types.SequenceLayer):
  """A 2D transpose convolution layer."""

  def __init__(
      self,
      *,
      in_features: int,
      filters: int,
      kernel_size: int | tuple[int, int],
      strides: int | tuple[int, int] = 1,
      dilation_rate: int | tuple[int, int] = 1,
      time_padding: str = 'valid',
      spatial_padding: str | tuple[int, int] = 'same',
      groups: int = 1,
      use_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] | None = None,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      precision=None,
      kernel_init=nnx.initializers.lecun_normal(),
      bias_init=nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    from sequence_layers.jax.types import validate_padding
    from sequence_layers.jax.types import validate_explicit_padding

    self._time_padding = validate_padding(time_padding)

    if self._time_padding in (
        types.PaddingMode.REVERSE_CAUSAL.value,
        types.PaddingMode.REVERSE_CAUSAL_VALID.value,
        types.PaddingMode.CAUSAL_VALID.value,
    ):
      raise ValueError(
          f'Unsupported padding mode: {self._time_padding}'
      )

    if isinstance(spatial_padding, str):
      self._spatial_padding = validate_padding(spatial_padding)
      if self._spatial_padding in (
          types.PaddingMode.REVERSE_CAUSAL.value,
          types.PaddingMode.REVERSE_CAUSAL_VALID.value,
          types.PaddingMode.CAUSAL_VALID.value,
      ):
        raise ValueError(
            f'Unsupported padding mode: {self._spatial_padding}'
        )
    else:
      self._spatial_padding = validate_explicit_padding(spatial_padding)

    self._kernel_size_val = utils.normalize_2tuple(kernel_size)
    self._strides_val = utils.normalize_2tuple(strides)
    self._dilation_rate_val = utils.normalize_2tuple(dilation_rate)

    self.in_features = in_features
    self.filters = filters
    self.groups = groups
    self._use_bias_flag = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.precision = precision

    if in_features % groups != 0:
      raise ValueError(
          f'Input features ({in_features}) must be divisible by groups'
          f' ({groups}).'
      )

    kernel_shape = self._kernel_size_val + (
        in_features // groups,
        filters,
    )
    self.kernel = nnx.Param(
        kernel_init(rngs.params(), kernel_shape, param_dtype)
    )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), (filters,), param_dtype)
      )

  @property
  def _kernel_size(self) -> tuple[int, ...]:
    return self._kernel_size_val

  @property
  def _strides(self) -> tuple[int, ...]:
    return self._strides_val

  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    return self._dilation_rate_val

  @property
  def _paddings(self) -> tuple[str | tuple[int, int], ...]:
    return (self._time_padding, self._spatial_padding)

  @property
  def supports_step(self) -> bool:
    return self._time_padding == types.PaddingMode.CAUSAL.value

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self._strides_val[0], 1)

  @functools.cached_property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return jax_conv.transpose_conv_receptive_field_per_step(
        self._kernel_size_val[0],
        self._strides_val[0],
        self._dilation_rate_val[0],
        self._time_padding,
    )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 2:
      raise ValueError(
          'Conv2DTranspose requires rank 4 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    spatial_output_size = jax_conv._compute_conv_transpose_output_length(
        input_shape[0],
        self._kernel_size_val[1],
        self._strides_val[1],
        dilation_rate=self._dilation_rate_val[1],
        padding=self._spatial_padding,
    )
    return (spatial_output_size, self.filters)

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self.compute_dtype
    )

  @property
  def _buffer_width(self) -> int:
    effective_kernel_size = utils.convolution_effective_kernel_size(
        self._kernel_size_val[0], self._dilation_rate_val[0]
    )
    return max(0, effective_kernel_size - self._strides_val[0])

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    output_spec = self.get_output_spec(input_spec, constants=constants)
    if buffer_width := self._buffer_width:
      return jnp.zeros(
          (batch_size, buffer_width) + output_spec.shape,
          dtype=output_spec.dtype,
      )
    else:
      return ()

  def _transpose_conv(
      self,
      x: jax.Array,
      explicit_padding: tuple[tuple[int, int], tuple[int, int]],
  ) -> tuple[jax.Array, jax.Array | None]:
    kernel = self.kernel[...]
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self.compute_dtype
    )
    x = x.astype(compute_dtype)
    kernel = kernel.astype(compute_dtype)

    bias = None
    if self._use_bias_flag:
      bias = self.bias[...].astype(compute_dtype)

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1),
        padding=explicit_padding,
        lhs_dilation=self._strides_val,
        rhs_dilation=self._dilation_rate_val,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=self.groups,
        batch_group_count=1,
        precision=self.precision,
    )

    return y, bias

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if self._kernel_size_val[0] > 1:
      x = x.mask_invalid()

    explicit_time_padding = jax_conv.transpose_conv_explicit_padding(
        self._kernel_size_val[0],
        self._strides_val[0],
        self._dilation_rate_val[0],
        self._time_padding,
    )

    explicit_spatial_padding = jax_conv.transpose_conv_explicit_padding(
        self._kernel_size_val[1],
        self._strides_val[1],
        self._dilation_rate_val[1],
        self._spatial_padding,
    )

    values, bias = self._transpose_conv(
        x.values, (explicit_time_padding, explicit_spatial_padding)
    )
    mask = jax_conv.compute_conv_transpose_mask(
        x.mask,
        self._kernel_size_val[0],
        self._strides_val[0],
        self._dilation_rate_val[0],
        self._time_padding,
    )

    if bias is not None:
      values = utils.bias_add(values, bias)

    if self.activation:
      values = self.activation(values)

    return types.Sequence(values, mask)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if self._kernel_size_val[0] > 1:
      x = x.mask_invalid()

    explicit_time_padding = jax_conv.transpose_conv_explicit_padding(
        self._kernel_size_val[0],
        self._strides_val[0],
        self._dilation_rate_val[0],
        types.PaddingMode.VALID.value,
    )

    explicit_spatial_padding = jax_conv.transpose_conv_explicit_padding(
        self._kernel_size_val[1],
        self._strides_val[1],
        self._dilation_rate_val[1],
        self._spatial_padding,
    )

    values, bias = self._transpose_conv(
        x.values, (explicit_time_padding, explicit_spatial_padding)
    )
    mask = jax_conv.compute_conv_transpose_mask(
        x.mask,
        self._kernel_size_val[0],
        self._strides_val[0],
        self._dilation_rate_val[0],
        self._time_padding,
    )

    if self._buffer_width:
      time = x.shape[1]

      state = jnp.pad(
          state,
          [
              [0, 0],
              [0, values.shape[1] - self._buffer_width],
              [0, 0],
              [0, 0],
          ],
      )

      values = values + state

      output_samples = self._strides_val[0] * time

      values, state = jnp.split(values, [output_samples], axis=1)

    if bias is not None:
      values = utils.bias_add(values, bias)

    if self.activation:
      values = self.activation(values)

    return types.Sequence(values, mask), state
