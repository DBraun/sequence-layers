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
"""NNX pooling layers."""

import abc
import fractions
import functools
from typing import Callable, Sequence as TypingSequence

import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import convolution as jax_conv
from sequence_layers.jax import utils
from sequence_layers.nnx import types


def _max_pool_init_value(dtype: types.DType) -> complex:
  if issubclass(dtype.type, jnp.floating):
    return -jnp.inf
  elif issubclass(dtype.type, jnp.integer):
    return jnp.iinfo(dtype).min
  elif dtype == jnp.bool_:
    return False
  else:
    raise ValueError(f'Unsupported dtype: {dtype}')


def _min_pool_init_value(dtype: types.DType) -> complex:
  if issubclass(dtype.type, jnp.floating):
    return jnp.inf
  elif issubclass(dtype.type, jnp.integer):
    return jnp.iinfo(dtype).max
  elif dtype == jnp.bool_:
    return True
  else:
    raise ValueError(f'Unsupported dtype: {dtype}')


def div_no_nan_grad(x, y):
  is_zero = y == 0
  return (
      jnp.where(is_zero, jnp.zeros_like(x), x)
      / jnp.where(is_zero, 1.0, y)
  )


def _reduce_window(
    x: jax.Array,
    init_value: complex,
    computation,
    window_dimensions: tuple[int, ...],
    window_strides: tuple[int, ...],
    window_dilation: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
) -> jax.Array:
  def pack(value, default):
    return (default,) + tuple(value) + (default,) * (x.ndim - len(value) - 1)

  return jax.lax.reduce_window(
      x,
      init_value=np.asarray(init_value, x.dtype),
      computation=computation,
      window_dimensions=pack(window_dimensions, 1),
      window_strides=pack(window_strides, 1),
      padding=pack(padding, (0, 0)),
      base_dilation=(1,) * x.ndim,
      window_dilation=pack(window_dilation, 1),
  )


class BasePooling1D(types.PreservesType, types.SequenceLayer,
                    metaclass=abc.ABCMeta):
  """Shared base logic for 1D pooling layers."""

  def __init__(
      self,
      *,
      pool_size: int,
      strides: int = 1,
      dilation_rate: int = 1,
      padding: str = 'valid',
  ):
    super().__init__()
    from sequence_layers.jax.types import validate_padding
    self._padding = validate_padding(padding)
    self.pool_size = pool_size
    self.strides = strides
    self.dilation_rate = dilation_rate

  @property
  @abc.abstractmethod
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    pass

  @abc.abstractmethod
  def _pad_value(self, input_dtype: types.DType) -> complex:
    pass

  def _layer(
      self,
      x: jax.Array,
      mask: jax.Array,
      padding: tuple[tuple[int, int], ...],
  ) -> jax.Array:
    del mask
    return _reduce_window(
        x,
        init_value=self._pad_value(x.dtype),
        computation=self._computation,
        window_dimensions=(self.pool_size,),
        window_strides=(self.strides,),
        window_dilation=(self.dilation_rate,),
        padding=padding,
    )

  @property
  def supports_step(self) -> bool:
    return self._padding in (
        types.PaddingMode.REVERSE_CAUSAL_VALID.value,
        types.PaddingMode.CAUSAL.value,
        types.PaddingMode.REVERSE_CAUSAL.value,
        types.PaddingMode.SEMICAUSAL.value,
    )

  @property
  def block_size(self) -> int:
    return self.strides

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self.strides)

  @property
  def input_latency(self) -> int:
    effective_pool_size = utils.convolution_effective_kernel_size(
        self.pool_size, self.dilation_rate
    )
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
        return effective_pool_size - 1
      case _:
        return 0

  @functools.cached_property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return jax_conv.conv_receptive_field_per_step(
        self.pool_size,
        self.strides,
        self.dilation_rate,
        self._padding,
    )

  @property
  def _buffer_width(self) -> int:
    effective_pool_size = utils.convolution_effective_kernel_size(
        self.pool_size, self.dilation_rate
    )
    match self._padding:
      case types.PaddingMode.SEMICAUSAL.value:
        return max(effective_pool_size - self.strides, 0)
      case (
          types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ):
        return (
            (effective_pool_size - 1)
            // self.strides
            * self.strides
        )
      case _:
        return effective_pool_size - 1

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    if not self._buffer_width:
      return ()
    return jax_conv.compute_conv_initial_state(
        batch_size,
        input_spec,
        self._buffer_width,
        self._padding,
        self._pad_value(input_spec.dtype),
    ).unmask()

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return tuple(input_shape)

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return input_dtype

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    x = x.mask_invalid(self._pad_value(x.dtype))

    if buffer_width := self._buffer_width:
      state = state.concatenate(x)
    else:
      state = x

    values = self._layer(
        state.values, state.mask, padding=((0, 0),)
    )
    mask = jax_conv.compute_conv_mask(
        mask=state.mask,
        kernel_size=self.pool_size,
        stride=self.strides,
        dilation_rate=self.dilation_rate,
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
    explicit_paddings = (
        utils.convolution_explicit_padding(
            self._padding,
            self.pool_size,
            self.strides,
            self.dilation_rate,
        ),
    )

    x = x.mask_invalid(self._pad_value(x.dtype))
    values = self._layer(x.values, x.mask, explicit_paddings)
    mask = jax_conv.compute_conv_mask(
        x.mask,
        self.pool_size,
        self.strides,
        self.dilation_rate,
        self._padding,
        is_step=False,
    )
    return types.Sequence(values, mask)


class MaxPooling1D(BasePooling1D):
  """A 1D max pooling layer."""

  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _max_pool_init_value(input_dtype)

  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.max


class MinPooling1D(BasePooling1D):
  """A 1D min pooling layer."""

  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _min_pool_init_value(input_dtype)

  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.min


class AveragePooling1D(BasePooling1D):
  """A 1D average pooling layer."""

  def __init__(
      self,
      *,
      pool_size: int,
      strides: int = 1,
      dilation_rate: int = 1,
      padding: str = 'valid',
      masked_average: bool = False,
  ):
    super().__init__(
        pool_size=pool_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
    )
    self.masked_average = masked_average

  def _pad_value(self, input_dtype: types.DType) -> complex:
    return 0

  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.add

  def _layer(
      self,
      x: jax.Array,
      mask: jax.Array,
      padding: tuple[tuple[int, int], ...],
  ) -> jax.Array:
    pad_value = self._pad_value(x.dtype)
    y_sum = _reduce_window(
        x,
        init_value=pad_value,
        computation=jax.lax.add,
        window_dimensions=(self.pool_size,),
        window_dilation=(self.dilation_rate,),
        window_strides=(self.strides,),
        padding=padding,
    )
    if self.masked_average:
      mask_sum = _reduce_window(
          mask.astype(x.dtype),
          init_value=pad_value,
          computation=jax.lax.add,
          window_dimensions=(self.pool_size,),
          window_dilation=(self.dilation_rate,),
          window_strides=(self.strides,),
          padding=padding,
      )
      mask_sum = jnp.expand_dims(mask_sum, range(2, y_sum.ndim))
      if issubclass(x.dtype.type, jnp.integer):
        y = jnp.floor_divide(y_sum, mask_sum)
      else:
        y = div_no_nan_grad(y_sum, mask_sum)
    elif issubclass(x.dtype.type, jnp.integer):
      y = y_sum // self.pool_size
    else:
      y = y_sum / self.pool_size
    return y


class BasePooling2D(types.PreservesType, types.SequenceLayer,
                    metaclass=abc.ABCMeta):
  """Shared base logic for 2D pooling layers."""

  def __init__(
      self,
      *,
      pool_size: int | tuple[int, int],
      strides: int | tuple[int, int] = 1,
      dilation_rate: int | tuple[int, int] = 1,
      time_padding: str = 'valid',
      spatial_padding: str | tuple[int, int] = 'same',
  ):
    super().__init__()
    from sequence_layers.jax.types import validate_padding
    from sequence_layers.jax.types import validate_explicit_padding

    self._time_padding = validate_padding(time_padding)
    if isinstance(spatial_padding, str):
      self._spatial_padding = validate_padding(spatial_padding)
    else:
      self._spatial_padding = validate_explicit_padding(spatial_padding)

    self._pool_size = utils.normalize_2tuple(pool_size)
    self._strides = utils.normalize_2tuple(strides)
    self._dilation_rate = utils.normalize_2tuple(dilation_rate)

  @property
  @abc.abstractmethod
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    pass

  @abc.abstractmethod
  def _pad_value(self, input_dtype: types.DType) -> complex:
    pass

  def _layer(
      self,
      x: jax.Array,
      mask: jax.Array,
      padding: tuple[tuple[int, int], ...],
  ) -> jax.Array:
    del mask
    return _reduce_window(
        x,
        init_value=self._pad_value(x.dtype),
        computation=self._computation,
        window_dimensions=self._pool_size,
        window_strides=self._strides,
        window_dilation=self._dilation_rate,
        padding=padding,
    )

  @property
  def _paddings(self) -> tuple:
    return (self._time_padding, self._spatial_padding)

  @property
  def supports_step(self) -> bool:
    return self._time_padding in (
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
    effective_pool_size = utils.convolution_effective_kernel_size(
        self._pool_size[0], self._dilation_rate[0]
    )
    match self._time_padding:
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
        return effective_pool_size - 1
      case _:
        return 0

  @functools.cached_property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return jax_conv.conv_receptive_field_per_step(
        self._pool_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._time_padding,
    )

  @property
  def _buffer_width(self) -> int:
    effective_pool_size = utils.convolution_effective_kernel_size(
        self._pool_size[0], self._dilation_rate[0]
    )
    match self._time_padding:
      case types.PaddingMode.SEMICAUSAL.value:
        return max(effective_pool_size - self._strides[0], 0)
      case (
          types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ):
        return (
            (effective_pool_size - 1)
            // self._strides[0]
            * self._strides[0]
        )
      case _:
        return effective_pool_size - 1

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    if not self._buffer_width:
      return ()
    return jax_conv.compute_conv_initial_state(
        batch_size,
        input_spec,
        self._buffer_width,
        self._time_padding,
        self._pad_value(input_spec.dtype),
    ).unmask()

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) < 2:
      raise ValueError(
          '2D pooling requires rank >= 4 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    spatial_output_size = utils.convolution_padding_output_size(
        input_shape[0],
        self._spatial_padding,
        kernel_size=self._pool_size[1],
        stride=self._strides[1],
        dilation_rate=self._dilation_rate[1],
    )
    return (spatial_output_size,) + tuple(input_shape[1:])

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return input_dtype

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    x = x.mask_invalid(self._pad_value(x.dtype))

    if buffer_width := self._buffer_width:
      state = state.concatenate(x)
    else:
      state = x

    spatial_padding = utils.convolution_explicit_padding(
        self._spatial_padding,
        self._pool_size[1],
        self._strides[1],
        self._dilation_rate[1],
    )
    values = self._layer(
        state.values, state.mask, padding=((0, 0), spatial_padding)
    )
    mask = jax_conv.compute_conv_mask(
        mask=state.mask,
        kernel_size=self._pool_size[0],
        stride=self._strides[0],
        dilation_rate=self._dilation_rate[0],
        padding=self._time_padding,
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
    time_padding = utils.convolution_explicit_padding(
        self._time_padding,
        self._pool_size[0],
        self._strides[0],
        self._dilation_rate[0],
    )
    spatial_padding = utils.convolution_explicit_padding(
        self._spatial_padding,
        self._pool_size[1],
        self._strides[1],
        self._dilation_rate[1],
    )

    x = x.mask_invalid(self._pad_value(x.dtype))
    values = self._layer(
        x.values, x.mask, (time_padding, spatial_padding)
    )
    mask = jax_conv.compute_conv_mask(
        x.mask,
        self._pool_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._time_padding,
        is_step=False,
    )
    return types.Sequence(values, mask)


class MaxPooling2D(BasePooling2D):
  """A 2D max pooling layer."""

  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _max_pool_init_value(input_dtype)

  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.max


class MinPooling2D(BasePooling2D):
  """A 2D min pooling layer."""

  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _min_pool_init_value(input_dtype)

  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.min


class AveragePooling2D(BasePooling2D):
  """A 2D average pooling layer."""

  def __init__(
      self,
      *,
      pool_size: int | tuple[int, int],
      strides: int | tuple[int, int] = 1,
      dilation_rate: int | tuple[int, int] = 1,
      time_padding: str = 'valid',
      spatial_padding: str | tuple[int, int] = 'same',
      masked_average: bool = False,
  ):
    super().__init__(
        pool_size=pool_size,
        strides=strides,
        dilation_rate=dilation_rate,
        time_padding=time_padding,
        spatial_padding=spatial_padding,
    )
    self.masked_average = masked_average

  def _pad_value(self, input_dtype: types.DType) -> complex:
    return 0

  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.add

  def _layer(
      self,
      x: jax.Array,
      mask: jax.Array,
      padding: tuple[tuple[int, int], ...],
  ) -> jax.Array:
    pad_value = self._pad_value(x.dtype)
    y_sum = _reduce_window(
        x,
        init_value=pad_value,
        computation=jax.lax.add,
        window_dimensions=self._pool_size,
        window_dilation=self._dilation_rate,
        window_strides=self._strides,
        padding=padding,
    )
    pool_volume = int(np.prod(self._pool_size))
    if self.masked_average:
      mask_sum = _reduce_window(
          mask.astype(x.dtype),
          init_value=pad_value,
          computation=jax.lax.add,
          window_dimensions=(self._pool_size[0],),
          window_dilation=(self._dilation_rate[0],),
          window_strides=(self._strides[0],),
          padding=padding[:1],
      )
      spatial_volume = int(np.prod(self._pool_size[1:]))
      mask_sum = jnp.expand_dims(mask_sum, range(2, y_sum.ndim))
      mask_sum = mask_sum * spatial_volume
      if issubclass(x.dtype.type, jnp.integer):
        y = jnp.floor_divide(y_sum, mask_sum)
      else:
        y = div_no_nan_grad(y_sum, mask_sum)
    elif issubclass(x.dtype.type, jnp.integer):
      y = y_sum // pool_volume
    else:
      y = y_sum / pool_volume
    return y


class BasePooling3D(types.PreservesType, types.SequenceLayer,
                    metaclass=abc.ABCMeta):
  """Shared base logic for 3D pooling layers."""

  def __init__(
      self,
      *,
      pool_size: int | tuple[int, int, int],
      strides: int | tuple[int, int, int] = 1,
      dilation_rate: int | tuple[int, int, int] = 1,
      time_padding: str = 'valid',
      spatial_padding: tuple[
          str | tuple[int, int],
          str | tuple[int, int],
      ] = ('same', 'same'),
  ):
    super().__init__()
    from sequence_layers.jax.types import validate_padding

    self._time_padding = validate_padding(time_padding)

    if len(spatial_padding) != 2:
      raise ValueError(
          '3D pooling expects 2 spatial padding modes got:'
          f' {spatial_padding}'
      )
    self._spatial_padding = tuple(
        validate_padding(s) if isinstance(s, str) else s
        for s in spatial_padding
    )

    self._pool_size = utils.normalize_3tuple(pool_size)
    self._strides = utils.normalize_3tuple(strides)
    self._dilation_rate = utils.normalize_3tuple(dilation_rate)

  @property
  @abc.abstractmethod
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    pass

  @abc.abstractmethod
  def _pad_value(self, input_dtype: types.DType) -> complex:
    pass

  def _layer(
      self,
      x: jax.Array,
      mask: jax.Array,
      padding: tuple[tuple[int, int], ...],
  ) -> jax.Array:
    del mask
    return _reduce_window(
        x,
        init_value=self._pad_value(x.dtype),
        computation=self._computation,
        window_dimensions=self._pool_size,
        window_strides=self._strides,
        window_dilation=self._dilation_rate,
        padding=padding,
    )

  @property
  def _paddings(self) -> tuple:
    return (self._time_padding,) + self._spatial_padding

  @property
  def supports_step(self) -> bool:
    return self._time_padding in (
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
    effective_pool_size = utils.convolution_effective_kernel_size(
        self._pool_size[0], self._dilation_rate[0]
    )
    match self._time_padding:
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
        return effective_pool_size - 1
      case _:
        return 0

  @functools.cached_property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return jax_conv.conv_receptive_field_per_step(
        self._pool_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._time_padding,
    )

  @property
  def _buffer_width(self) -> int:
    effective_pool_size = utils.convolution_effective_kernel_size(
        self._pool_size[0], self._dilation_rate[0]
    )
    match self._time_padding:
      case types.PaddingMode.SEMICAUSAL.value:
        return max(effective_pool_size - self._strides[0], 0)
      case (
          types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ):
        return (
            (effective_pool_size - 1)
            // self._strides[0]
            * self._strides[0]
        )
      case _:
        return effective_pool_size - 1

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    if not self._buffer_width:
      return ()
    return jax_conv.compute_conv_initial_state(
        batch_size,
        input_spec,
        self._buffer_width,
        self._time_padding,
        self._pad_value(input_spec.dtype),
    ).unmask()

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) < 3:
      raise ValueError(
          '3D pooling requires rank >= 5 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    spatial_output_sizes = [
        utils.convolution_padding_output_size(
            input_shape[i],
            self._spatial_padding[i],
            kernel_size=self._pool_size[1 + i],
            stride=self._strides[1 + i],
            dilation_rate=self._dilation_rate[1 + i],
        )
        for i in range(2)
    ]
    return tuple(spatial_output_sizes) + tuple(input_shape[2:])

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return input_dtype

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    x = x.mask_invalid(self._pad_value(x.dtype))

    if buffer_width := self._buffer_width:
      state = state.concatenate(x)
    else:
      state = x

    spatial_paddings = tuple(
        utils.convolution_explicit_padding(
            self._spatial_padding[i],
            self._pool_size[1 + i],
            self._strides[1 + i],
            self._dilation_rate[1 + i],
        )
        for i in range(2)
    )
    values = self._layer(
        state.values, state.mask,
        padding=((0, 0),) + spatial_paddings,
    )
    mask = jax_conv.compute_conv_mask(
        mask=state.mask,
        kernel_size=self._pool_size[0],
        stride=self._strides[0],
        dilation_rate=self._dilation_rate[0],
        padding=self._time_padding,
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
    time_padding = utils.convolution_explicit_padding(
        self._time_padding,
        self._pool_size[0],
        self._strides[0],
        self._dilation_rate[0],
    )
    spatial_paddings = tuple(
        utils.convolution_explicit_padding(
            self._spatial_padding[i],
            self._pool_size[1 + i],
            self._strides[1 + i],
            self._dilation_rate[1 + i],
        )
        for i in range(2)
    )

    x = x.mask_invalid(self._pad_value(x.dtype))
    values = self._layer(
        x.values, x.mask, (time_padding,) + spatial_paddings
    )
    mask = jax_conv.compute_conv_mask(
        x.mask,
        self._pool_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._time_padding,
        is_step=False,
    )
    return types.Sequence(values, mask)


class MaxPooling3D(BasePooling3D):
  """A 3D max pooling layer."""

  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _max_pool_init_value(input_dtype)

  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.max


class MinPooling3D(BasePooling3D):
  """A 3D min pooling layer."""

  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _min_pool_init_value(input_dtype)

  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.min


class AveragePooling3D(BasePooling3D):
  """A 3D average pooling layer."""

  def __init__(
      self,
      *,
      pool_size: int | tuple[int, int, int],
      strides: int | tuple[int, int, int] = 1,
      dilation_rate: int | tuple[int, int, int] = 1,
      time_padding: str = 'valid',
      spatial_padding: tuple[
          str | tuple[int, int],
          str | tuple[int, int],
      ] = ('same', 'same'),
      masked_average: bool = False,
  ):
    super().__init__(
        pool_size=pool_size,
        strides=strides,
        dilation_rate=dilation_rate,
        time_padding=time_padding,
        spatial_padding=spatial_padding,
    )
    self.masked_average = masked_average

  def _pad_value(self, input_dtype: types.DType) -> complex:
    return 0

  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.add

  def _layer(
      self,
      x: jax.Array,
      mask: jax.Array,
      padding: tuple[tuple[int, int], ...],
  ) -> jax.Array:
    pad_value = self._pad_value(x.dtype)
    y_sum = _reduce_window(
        x,
        init_value=pad_value,
        computation=jax.lax.add,
        window_dimensions=self._pool_size,
        window_dilation=self._dilation_rate,
        window_strides=self._strides,
        padding=padding,
    )
    pool_volume = int(np.prod(self._pool_size))
    if self.masked_average:
      mask_sum = _reduce_window(
          mask.astype(x.dtype),
          init_value=pad_value,
          computation=jax.lax.add,
          window_dimensions=(self._pool_size[0],),
          window_dilation=(self._dilation_rate[0],),
          window_strides=(self._strides[0],),
          padding=padding[:1],
      )
      spatial_volume = int(np.prod(self._pool_size[1:]))
      mask_sum = jnp.expand_dims(mask_sum, range(2, y_sum.ndim))
      mask_sum = mask_sum * spatial_volume
      if issubclass(x.dtype.type, jnp.integer):
        y = jnp.floor_divide(y_sum, mask_sum)
      else:
        y = div_no_nan_grad(y_sum, mask_sum)
    elif issubclass(x.dtype.type, jnp.integer):
      y = y_sum // pool_volume
    else:
      y = y_sum / pool_volume
    return y
