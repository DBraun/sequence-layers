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
"""Simple (generally stateless) NNX layers."""

import abc
import fractions
import functools
from typing import Callable, Mapping

import einops
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import utils
from sequence_layers.nnx import types


class Identity(types.PreservesType, types.StatelessPointwise):
  """Identity pass-through of the input."""

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return x


class Relu(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Relu layer."""

  @property
  def mask_required(self):
    return False

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.relu(values), mask


class Gelu(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Gaussian Error Linear Unit (GELU) layer."""

  def __init__(self, *, approximate: bool = True):
    super().__init__()
    self.approximate = approximate

  @property
  def mask_required(self):
    return False

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.gelu(values, approximate=self.approximate), mask


class Scale(types.PreservesType, types.Stateless):
  """Scales the input by a provided constant or array."""

  def __init__(self, *, scale):
    super().__init__()
    self.scale = np.asarray(scale)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    return jnp.broadcast_shapes(input_shape, self.scale.shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return x.apply_values_masked(
        lambda v: v * self.scale.astype(v.dtype)
    )


class MaskInvalid(types.PreservesType, types.StatelessPointwise):
  """Masks the input sequence."""

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return x.mask_invalid()


class Dropout(types.PreservesType, types.StatelessPointwise):
  """Computes dropout using NNX RNGs."""

  def __init__(
      self,
      *,
      rate: float = 0.0,
      broadcast_dims: tuple[int, ...] = (),
      rngs: nnx.Rngs | None = None,
  ):
    super().__init__()
    self.rate = rate
    self.broadcast_dims = broadcast_dims
    if rngs is not None:
      self.dropout_key = rngs.dropout()
    else:
      self.dropout_key = None

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
          'Step-wise training is not supported for Dropout yet.'
      )
    return self.layer(x, constants=constants), state

  def apply_dropout(self, x: jax.Array) -> jax.Array:
    """Applies dropout to array x."""
    if (self.rate == 0.0) or self.deterministic:
      return x

    if self.rate == 1.0:
      return jnp.zeros_like(x)

    keep_prob = 1.0 - self.rate
    self.dropout_key, rng = jax.random.split(self.dropout_key)
    broadcast_shape = list(x.shape)
    for dim in self.broadcast_dims:
      broadcast_shape[dim] = 1

    input_is_floating = jnp.issubdtype(x.dtype, jnp.inexact)
    use_select = not input_is_floating
    mask_dtype = x.dtype if input_is_floating else jnp.float32

    dropout_mask = jax.random.uniform(
        rng, shape=broadcast_shape, dtype=mask_dtype
    )

    if use_select:
      dropout_mask = dropout_mask < keep_prob
      if input_is_floating:
        x /= keep_prob
    else:
      dropout_mask = jnp.floor(keep_prob + dropout_mask) / keep_prob

    dropout_mask = jnp.broadcast_to(dropout_mask, x.shape)

    if use_select:
      x = jax.lax.select(dropout_mask, x, jnp.zeros_like(x))
    else:
      x *= dropout_mask
    return x

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return x.apply_values_masked(self.apply_dropout)


class Embedding(types.Stateless):
  """Computes embeddings of integer input codes."""

  def __init__(
      self,
      *,
      num_embeddings: int,
      dimension: int,
      param_dtype: types.DType = jnp.float32,
      compute_dtype: types.DType | None = None,
      embedding_init=nnx.initializers.lecun_normal(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.dimension = dimension
    self.param_dtype = param_dtype
    self.compute_dtype = compute_dtype
    self.embedding = nnx.Param(
        embedding_init(
            rngs.params(), (num_embeddings, dimension), param_dtype
        )
    )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return tuple(input_shape) + (self.dimension,)

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    if not jnp.issubdtype(input_dtype, jnp.integer):
      raise ValueError(
          'Input to Embedding must be an integer type, got:'
          f' {input_dtype}'
      )
    if self.compute_dtype is None:
      return self.param_dtype
    return self.compute_dtype

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if not jnp.issubdtype(x.dtype, jnp.integer):
      raise ValueError(
          'Input to Embedding must be an integer type, got:'
          f' {x.dtype}'
      )
    embedding = self.embedding[...]
    if self.compute_dtype is not None:
      embedding = embedding.astype(self.compute_dtype)
    return x.apply_values(lambda v: jnp.take(embedding, v, axis=0))


# ---------------------------------------------------------------------------
# Activation layers (StatelessPointwiseFunctor)
# ---------------------------------------------------------------------------


class Tanh(types.PreservesType, types.StatelessPointwiseFunctor):
  """A tanh layer."""

  @property
  def mask_required(self):
    return False

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.tanh(values), mask


class Sigmoid(types.PreservesType, types.StatelessPointwiseFunctor):
  """A sigmoid layer."""

  @property
  def mask_required(self):
    return False

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.sigmoid(values), mask


class Swish(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Swish layer."""

  @property
  def mask_required(self):
    return False

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.swish(values), mask


class Softplus(types.PreservesType, types.StatelessPointwiseFunctor):
  """A softplus layer."""

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.softplus(values), mask


class Softmax(types.PreservesType, types.StatelessPointwiseFunctor):
  """A softmax layer."""

  def __init__(self, *, axis: int = -1):
    super().__init__()
    self.axis = axis

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    axis = self.axis
    if (axis if axis >= 0 else values.ndim + axis) < 2:
      raise ValueError(
          'The softmax cannot be applied on the batch or time dimension'
          f' (got {axis=} for shape={values.shape})'
      )
    return jax.nn.softmax(values, axis=axis), mask


class LeakyRelu(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Leaky Relu layer."""

  def __init__(self, *, negative_slope: float = 0.01):
    super().__init__()
    self.negative_slope = negative_slope

  @property
  def mask_required(self):
    return False

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.leaky_relu(values, self.negative_slope), mask


class Elu(types.PreservesType, types.StatelessPointwiseFunctor):
  """An elu activation layer."""

  def __init__(self, *, alpha: float = 1.0):
    super().__init__()
    self.alpha = alpha

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.elu(values, self.alpha), mask


class Exp(types.PreservesType, types.StatelessPointwiseFunctor):
  """An exp layer."""

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jnp.exp(values), mask


class Log(types.PreservesType, types.StatelessPointwiseFunctor):
  """A log layer."""

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jnp.log(values), mask


class Abs(types.StatelessPointwiseFunctor):
  """Absolute value layer."""

  @property
  def mask_required(self):
    return False

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jnp.abs(values), mask

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
        return input_dtype


class Power(types.PreservesType, types.StatelessPointwiseFunctor):
  """Raises the input to the specified power."""

  def __init__(self, *, power: float = 1.0):
    super().__init__()
    self.power = power

  @property
  def mask_required(self):
    return False

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jnp.power(values, self.power), mask


class Cast(types.StatelessPointwiseFunctor):
  """Cast input values to the specified type."""

  def __init__(self, *, dtype: types.DType):
    super().__init__()
    self._cast_dtype = dtype

  @property
  def mask_required(self):
    return False

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return values.astype(self._cast_dtype), mask

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return self._cast_dtype


# ---------------------------------------------------------------------------
# Gated units
# ---------------------------------------------------------------------------


class GatedUnit(types.PreservesType, types.Stateless):
  """Computes a generalized Gated Unit, reducing input channels by 2x."""

  def __init__(
      self,
      *,
      feature_activation: Callable[[types.ValuesT], types.ValuesT]
      | None = None,
      gate_activation: Callable[[types.ValuesT], types.ValuesT]
      | None = None,
  ):
    super().__init__()
    self.feature_activation = feature_activation
    self.gate_activation = gate_activation

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    channels = input_shape[-1]
    if channels % 2 != 0:
      raise ValueError(
          f'Final dimension of input ({input_shape=}) must have an'
          ' even number of channels.'
      )
    return tuple(input_shape[:-1]) + (channels // 2,)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    feature, gate = jnp.split(x.values, 2, axis=-1)
    if self.feature_activation:
      feature = self.feature_activation(feature)
    if self.gate_activation:
      gate = self.gate_activation(gate)
    values = feature * gate
    return types.Sequence(values, x.mask)


class GatedLinearUnit(GatedUnit):
  """Computes a Gated Linear Unit, reducing input channels by 2x."""

  def __init__(self):
    super().__init__(gate_activation=jax.nn.sigmoid)


class GatedTanhUnit(GatedUnit):
  """Computes a Gated Tanh Unit, reducing input channels by 2x."""

  def __init__(self):
    super().__init__(
        feature_activation=jax.nn.tanh,
        gate_activation=jax.nn.sigmoid,
    )


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------


class Flatten(types.PreservesType, types.Stateless):
  """Flattens the channel dimensions to [batch, time, prod(channels)]."""

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return (int(np.prod(input_shape)),)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    batch_size, time = x.values.shape[:2]
    num_elements = int(np.prod(x.channel_shape))
    return x.apply_values_masked(
        jnp.reshape, [batch_size, time, num_elements]
    )


class Squeeze(types.PreservesType, types.Stateless):
  """Squeezes (removes) size-1 channel dimensions."""

  def __init__(self, *, axis: int | tuple[int, ...] | None = None):
    super().__init__()
    if isinstance(axis, int):
      axis = (axis,)
    self._axis = axis

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if self._axis is None:
      return tuple(d for d in input_shape if d != 1)
    result = list(input_shape)
    for a in sorted(self._axis, reverse=True):
      idx = a if a >= 0 else len(result) + a
      if result[idx] != 1:
        raise ValueError(
            f'Cannot squeeze axis {a} of shape {input_shape} '
            'because it is not 1.'
        )
      result.pop(idx)
    return tuple(result)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if self._axis is None:
      axes = tuple(
          i + 2 for i, d in enumerate(x.channel_shape) if d == 1
      )
    else:
      axes = tuple(a + 2 if a >= 0 else a for a in self._axis)
    return x.apply_values_masked(jnp.squeeze, axis=axes)


class ExpandDims(types.PreservesType, types.Stateless):
  """Expands channel dimensions with size-1 axes."""

  def __init__(self, *, axis: int | tuple[int, ...]):
    super().__init__()
    if isinstance(axis, int):
      axis = (axis,)
    self._axis = axis

  def _normalize_axes(self, input_shape: types.ShapeLike) -> list[int]:
    rank = len(input_shape)
    dims = [a + rank + 1 if a < 0 else a for a in self._axis]
    dims = sorted(set(dims))
    for dim in dims:
      if dim < 0 or dim > rank:
        raise ValueError(
            f'ExpandDims axes must refer to channels dimensions.'
            f' Got: {self._axis}.'
        )
    return dims

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    dims = self._normalize_axes(input_shape)
    output_shape = list(input_shape)
    for a in dims:
      output_shape.insert(a, 1)
    return tuple(output_shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    dims = [
        2 + d for d in self._normalize_axes(x.channel_shape)
    ]
    return x.apply_values_masked(jnp.expand_dims, dims)


class Reshape(types.PreservesType, types.Stateless):
  """Reshapes the channels dimension of the input."""

  def __init__(self, *, output_shape: tuple[int, ...]):
    super().__init__()
    self._output_shape = tuple(output_shape)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if np.prod(input_shape) != np.prod(self._output_shape):
      raise ValueError(
          f'Reshape output_shape={self._output_shape} must have the'
          f' same number of elements as {input_shape=}.'
      )
    return self._output_shape

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return x.apply_values_masked(
        lambda v: jnp.reshape(v, v.shape[:2] + self._output_shape)
    )


# ---------------------------------------------------------------------------
# Resampling layers
# ---------------------------------------------------------------------------


class Downsample1D(
    types.PreservesType, types.PreservesShape, types.Stateless
):
  """A 1D downsampling layer."""

  def __init__(self, *, rate: int):
    super().__init__()
    self.rate = rate

  @property
  def block_size(self) -> int:
    return self.rate

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self.rate)

  @property
  def input_latency(self) -> int:
    return self.rate - 1

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return type(x)(
        x.values[:, :: self.rate], x.mask[:, :: self.rate]
    )


class Upsample1D(
    types.PreservesType, types.PreservesShape, types.Stateless
):
  """A 1D upsampling layer."""

  def __init__(self, *, rate: int):
    super().__init__()
    self.rate = rate

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self.rate)

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return {s: (0, 0) for s in range(self.rate)}

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return type(x)(
        jnp.repeat(x.values, self.rate, axis=1),
        jnp.repeat(x.mask, self.rate, axis=1),
    )


# ---------------------------------------------------------------------------
# Pointwise math ops with broadcasting
# ---------------------------------------------------------------------------


class Add(types.PreservesType, types.Stateless):
  """Adds a constant or array to the input."""

  def __init__(self, *, shift):
    super().__init__()
    self._shift = np.asarray(shift)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return jnp.broadcast_shapes(input_shape, self._shift.shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return x.apply_values(
        lambda v: v + self._shift.astype(v.dtype)
    )


class Maximum(types.PreservesType, types.Stateless):
  """Clips the input to be at least the provided maximum value."""

  def __init__(self, *, maximum):
    super().__init__()
    self._maximum = np.asarray(maximum)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return jnp.broadcast_shapes(input_shape, self._maximum.shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return x.apply_values(
        lambda v: jnp.maximum(v, self._maximum.astype(v.dtype))
    )


class Minimum(types.PreservesType, types.Stateless):
  """Clips the input to be at most the provided minimum value."""

  def __init__(self, *, minimum):
    super().__init__()
    self._minimum = np.asarray(minimum)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return jnp.broadcast_shapes(input_shape, self._minimum.shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return x.apply_values(
        lambda v: jnp.minimum(v, self._minimum.astype(v.dtype))
    )


# ---------------------------------------------------------------------------
# Channel reductions
# ---------------------------------------------------------------------------


class _ReduceChannels(
    types.PreservesType, types.Stateless, metaclass=abc.ABCMeta
):
  """Abstract base class for reductions over channel dimensions."""

  def __init__(
      self,
      *,
      axis: int | tuple[int, ...] | None = -1,
      keepdims: bool = False,
  ):
    super().__init__()
    if isinstance(axis, int):
      axis = (axis,)
    self._axis = axis
    self.keepdims = keepdims

  @property
  @abc.abstractmethod
  def _reduce_fn(self) -> Callable:
    ...

  def _validate_axis(
      self, input_shape: types.ShapeLike
  ) -> tuple[int, ...]:
    rank = len(input_shape) + 2
    axis = self._axis
    if axis is not None:
      axis = tuple(a + rank if a < 0 else a for a in axis)
    else:
      axis = tuple(range(2, rank))
    for a in axis:
      if a < 2 or a >= rank:
        raise ValueError(
            f'Reduction axis {a} is invalid for input shape'
            f' {input_shape}. The batch and time dimensions cannot'
            ' be reduced over.'
        )
    return axis

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    axis = self._validate_axis(input_shape)
    if self.keepdims:
      return tuple(
          1 if i + 2 in axis else d
          for i, d in enumerate(input_shape)
      )
    else:
      return tuple(
          d for i, d in enumerate(input_shape) if i + 2 not in axis
      )

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    axis = self._validate_axis(x.channel_shape)
    return x.apply_values_masked(
        self._reduce_fn, axis=axis, keepdims=self.keepdims
    )


class Mean(_ReduceChannels):
  """Computes the mean over the specified axes."""

  @property
  def _reduce_fn(self):
    return jnp.mean


class Min(_ReduceChannels):
  """Computes the minimum over the specified axes."""

  @property
  def _reduce_fn(self):
    return jnp.min


class Max(_ReduceChannels):
  """Computes the maximum over the specified axes."""

  @property
  def _reduce_fn(self):
    return jnp.max


class Sum(_ReduceChannels):
  """Computes the sum over the specified axes."""

  @property
  def _reduce_fn(self):
    return jnp.sum


# ---------------------------------------------------------------------------
# Learnable activations
# ---------------------------------------------------------------------------


class PRelu(types.PreservesType, types.StatelessPointwiseFunctor):
  """Parametric Relu with a learnable negative slope."""

  def __init__(
      self,
      *,
      negative_slope_init: float = 0.01,
      param_dtype: types.DType = jnp.float32,
  ):
    super().__init__()
    self.negative_slope = nnx.Param(
        jnp.array(negative_slope_init, param_dtype)
    )

  @property
  def mask_required(self) -> bool:
    return False

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return (
        jnp.where(
            values >= 0,
            values,
            self.negative_slope[...].astype(values.dtype) * values,
        ),
        mask,
    )


class Affine(types.PreservesType, types.Stateless):
  """Learnable additive bias and multiplicative scale."""

  def __init__(
      self,
      *,
      use_bias: bool = True,
      use_scale: bool = True,
      shape: types.ShapeLike = (),
      param_dtype: types.DType = jnp.float32,
      scale_init=nnx.initializers.ones_init(),
      bias_init=nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.use_bias = use_bias
    self.use_scale = use_scale
    self._shape = tuple(shape) if shape else ()
    if use_scale:
      self.scale = nnx.Param(
          scale_init(rngs.params(), self._shape, param_dtype)
      )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), self._shape, param_dtype)
      )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) < len(self._shape):
      raise ValueError(
          f'The parameter has too many dimensions (input:'
          f' {len(input_shape)}, parameter: {len(self._shape)})'
      )
    return jnp.broadcast_shapes(input_shape, self._shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    _ = self.get_output_shape(x.channel_shape)
    if self.use_scale:
      x = x.apply_values(
          lambda v: v * self.scale[...].astype(v.dtype)
      )
    if self.use_bias:
      x = x.apply_values(
          lambda v: v + self.bias[...].astype(v.dtype)
      )
    return x


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------


class Transpose(types.PreservesType, types.Stateless):
  """Transposes (permutes) the channel dimensions of the input."""

  def __init__(self, *, axes: tuple[int, ...] | None = None):
    super().__init__()
    if axes is not None:
      if 0 in axes or 1 in axes:
        raise ValueError("Can't transpose batch or time dimension.")
      axes = tuple(axes)
    self._axes = axes

  def _validate_axes(
      self, input_shape: types.ShapeLike
  ) -> tuple[int, ...]:
    input_axes = tuple(range(2, 2 + len(input_shape)))
    if self._axes is None:
      return input_axes[::-1]
    sorted_axes = tuple(sorted(self._axes))
    if sorted_axes != input_axes:
      raise ValueError(
          f'The provided axes {sorted_axes} does not match those'
          f' of the input {input_axes}.'
      )
    return tuple(self._axes)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    axes = self._validate_axes(input_shape)
    return tuple(input_shape[a - 2] for a in axes)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    axes = self._validate_axes(x.channel_shape)
    return x.apply_values_masked(
        lambda v: jnp.transpose(v, (0, 1) + axes)
    )


class SwapAxes(types.PreservesType, types.Stateless):
  """Swaps two channel axes of the input."""

  def __init__(self, *, axis1: int, axis2: int):
    super().__init__()
    self.axis1 = axis1
    self.axis2 = axis2

  def _get_permutation(
      self, input_shape: types.ShapeLike
  ) -> tuple[int, ...]:
    rank = len(input_shape)
    a1 = self.axis1 + rank if self.axis1 < 0 else self.axis1
    a2 = self.axis2 + rank if self.axis2 < 0 else self.axis2
    if a1 < 0 or a1 >= rank or a2 < 0 or a2 >= rank:
      raise ValueError(
          f'SwapAxes axes ({self.axis1}, {self.axis2}) out of range'
          f' for input shape {input_shape}.'
      )
    axes = list(range(2, 2 + rank))
    axes[a1], axes[a2] = axes[a2], axes[a1]
    return tuple(axes)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    axes = self._get_permutation(input_shape)
    return tuple(input_shape[a - 2] for a in axes)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    axes = self._get_permutation(x.channel_shape)
    return x.apply_values_masked(
        lambda v: jnp.transpose(v, (0, 1) + axes)
    )


class MoveAxis(types.PreservesType, types.Stateless):
  """Moves one or more channel axes to new positions."""

  def __init__(
      self,
      *,
      source: int | tuple[int, ...],
      destination: int | tuple[int, ...],
  ):
    super().__init__()
    if isinstance(source, int):
      source = (source,)
    if isinstance(destination, int):
      destination = (destination,)
    if len(source) != len(destination):
      raise ValueError(
          f'source ({source}) and destination ({destination}) must'
          ' have the same number of elements.'
      )
    self._source = tuple(source)
    self._destination = tuple(destination)

  def _get_permutation(
      self, input_shape: types.ShapeLike
  ) -> tuple[int, ...]:
    rank = len(input_shape)
    src = tuple(s + rank if s < 0 else s for s in self._source)
    dst = tuple(d + rank if d < 0 else d for d in self._destination)
    # Compute the permutation using numpy.moveaxis on a dummy.
    order = [n for n in range(rank) if n not in src]
    for d, s in sorted(zip(dst, src)):
      order.insert(d, s)
    return tuple(a + 2 for a in order)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    axes = self._get_permutation(input_shape)
    return tuple(input_shape[a - 2] for a in axes)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    axes = self._get_permutation(x.channel_shape)
    return x.apply_values_masked(
        lambda v: jnp.transpose(v, (0, 1) + axes)
    )


class Slice(types.PreservesType, types.Stateless):
  """Slices the channel dimensions of input tensors."""

  def __init__(
      self,
      *,
      slices: tuple[tuple[int | None, int | None, int | None]
                    | int | None, ...],
  ):
    super().__init__()
    self._slices = tuple(slices)

  def _as_slices(self):
    return tuple(
        slice(*s) if isinstance(s, tuple) else s
        for s in self._slices
    )

  def _validate(self, input_shape: types.ShapeLike):
    non_none = sum(1 for s in self._slices if s is not None)
    if non_none != len(input_shape):
      raise ValueError(
          f'Slice has wrong size for input: {input_shape=}'
          f' slices={self._slices}'
      )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    self._validate(input_shape)
    output_dims = []
    input_index = 0
    for slice_i in self._slices:
      if isinstance(slice_i, tuple):
        s = slice(*slice_i)
        output_dims.append(
            len(range(*s.indices(input_shape[input_index])))
        )
        input_index += 1
      elif isinstance(slice_i, int):
        input_index += 1
      elif slice_i is None:
        output_dims.append(1)
      else:
        raise NotImplementedError(
            f'Unsupported slice type: {type(slice_i)}, {slice_i}'
        )
    return tuple(output_dims)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    full_slice = (
        slice(None), slice(None)
    ) + self._as_slices()
    self._validate(x.channel_shape)
    return x.apply_values_masked(lambda v: v[full_slice])


# ---------------------------------------------------------------------------
# Other layers
# ---------------------------------------------------------------------------


class OneHot(types.Stateless):
  """Computes one-hot vector of integer input codes."""

  def __init__(self, *, depth: int, compute_dtype: types.DType = jnp.float32):
    super().__init__()
    self.depth = depth
    self.compute_dtype = compute_dtype

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return tuple(input_shape) + (self.depth,)

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    if not jnp.issubdtype(input_dtype, jnp.integer):
      raise ValueError(
          'Input to OneHot must be an integer type, got:'
          f' {input_dtype}'
      )
    return self.compute_dtype

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if not jnp.issubdtype(x.dtype, jnp.integer):
      raise ValueError(
          'Input to OneHot must be an integer type, got:'
          f' {x.dtype}'
      )
    return x.apply_values(
        lambda v: jax.nn.one_hot(v, self.depth, dtype=self.compute_dtype)
    )


class Argmax(types.Stateless):
  """Computes argmax over the last channel dimension."""

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return tuple(input_shape[:-1])

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return jnp.int32

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return x.apply_values(jnp.argmax, axis=-1)


class Lambda(types.Stateless):
  """Wraps a stateless Python function as a SequenceLayer."""

  def __init__(
      self,
      *,
      fn: Callable,
      sequence_input: bool = False,
      mask_required: bool = True,
      expected_input_spec: types.ShapeDType | None = None,
  ):
    super().__init__()
    self._fn = fn
    self.sequence_input = sequence_input
    self._mask_required = mask_required
    self.expected_input_spec = expected_input_spec

  def get_output_spec(
      self,
      input_spec: types.ChannelSpec,
      *,
      constants: types.Constants | None = None,
  ) -> types.ChannelSpec:
    if self.sequence_input:
      input_spec_for_eval = types.Sequence(
          types.ShapeDType(
              (1, 1) + tuple(input_spec.shape), input_spec.dtype
          ),
          types.ShapeDType((1, 1), jnp.bool_),
      )
    else:
      input_spec_for_eval = types.ShapeDType(
          (1, 1) + tuple(input_spec.shape), input_spec.dtype
      )
    output_spec = jax.eval_shape(self._fn, input_spec_for_eval)
    return jax.ShapeDtypeStruct(
        output_spec.shape[2:], output_spec.dtype
    )

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    if self.expected_input_spec is None:
      raise ValueError(
          'get_output_dtype requires expected_input_spec.'
      )
    return self.get_output_spec(
        jax.ShapeDtypeStruct(
            tuple(self.expected_input_spec.shape), input_dtype
        )
    ).dtype

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if self.expected_input_spec is None:
      raise ValueError(
          'get_output_shape requires expected_input_spec.'
      )
    return self.get_output_spec(
        jax.ShapeDtypeStruct(
            tuple(input_shape),
            self.expected_input_spec.dtype,
        )
    ).shape

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if self.sequence_input:
      y = self._fn(x)
      if y.shape[:2] != x.shape[:2]:
        raise ValueError(
            f'Lambda function should not change the batch or time'
            f' shape. fn({x.shape=}) -> {y.shape}'
        )
    else:
      values = self._fn(x.values)
      if values.shape[:2] != x.shape[:2]:
        raise ValueError(
            f'Lambda function should not change the batch or time'
            f' shape. fn({x.shape=}) -> {values.shape=}'
        )
      if self._mask_required:
        y = types.Sequence(values, x.mask)
      else:
        y = type(x)(values, x.mask)
    return y


class Mod(types.PreservesType, types.Stateless):
  """Returns the remainder of division of the input by a divisor."""

  def __init__(self, *, divisor: float | np.ndarray):
    super().__init__()
    self._divisor = np.asarray(divisor)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return jnp.broadcast_shapes(input_shape, self._divisor.shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return x.apply_values_masked(
        lambda v: jnp.mod(v, self._divisor.astype(v.dtype))
    )


class Snake(types.PreservesType, types.StatelessPointwiseFunctor):
  """The "snake" activation: x + 1/beta * sin^2(x*alpha).

  Originally proposed in: https://arxiv.org/abs/2006.08195
  Extension from BigVGAN: https://arxiv.org/abs/2206.04658
  """

  def __init__(
      self,
      *,
      features: int | tuple[int, ...],
      separate_beta: bool = True,
      param_dtype: types.DType = jnp.float32,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self._separate_beta = separate_beta
    if isinstance(features, int):
      features = (features,)
    self.alpha_log = nnx.Param(
        nnx.initializers.zeros_init()(
            rngs.params(), features, param_dtype
        )
    )
    self._has_beta = separate_beta
    if separate_beta:
      self.beta_log = nnx.Param(
          nnx.initializers.zeros_init()(
              rngs.params(), features, param_dtype
          )
      )

  @property
  def mask_required(self) -> bool:
    return False

  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    alpha = jnp.exp(self.alpha_log[...])
    # Broadcast: add batch and time dims.
    expand = (jnp.newaxis,) * (values.ndim - alpha.ndim)
    alpha = alpha[expand]
    if self._has_beta:
      beta = jnp.exp(self.beta_log[...])
      beta = beta[expand]
    else:
      beta = alpha
    values = values + jnp.square(jnp.sin(values * alpha)) / (
        beta + 1e-12
    )
    return values, mask


class EmbeddingTranspose(types.Stateless):
  """Wraps an Embedding layer for weight-shared pre-softmax projection."""

  def __init__(
      self,
      *,
      embedding: 'Embedding',
      use_bias: bool = True,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType | None = None,
      bias_init=nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self._embedding = embedding
    self._use_bias = use_bias
    self._compute_dtype = compute_dtype
    self._param_dtype = param_dtype or jnp.float32

    if use_bias:
      self.bias = nnx.Param(
          bias_init(
              rngs.params(),
              (embedding.num_embeddings,),
              self._param_dtype,
          )
      )

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype,
        self._param_dtype,
        dtype=self._compute_dtype,
    )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if (
        not input_shape
        or input_shape[-1] != self._embedding.dimension
    ):
      raise ValueError(
          "Input query's final channel dimension must be equal"
          ' to the embedding dimension.'
      )
    return (*input_shape[:-1], self._embedding.num_embeddings)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self._compute_dtype
    )
    embedding_table = self._embedding.embedding[...].astype(
        compute_dtype
    )

    def attend_fn(v):
      v = v.astype(compute_dtype)
      y = jnp.einsum('...d,nd->...n', v, embedding_table)
      if self._use_bias:
        bias = self.bias[...].astype(compute_dtype)
        y = utils.bias_add(y, bias)
      return y

    if self._use_bias:
      return x.apply_values(attend_fn)
    else:
      return x.apply_values_masked(attend_fn)


class Emit(
    types.PreservesType,
    types.PreservesShape,
    types.StatelessEmitting,
):
  """An identity layer that emits its input."""

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Sequence]:
    return x, x


class NamedEmit(
    types.PreservesType,
    types.PreservesShape,
    types.StatelessEmitting,
):
  """An identity layer that emits its input with a named output."""

  def __init__(self, *, emit_name: str):
    super().__init__()
    self._emit_name = emit_name

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    return x, {self._emit_name: x}


class Upsample2D(types.PreservesType, types.Stateless):
  """A 2D upsampling layer."""

  def __init__(self, *, rate: int | tuple[int, int]):
    super().__init__()
    if isinstance(rate, int):
      rate = (rate, rate)
    self._rate = tuple(rate)

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self._rate[0])

  @property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    return {s: (0, 0) for s in range(self._rate[0])}

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 2:
      raise ValueError(
          'Upsample2D requires rank 4 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    return (input_shape[0] * self._rate[1], input_shape[1])

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    values = jnp.repeat(x.values, self._rate[0], axis=1)
    values = jnp.repeat(values, self._rate[1], axis=2)
    mask = jnp.repeat(x.mask, self._rate[0], axis=1)
    return type(x)(values, mask)


class EinopsRearrange(types.PreservesType, types.Stateless):
  """A wrapper for einops.rearrange on channel dimensions."""

  def __init__(
      self,
      *,
      pattern: str,
      axes_lengths: Mapping[str, int] | None = None,
  ):
    super().__init__()
    if '->' not in pattern:
      raise ValueError(
          f'The input pattern is not valid (got {pattern}).'
      )
    labels = set(
        pattern.replace('(', ' ').replace(')', ' ').split(' ')
    )
    if 'batch' in labels or 'time' in labels:
      raise ValueError(
          '`batch` and `time` are reserved axes labels'
          f' (got {pattern}).'
      )
    self._pattern = pattern
    self._axes_lengths = dict(axes_lengths) if axes_lengths else {}

  def _get_rearrange_fn(self) -> Callable[[jax.Array], jax.Array]:
    before, after = self._pattern.split('->')
    pattern = f'batch time {before} -> batch time {after}'
    return functools.partial(
        einops.rearrange, pattern=pattern, **self._axes_lengths
    )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    rearrange_fn = self._get_rearrange_fn()
    output = jax.eval_shape(
        rearrange_fn, jnp.zeros((1, 1) + tuple(input_shape))
    )
    return tuple(output.shape[2:])

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    rearrange_fn = self._get_rearrange_fn()
    return x.apply_values(rearrange_fn)
