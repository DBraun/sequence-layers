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
"""Dense NNX layers."""

import typing
from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp
from sequence_layers.jax import utils
from sequence_layers.nnx import types


class Dense(types.Stateless):
  """A basic dense layer."""

  def __init__(
      self,
      *,
      in_features: int,
      features: int,
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
    self.in_features = in_features
    self.features = features
    self.use_bias = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.precision = precision

    self.kernel = nnx.Param(
        kernel_init(rngs.params(), (in_features, features), param_dtype)
    )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), (features,), param_dtype)
      )

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self.compute_dtype
    )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    if not input_shape:
      raise ValueError(
          f'Dense requires at least rank 3 input. Got: {input_shape=}'
      )
    return tuple(input_shape[:-1]) + (self.features,)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if x.ndim < 3:
      raise ValueError(
          f'Dense requires at least rank 3 input. Got: {x.shape=}'
      )

    kernel = self.kernel[...]
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self.compute_dtype
    )
    kernel = kernel.astype(compute_dtype)

    def dense_fn(v):
      v = v.astype(compute_dtype)
      y = jnp.einsum('...a,ab->...b', v, kernel, precision=self.precision)
      if self.use_bias:
        bias = self.bias[...].astype(compute_dtype)
        y = utils.bias_add(y, bias)
      if self.activation is not None:
        y = self.activation(y)
      return y

    # Preserve masked state if no bias or activation are in use.
    if self.use_bias or self.activation is not None:
      return x.apply_values(dense_fn)
    else:
      return x.apply_values_masked(dense_fn)


class DenseShaped(types.Stateless):
  """A dense layer that reshapes channels to an arbitrary output shape."""

  def __init__(
      self,
      *,
      in_shape: types.ShapeLike,
      output_shape: types.ShapeLike,
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
    self._in_shape = tuple(in_shape)
    self._output_shape = tuple(output_shape)
    self.use_bias = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.precision = precision

    # Build einsum equation.
    input_dims = ''.join(
        chr(ord('a') + i) for i in range(len(self._in_shape))
    )
    output_dims = ''.join(
        chr(ord('a') + i + len(self._in_shape))
        for i in range(len(self._output_shape))
    )

    in_kernel_shape = self._in_shape if self._in_shape else (1,)
    in_weight_dims = input_dims if input_dims else 'I'
    out_kernel_shape = self._output_shape if self._output_shape else (1,)
    out_weight_dims = output_dims if output_dims else 'O'

    self._equation = (
        f'BT{input_dims},{in_weight_dims}{out_weight_dims}'
        f'->BT{output_dims}'
    )

    kernel_shape = in_kernel_shape + out_kernel_shape
    self.kernel = nnx.Param(
        kernel_init(rngs.params(), kernel_shape, param_dtype)
    )
    if use_bias:
      bias_shape = out_kernel_shape
      self.bias = nnx.Param(
          bias_init(rngs.params(), bias_shape, param_dtype)
      )

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self.compute_dtype
    )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return self._output_shape

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    kernel = self.kernel[...]
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self.compute_dtype
    )
    kernel = kernel.astype(compute_dtype)

    def dense_fn(v):
      v = v.astype(compute_dtype)
      y = jnp.einsum(
          self._equation, v, kernel, precision=self.precision
      )
      if self.use_bias:
        bias = self.bias[...].astype(compute_dtype)
        y = utils.bias_add(y, bias)
      if self.activation is not None:
        y = self.activation(y)
      return y

    if self.use_bias or self.activation is not None:
      return x.apply_values(dense_fn)
    else:
      return x.apply_values_masked(dense_fn)


def _parse_einsum_equation(equation: str) -> tuple[str, str, str]:
  """Parse and validate an EinsumDense equation."""
  if '->' not in equation:
    raise ValueError(
        f'equation is not valid for EinsumDense: {equation}'
    )
  left, output_spec = equation.split('->')
  input_spec, kernel_spec = left.split(',')
  if not input_spec.startswith('...') or not output_spec.startswith('...'):
    raise ValueError('Equation must be of the form "...X,Y->...Z".')
  if 3 + len(set(input_spec[3:])) != len(input_spec):
    raise ValueError(
        f'Equation {input_spec=} must not contain duplicate variables.'
    )
  if 3 + len(set(output_spec[3:])) != len(output_spec):
    raise ValueError(
        f'Equation {output_spec=} must not contain duplicate variables.'
    )
  return input_spec, kernel_spec, output_spec


def _resolve_output_shape(
    equation: str,
    input_shape: types.ShapeLike,
    output_shape: tuple[int | None, ...],
) -> types.Shape:
  """Resolve None entries in output_shape using input_shape."""
  input_spec, _, output_spec = _parse_einsum_equation(equation)
  input_spec_trimmed = input_spec[3:]
  output_spec_trimmed = output_spec[3:]

  if len(input_spec_trimmed) != len(input_shape):
    raise ValueError(
        f'Equation {input_spec_trimmed=} does not match'
        f' {input_shape=} rank.'
    )

  input_dims = {
      d: input_shape[i] for i, d in enumerate(input_spec_trimmed)
  }

  resolved = list(output_shape)
  if len(output_spec_trimmed) != len(resolved):
    raise ValueError(
        f'Equation {output_spec_trimmed=} does not match'
        f' {output_shape=}.'
    )
  for i, d in enumerate(output_spec_trimmed):
    if resolved[i] is None:
      resolved[i] = input_dims[d]
    elif d in input_dims and resolved[i] != input_dims[d]:
      raise ValueError(
          'Input shape and output shape inconsistent for dimension'
          f' {d=}. {output_shape=} {input_shape=}'
      )
  return typing.cast(types.Shape, tuple(resolved))


class EinsumDense(types.Stateless):
  """A dense layer that transforms channel shape with an einsum equation.

  Equation input and output specs must have leading ellipses to
  broadcast over the batch and time dimensions.

  Example::

    Input sequence: [b, t, c1, c2, c3]
    equation = '...abc,bd->...bd'
    output_shape = [None, c4]
    bias_axes = 'd'
    Output sequence: [b, t, c2, c4]

    Kernel shape: [c2, c4]
    Bias shape: [c4]
  """

  def __init__(
      self,
      *,
      equation: str,
      input_shape: types.ShapeLike,
      output_shape: tuple[int | None, ...],
      bias_axes: str = '',
      activation: Callable[[jax.Array], jax.Array] | None = None,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      precision=None,
      kernel_init=nnx.initializers.lecun_normal(),
      bias_init=nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self._equation = equation
    self._input_shape = tuple(input_shape)
    self._output_shape_spec = tuple(output_shape)
    self._bias_axes = bias_axes
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.precision = precision

    # Resolve output shape and compute kernel/bias shapes.
    resolved_output = _resolve_output_shape(
        equation, self._input_shape, self._output_shape_spec
    )
    self._resolved_output_shape = resolved_output

    input_spec, kernel_spec, output_spec = _parse_einsum_equation(
        equation
    )
    input_spec_trimmed = input_spec[3:]
    output_spec_trimmed = output_spec[3:]

    # Full shapes with batch+time prefix for the utility function.
    full_input_shape = (0, 0) + tuple(self._input_shape)
    kernel_shape, bias_shape, _ = utils.einsum_analyze_split_string(
        (input_spec_trimmed, kernel_spec, output_spec_trimmed),
        bias_axes,
        full_input_shape,
        list(resolved_output),
        left_elided=True,
    )

    self.kernel = nnx.Param(
        kernel_init(rngs.params(), tuple(kernel_shape), param_dtype)
    )
    self._has_bias = bias_shape is not None
    if self._has_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), tuple(bias_shape), param_dtype)
      )

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self.compute_dtype
    )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return _resolve_output_shape(
        self._equation, input_shape, self._output_shape_spec
    )

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    kernel = self.kernel[...]
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self.compute_dtype
    )
    kernel = kernel.astype(compute_dtype)

    has_bias = self._has_bias
    has_activation = self.activation is not None
    if has_bias:
      bias = self.bias[...].astype(compute_dtype)

    def einsum_fn(v):
      v = v.astype(compute_dtype)
      y = jnp.einsum(
          self._equation, v, kernel, precision=self.precision
      )
      if has_bias:
        y = utils.bias_add(y, bias)
      if has_activation:
        y = self.activation(y)
      return y

    if has_bias or has_activation:
      return x.apply_values(einsum_fn)
    else:
      return x.apply_values_masked(einsum_fn)
