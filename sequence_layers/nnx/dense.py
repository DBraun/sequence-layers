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
