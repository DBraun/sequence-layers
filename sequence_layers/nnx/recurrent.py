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
"""NNX recurrent layers."""

from typing import Callable, Literal

import einops
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import recurrentgemma
from recurrentgemma.jax import layers as rg_layers
from sequence_layers.jax import utils
from sequence_layers.nnx import types


def unit_forget_bias(key, shape, dtype) -> jax.Array:
  """An initializer for LSTM bias that sets the forget gate bias to one."""
  del key
  if len(shape) != 1 or shape[0] % 4 != 0:
    raise ValueError(
        f'Expected a single dimensional shape divisible by 4, got: {shape}.'
    )
  units = shape[0] // 4
  return jnp.concatenate(
      [
          jnp.zeros([units], dtype),
          jnp.ones([units], dtype),
          jnp.zeros([2 * units], dtype),
      ],
      axis=0,
  )


def orthogonal_init(scale: float = 1.0, column_axis: int = -1):
  def init(
      key: jax.Array, shape: types.Shape, dtype: types.DType
  ) -> jax.Array:
    value = nnx.initializers.orthogonal(scale=scale, column_axis=column_axis)(
        key, shape, jnp.float32
    )
    return value.astype(dtype)
  return init


class LSTM(types.SequenceLayer):
  """A Long Short-term Memory (LSTM) layer."""

  def __init__(
      self,
      *,
      in_features: int,
      units: int,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      precision=None,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh,
      recurrent_activation: Callable[[jax.Array], jax.Array] = jax.nn.sigmoid,
      use_bias: bool = True,
      kernel_init=nnx.initializers.lecun_normal(),
      recurrent_kernel_init=None,
      bias_init=unit_forget_bias,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.in_features = in_features
    self.units = units
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.precision = precision
    self.activation = activation
    self.recurrent_activation = recurrent_activation
    self._use_bias = use_bias

    if recurrent_kernel_init is None:
      recurrent_kernel_init = orthogonal_init()

    # Input projection: [in_features, 4 * units]
    self.kernel = nnx.Param(
        kernel_init(rngs.params(), (in_features, 4 * units), param_dtype)
    )
    # Recurrent projection: [units, 4 * units]
    self.recurrent_kernel = nnx.Param(
        recurrent_kernel_init(
            rngs.params(), (units, 4 * units), param_dtype
        )
    )
    if use_bias:
      self.bias = nnx.Param(
          bias_init(rngs.params(), (4 * units,), param_dtype)
      )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return (self.units,)

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
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return {0: (-np.inf, 0)}

  def _cell(
      self,
      x: jax.Array,
      state: tuple[jax.Array, jax.Array],
  ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Runs one LSTM step on [b, 1, d] input."""
    if x.ndim != 3 or x.shape[1] != 1:
      raise ValueError(f'Expected [b, 1, d] inputs, got: {x.shape}.')
    c_tm1, h_tm1 = state

    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self.compute_dtype
    )

    kernel = self.kernel[...].astype(compute_dtype)
    recurrent_kernel = self.recurrent_kernel[...].astype(compute_dtype)
    x = x.astype(compute_dtype)
    h_tm1 = h_tm1.astype(compute_dtype)

    # Project [b, 1, d] -> [b, 1, 4*units]
    z = jnp.einsum(
        '...d,dh->...h', x, kernel, precision=self.precision
    )
    # Project [b, 1, h] -> [b, 1, 4*units]
    z += jnp.einsum(
        '...u,uh->...h', h_tm1, recurrent_kernel,
        precision=self.precision,
    )

    if self._use_bias:
      bias = self.bias[...].astype(compute_dtype)
      z = utils.bias_add(z, bias)

    z0, z1, z2, z3 = jnp.split(
        z,
        [self.units, 2 * self.units, 3 * self.units],
        axis=-1,
    )

    i = self.recurrent_activation(z0)
    f = self.recurrent_activation(z1)
    c = f * c_tm1.astype(compute_dtype) + i * self.activation(z2)
    o = self.recurrent_activation(z3)
    h = o * self.activation(c)
    return h, (c, h)

  def _step_one(
      self,
      x_t: types.Sequence,
      state: types.State,
  ) -> tuple[types.Sequence, types.State]:
    """Processes a single timestep [b, 1, d]."""
    y, new_state = self._cell(x_t.values, state)

    def copy_state_through(
        new_a: jax.Array, a: jax.Array
    ) -> jax.Array:
      assert new_a.ndim >= 2, (new_a.shape, new_a.dtype)
      mask = x_t.mask.reshape(
          x_t.mask.shape + (1,) * (new_a.ndim - 2)
      )
      return jnp.where(mask, new_a, a)

    state = jax.tree.map(copy_state_through, new_state, state)
    return types.Sequence(y, x_t.mask), state

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if x.shape[1] == 1:
      return self._step_one(x, state)

    # For multiple timesteps, unroll statically one-at-a-time.
    outputs = []
    for t in range(x.shape[1]):
      x_t = x[:, t : t + 1]
      y_t, state = self._step_one(x_t, state)
      outputs.append(y_t)
    return types.Sequence.concatenate_sequences(outputs), state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    # Use a Python loop to process timesteps sequentially.
    state = self.get_initial_state(
        batch_size=x.shape[0],
        input_spec=x.channel_spec,
        constants=constants,
    )
    outputs = []
    for t in range(x.shape[1]):
      x_t = x[:, t : t + 1]
      y_t, state = self.step(x_t, state, constants=constants)
      outputs.append(y_t)
    return types.Sequence.concatenate_sequences(outputs)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    compute_dtype = self.get_output_dtype(
        input_spec.dtype, constants=constants
    )
    c = jnp.zeros((batch_size, 1, self.units), dtype=compute_dtype)
    h = jnp.zeros((batch_size, 1, self.units), dtype=compute_dtype)
    return (c, h)


def _rnn_real_param_init(
    min_rad: float,
    max_rad: float,
    transform: str = 'softplus',
    eps: float = 1e-8,
):
  """Initializes the `A` real parameter of the RG-LRU."""

  def init(key, shape, dtype=jnp.float32):
    unif = jax.random.uniform(key, shape=shape)
    a_real = 0.5 * jnp.log(
        unif * (max_rad**2 - min_rad**2) + min_rad**2 + eps
    )
    if transform == 'softplus':
      return jnp.log(jnp.exp(-a_real) - 1.0).astype(dtype)
    else:
      raise NotImplementedError()

  return init


def _rnn_imag_param_init(max_rad: float):
  """Initializes the `A` imag parameter of the RG-LRU."""

  def init(key, shape, dtype=jnp.float32):
    unif = jax.random.uniform(key, shape=shape)
    return (jnp.pi * max_rad * unif).astype(dtype)

  return init


class RGLRU(types.SequenceLayer):
  """A Real-Gated Linear Recurrent Unit (RG-LRU) layer.

  From the Griffin architecture: https://arxiv.org/abs/2402.19427

  Implementation follows https://github.com/google-deepmind/recurrentgemma.

  WARNING: The current implementation is not able to work with anything but
    left-aligned ragged masks. Non-contiguous masks will incorrectly compute
    results on the mask=False timesteps.
  """

  ScanType = Literal[
      'auto', 'linear_native', 'associative_native', 'linear_pallas'
  ]

  def __init__(
      self,
      *,
      in_features: int,
      units: int,
      num_heads: int,
      scan_type: ScanType = 'auto',
      scan_sharding: types.Sharding | None = None,
      only_real: bool = True,
      min_rad: float = 0.9,
      gate_kernel_variance_scale: float = 1.0,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self._units = units
    self._num_heads = num_heads
    self._scan_type_str = scan_type
    self._scan_sharding = scan_sharding
    self._only_real = only_real
    self._compute_dtype = compute_dtype
    self._param_dtype = param_dtype

    if not only_real:
      if units % 2 != 0:
        raise ValueError(
            f'If only_real=False, units must be even, got {units=}.'
        )
      if min_rad >= 0.999:
        raise ValueError(
            'If only_real=False, min_rad must be < 0.999, got'
            f' {min_rad=}.'
        )

    width_output = units if only_real else units // 2
    if width_output % num_heads != 0:
      raise ValueError(
          'num_heads must divide the output width, got'
          f' {num_heads=} and {width_output=}.'
      )
    units_per_head = width_output // num_heads
    # For the gate einsum: ...hi,hij->...hj
    # Input is [batch, time, num_heads, in_features // num_heads]
    in_per_head = in_features // num_heads

    # A real parameter.
    self.a_real_param = nnx.Param(
        _rnn_real_param_init(min_rad=min_rad, max_rad=0.999)(
            rngs.params(), (width_output,), param_dtype
        )
    )

    # A imaginary parameter (only if complex).
    self._has_a_imag = not only_real
    if not only_real:
      self.a_imag_param = nnx.Param(
          _rnn_imag_param_init(max_rad=0.1)(
              rngs.params(), (width_output,), param_dtype
          )
      )

    # Input gate: ...hi,hij->...hj
    gate_kernel_init = nnx.initializers.variance_scaling(
        scale=gate_kernel_variance_scale,
        mode='fan_in',
        distribution='normal',
    )
    self.input_gate_kernel = nnx.Param(
        gate_kernel_init(
            rngs.params(),
            (num_heads, in_per_head, units_per_head),
            param_dtype,
        )
    )
    self.input_gate_bias = nnx.Param(
        nnx.initializers.zeros_init()(
            rngs.params(),
            (num_heads, units_per_head),
            param_dtype,
        )
    )

    # A gate: ...hi,hij->...hj
    self.a_gate_kernel = nnx.Param(
        gate_kernel_init(
            rngs.params(),
            (num_heads, in_per_head, units_per_head),
            param_dtype,
        )
    )
    self.a_gate_bias = nnx.Param(
        nnx.initializers.zeros_init()(
            rngs.params(),
            (num_heads, units_per_head),
            param_dtype,
        )
    )

  @property
  def _scan_type_enum(self) -> recurrentgemma.common.ScanType:
    match self._scan_type_str:
      case 'auto':
        return recurrentgemma.common.ScanType.AUTO
      case 'linear_native':
        return recurrentgemma.common.ScanType.LINEAR_NATIVE
      case 'associative_native':
        return recurrentgemma.common.ScanType.ASSOCIATIVE_NATIVE
      case 'linear_pallas':
        return recurrentgemma.common.ScanType.LINEAR_PALLAS
      case _:
        raise ValueError(
            f'Unknown scan type: {self._scan_type_str}'
        )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return (self._units,)

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self._compute_dtype
    )

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return {0: (-np.inf, 0)}

  def _merged_to_complex(
      self, x: jax.Array
  ) -> recurrentgemma.complex_lib.RealOrComplex:
    if self._only_real:
      return x
    assert x.shape[-1] % 2 == 0
    return self._real_imag_complex(*jnp.split(x, 2, axis=-1))

  def _real_imag_complex(
      self,
      real: jax.Array,
      imag: jax.Array | None,
  ) -> recurrentgemma.complex_lib.RealOrComplex:
    if self._only_real:
      assert imag is None
      return real
    if self._use_custom_complex(real.dtype):
      return recurrentgemma.complex_lib.Complex(real, imag)
    else:
      return real + 1j * imag

  def _use_custom_complex(self, real_dtype: jnp.dtype) -> bool:
    return (
        real_dtype in (jnp.bfloat16, jnp.float16)
        or self._scan_type_enum
        == recurrentgemma.common.ScanType.LINEAR_PALLAS
    )

  def _complex_to_merged(
      self, x: recurrentgemma.complex_lib.RealOrComplex
  ) -> jax.Array:
    if self._only_real:
      assert not isinstance(
          x, recurrentgemma.complex_lib.Complex
      ) and not jnp.iscomplexobj(x)
      return x
    else:
      return jnp.concatenate([x.real, x.imag], axis=-1)

  def _cell(
      self,
      x: types.Sequence,
      h: jax.Array,
      segment_pos: jax.Array,
  ) -> tuple[types.Sequence, jax.Array]:
    mask = x.mask
    x_val = x.values

    compute_dtype = utils.get_promoted_dtype(
        x_val.dtype, self._param_dtype, dtype=self._compute_dtype
    )

    a_real_param = self.a_real_param[...].astype(compute_dtype)
    a_imag_param = None
    if self._has_a_imag:
      a_imag_param = self.a_imag_param[...].astype(compute_dtype)

    x_val = x_val.astype(compute_dtype)

    # Group x into heads.
    x_heads = einops.rearrange(
        x_val, '... (h i) -> ... h i', h=self._num_heads
    )

    # Compute input gate.
    ig_kernel = self.input_gate_kernel[...].astype(compute_dtype)
    ig_bias = self.input_gate_bias[...].astype(compute_dtype)
    gate_x = jnp.einsum('...hi,hij->...hj', x_heads, ig_kernel)
    gate_x = gate_x + ig_bias
    gate_x = recurrentgemma.complex_lib.sigmoid(gate_x)
    gate_x = einops.rearrange(
        gate_x, '... h j -> ... (h j)', h=self._num_heads
    )

    # Compute A gate.
    ag_kernel = self.a_gate_kernel[...].astype(compute_dtype)
    ag_bias = self.a_gate_bias[...].astype(compute_dtype)
    gate_a = jnp.einsum('...hi,hij->...hj', x_heads, ag_kernel)
    gate_a = gate_a + ag_bias
    gate_a = recurrentgemma.complex_lib.sigmoid(gate_a)
    gate_a = einops.rearrange(
        gate_a, '... h j -> ... (h j)', h=self._num_heads
    )

    # Compute the parameter `A` of the recurrence.
    log_a_real = (
        -8.0
        * gate_a
        * recurrentgemma.complex_lib.softplus(a_real_param)
    )

    if self._only_real:
      a = recurrentgemma.complex_lib.exp(log_a_real)
    else:
      log_a_imag = a_imag_param * gate_a
      log_a_complex = self._real_imag_complex(
          log_a_real, log_a_imag
      )
      a = recurrentgemma.complex_lib.exp(log_a_complex)

    mag_a_squared = recurrentgemma.complex_lib.exp(2 * log_a_real)

    x_val = self._merged_to_complex(x_val)

    assert h.dtype == jnp.float32, h.dtype
    h = self._merged_to_complex(h)

    # Gate the input.
    gated_x = x_val * gate_x

    # Apply gamma normalization.
    reset = (segment_pos == 0).astype(a.dtype)
    multiplier = rg_layers.sqrt_bound_derivative(
        1 - mag_a_squared, max_gradient=1000
    )
    multiplier = (
        reset[..., jnp.newaxis]
        + (1 - reset)[..., jnp.newaxis] * multiplier
    )
    normalized_x = gated_x * multiplier.astype(gated_x.dtype)

    y, h = recurrentgemma.scan.linear_scan(
        x=normalized_x,
        a=a * (1 - reset[..., jnp.newaxis]),
        h0=h,
        scan_type=self._scan_type_enum,
        sharding_spec=self._scan_sharding,
        unroll=128,
    )

    y = self._complex_to_merged(y)
    h = self._complex_to_merged(h)
    assert h.dtype == jnp.float32, h.dtype
    return types.Sequence(y, mask), h

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    h, start_index = state
    segment_pos = (
        start_index[:, jnp.newaxis]
        + jnp.arange(x.shape[1])[jnp.newaxis, :]
    )
    y, h = self._cell(x, h, segment_pos)
    return y, (h, start_index + x.shape[1])

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    segment_pos = jnp.arange(x.shape[1])[jnp.newaxis, :]
    h, _ = self.get_initial_state(
        batch_size=x.shape[0],
        input_spec=x.channel_spec,
        constants=constants,
    )
    y, _ = self._cell(x, h, segment_pos)
    return y

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    h = jnp.zeros(
        (batch_size, self._units), dtype=jnp.float32
    )
    start_index = jnp.zeros([batch_size], jnp.int32)
    return h, start_index
