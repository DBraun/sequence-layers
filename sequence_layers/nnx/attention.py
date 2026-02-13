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
"""NNX attention layers."""

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import utils
from sequence_layers.nnx import dense as dense_lib
from sequence_layers.nnx import simple as simple_lib
from sequence_layers.nnx import types


# A negative enough value to underflow to hard zero in softmax.
_INVALID_LOGIT_VALUE = -1e9

# Default kernel initializer for input projections.
# Kernel shape: [input_dimension, num_heads, units_per_head].
_input_projection_default_kernel_init = nnx.initializers.lecun_normal()


class SelfAttentionEmits(struct.PyTreeNode):
  """Emits produced by self attention layers."""
  probabilities: types.Sequence | types.ShapeDType


def _soft_cap_attention_logits(
    logits: jax.Array, cap: float
) -> jax.Array:
  cap = jnp.asarray(cap, logits.dtype)
  return cap * jax.nn.tanh(logits / cap)


def _mask_attention_logits(
    logits: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
  jnp.broadcast_shapes(logits.shape, valid_mask.shape)
  return jnp.where(
      valid_mask,
      logits,
      jnp.asarray(_INVALID_LOGIT_VALUE, dtype=logits.dtype),
  )


def _scale_query(
    queries: jax.Array,
    per_dim_scale: jax.Array | None,
    query_scale: float | None,
) -> jax.Array:
  """Scales queries with per_dim_scale or rsqrt(units_per_head)."""
  if query_scale is None:
    units_per_head = queries.shape[3]
    query_scale = 1 / np.sqrt(units_per_head)

  if per_dim_scale is not None:
    r_softplus_0 = 1.442695041
    scale = jnp.array(r_softplus_0 * query_scale, dtype=queries.dtype)
    queries *= scale * jax.nn.softplus(
        per_dim_scale.astype(queries.dtype)
    )
  else:
    queries *= jnp.array(query_scale, queries.dtype)
  return queries


def _zero_fully_masked_attention_probabilities(
    probabilities: jax.Array,
    logit_visibility_mask: jax.Array,
) -> jax.Array:
  not_fully_masked = jnp.any(
      logit_visibility_mask, keepdims=True, axis=-1
  )
  return jnp.where(
      not_fully_masked,
      probabilities,
      jnp.zeros_like(probabilities),
  )


def _dot_product_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    logit_visibility_mask: jax.Array,
    logit_bias: jax.Array | None,
    attention_logits_soft_cap: float | None,
    attention_probabilities_dropout: simple_lib.Dropout | None,
    per_dim_scale: jax.Array | None,
    query_scale: float | None,
    precision: jax.lax.PrecisionLike,
    zero_fully_masked: bool,
    compute_dtype: types.DType | None,
) -> tuple[jax.Array, jax.Array]:
  """Computes dot product attention.

  Args:
    queries: [batch, query_time, num_query_heads, units_per_head].
    keys: [batch, key_time, num_kv_heads, units_per_head].
    values: [batch, key_time, num_kv_heads, units_per_head].
    logit_visibility_mask: Broadcastable to [batch, num_heads, query_time,
      key_time].
    logit_bias: Optional broadcastable to [batch, num_heads, query_time,
      key_time].
    attention_logits_soft_cap: Optional soft cap for logits.
    attention_probabilities_dropout: Optional dropout layer.
    per_dim_scale: Optional [units_per_head] per-dim scale.
    query_scale: Optional manual query scale.
    precision: Einsum precision.
    zero_fully_masked: Zero out context vectors for fully-masked queries.
    compute_dtype: Dtype for computations.

  Returns:
    context_vectors: [batch, query_time, num_query_heads, units_per_head].
    probabilities: [batch, query_time, num_query_heads, key_time].
  """
  num_heads = queries.shape[2]
  num_kv_heads = keys.shape[2]

  if num_heads != num_kv_heads:
    return _dot_product_attention_gqa(
        queries, keys, values, logit_visibility_mask, logit_bias,
        attention_logits_soft_cap, attention_probabilities_dropout,
        per_dim_scale, query_scale, precision, zero_fully_masked,
        compute_dtype,
    )

  q_dtype = utils.get_promoted_dtype(queries.dtype, dtype=compute_dtype)
  qk_dtype = utils.get_promoted_dtype(
      q_dtype, keys.dtype, dtype=compute_dtype
  )
  qkv_dtype = utils.get_promoted_dtype(
      qk_dtype, values.dtype, dtype=compute_dtype
  )
  queries = _scale_query(
      queries.astype(q_dtype), per_dim_scale, query_scale
  )
  queries = queries.astype(qk_dtype)
  keys = keys.astype(qk_dtype)

  logits = jnp.einsum(
      'BiNH,BjNH->BNij', queries, keys, precision=precision
  )

  if logit_bias is not None:
    jnp.broadcast_shapes(logit_bias.shape, logits.shape)
    logits += logit_bias.astype(logits.dtype)

  if attention_logits_soft_cap:
    logits = _soft_cap_attention_logits(logits, attention_logits_soft_cap)

  logits = _mask_attention_logits(logits, logit_visibility_mask)

  probabilities = utils.run_in_at_least_fp32(
      lambda l: jax.nn.softmax(l, axis=-1).astype(qkv_dtype),
      restore_dtypes=False,
  )(logits)

  if attention_probabilities_dropout:
    probabilities = attention_probabilities_dropout.apply_dropout(
        probabilities
    )

  if zero_fully_masked:
    probabilities = _zero_fully_masked_attention_probabilities(
        probabilities, logit_visibility_mask
    )

  context_vectors = jnp.einsum(
      'BNts,BsNH->BtNH',
      probabilities,
      values.astype(qkv_dtype),
      precision=precision,
  )

  # Transpose probabilities to [batch, query_time, num_heads, key_time].
  probabilities = jnp.transpose(probabilities, [0, 2, 1, 3])
  return context_vectors, probabilities


def _dot_product_attention_gqa(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    logit_visibility_mask: jax.Array,
    logit_bias: jax.Array | None,
    attention_logits_soft_cap: float | None,
    attention_probabilities_dropout: simple_lib.Dropout | None,
    per_dim_scale: jax.Array | None,
    query_scale: float | None,
    precision: jax.lax.PrecisionLike,
    zero_fully_masked: bool,
    compute_dtype: types.DType | None,
) -> tuple[jax.Array, jax.Array]:
  """Dot product attention with Grouped Query Attention (GQA)."""
  num_heads = queries.shape[2]
  num_kv_heads = keys.shape[2]

  if num_heads % num_kv_heads != 0:
    raise ValueError(
        f'{num_heads=} must be divisible by {num_kv_heads=}'
    )

  num_query_heads_per_kv_head = num_heads // num_kv_heads

  q_dtype = utils.get_promoted_dtype(queries.dtype, dtype=compute_dtype)
  qk_dtype = utils.get_promoted_dtype(
      q_dtype, keys.dtype, dtype=compute_dtype
  )
  qkv_dtype = utils.get_promoted_dtype(
      qk_dtype, values.dtype, dtype=compute_dtype
  )
  queries = _scale_query(
      queries.astype(q_dtype), per_dim_scale, query_scale
  )
  queries = queries.astype(qk_dtype)
  keys = keys.astype(qk_dtype)

  # Reshape queries to [batch, time, num_kv_heads, heads_per_kv, dim].
  queries = utils.split_dimension(
      queries, axis=2,
      shape=(num_kv_heads, num_query_heads_per_kv_head),
  )

  logits = jnp.einsum(
      'BjKH,BiKQH->BKQij', keys, queries, precision=precision
  )

  if logit_bias is not None:
    if logit_bias.shape[1] == 1:
      logit_bias = logit_bias[:, :, jnp.newaxis, :, :]
    else:
      logit_bias = utils.split_dimension(
          logit_bias, axis=1,
          shape=(num_kv_heads, num_query_heads_per_kv_head),
      )
    jnp.broadcast_shapes(logit_bias.shape, logits.shape)
    logits += logit_bias.astype(logits.dtype)

  if attention_logits_soft_cap:
    logits = _soft_cap_attention_logits(logits, attention_logits_soft_cap)

  logits = _mask_attention_logits(logits, logit_visibility_mask)

  probabilities = utils.run_in_at_least_fp32(
      lambda l: jax.nn.softmax(l, axis=-1).astype(qkv_dtype),
      restore_dtypes=False,
  )(logits)

  if attention_probabilities_dropout:
    probabilities = attention_probabilities_dropout.apply_dropout(
        probabilities
    )

  if zero_fully_masked:
    probabilities = _zero_fully_masked_attention_probabilities(
        probabilities, logit_visibility_mask
    )

  # Context vectors: [batch, num_kv_heads, heads_per_kv, query, dim].
  context_vectors = jnp.einsum(
      'BKQts,BsKH->BtKQH',
      probabilities,
      values.astype(qkv_dtype),
      precision=precision,
  )
  # Merge back to [batch, query_time, num_heads, units_per_head].
  context_vectors = jnp.reshape(
      context_vectors,
      context_vectors.shape[:2] + (num_heads,) + context_vectors.shape[4:],
  )

  # Merge and transpose probabilities to
  # [batch, query_time, num_heads, key_time].
  probabilities = jnp.reshape(
      probabilities,
      probabilities.shape[:1] + (num_heads,) + probabilities.shape[3:],
  )
  probabilities = jnp.transpose(probabilities, [0, 2, 1, 3])
  return context_vectors, probabilities


def _self_attention_layer_visibility_mask(
    max_past_horizon: int,
    max_future_horizon: int,
    time: int,
) -> jax.Array | None:
  """Compute a layer-wise visibility mask for self-attention."""
  if max_past_horizon != -1 or max_future_horizon != -1:
    num_lower = max(0, time - 1)
    if max_past_horizon != -1:
      num_lower = min(num_lower, max_past_horizon)

    num_upper = max(0, time - 1)
    if max_future_horizon != -1:
      num_upper = min(num_upper, max_future_horizon)

    visibility_mask = utils.ones_matrix_band_part(
        time, time,
        num_lower=num_lower, num_upper=num_upper,
        out_shape=[1, 1, time, time], out_dtype=jnp.bool_,
    )
  else:
    visibility_mask = None
  return visibility_mask


def _self_attention_step_visibility_mask(
    max_past_horizon: int,
    max_future_horizon: int,
    query_time: int,
    key_time: int,
) -> jax.Array | None:
  """Compute a step-wise visibility mask for self-attention."""
  if query_time == 1:
    return None
  else:
    return utils.ones_matrix_band_part(
        query_time, key_time,
        num_lower=0,
        num_upper=max_past_horizon + max_future_horizon,
        out_shape=[1, 1, query_time, key_time],
        out_dtype=jnp.bool_,
    )


class DotProductSelfAttention(types.Emitting):
  """A multi-headed dot-product self attention layer.

  Supports Grouped Query Attention (GQA) via num_kv_heads.
  """

  def __init__(
      self,
      *,
      num_heads: int,
      units_per_head: int,
      max_past_horizon: int,
      max_future_horizon: int = 0,
      num_kv_heads: int | None = None,
      attention_probabilities_dropout_rate: float = 0.0,
      use_bias: bool = False,
      per_dim_scale: bool = False,
      query_scale: float | None = None,
      attention_logits_soft_cap: float | None = None,
      zero_fully_masked: bool = False,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      precision: jax.lax.PrecisionLike = None,
      q_kernel_init: nnx.initializers.Initializer = (
          _input_projection_default_kernel_init
      ),
      k_kernel_init: nnx.initializers.Initializer = (
          _input_projection_default_kernel_init
      ),
      v_kernel_init: nnx.initializers.Initializer = (
          _input_projection_default_kernel_init
      ),
      bias_init: nnx.initializers.Initializer = (
          nnx.initializers.zeros_init()
      ),
      in_features: int,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    if num_heads <= 0:
      raise ValueError(f'num_heads must be positive, got {num_heads}.')
    if units_per_head <= 0:
      raise ValueError(
          f'units_per_head must be positive, got {units_per_head}.'
      )
    if max_past_horizon < -1:
      raise ValueError(
          f'max_past_horizon must be >= -1, got {max_past_horizon}.'
      )
    if max_future_horizon < -1:
      raise ValueError(
          f'max_future_horizon must be >= -1, got {max_future_horizon}.'
      )
    if max_past_horizon == 0 and max_future_horizon == 0:
      raise ValueError(
          'Both max_past_horizon and max_future_horizon are 0.'
      )
    if attention_logits_soft_cap and attention_logits_soft_cap < 0.0:
      raise ValueError(
          f'{attention_logits_soft_cap=} should be None or non-negative.'
      )

    self._num_heads = num_heads
    self._units_per_head = units_per_head
    self._max_past_horizon = max_past_horizon
    self._max_future_horizon = max_future_horizon
    self._num_kv_heads = num_kv_heads or num_heads
    self._use_bias = use_bias
    self._query_scale = query_scale
    self._attention_logits_soft_cap = attention_logits_soft_cap
    self._zero_fully_masked = zero_fully_masked
    self._compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self._precision = precision

    if num_heads % self._num_kv_heads != 0:
      raise ValueError(
          f'{num_heads=} must be divisible by '
          f'{self._num_kv_heads=}.'
      )

    # Q/K/V projection kernels.
    self.q_kernel = nnx.Param(
        q_kernel_init(
            rngs.params(),
            (in_features, num_heads, units_per_head),
            param_dtype,
        )
    )
    self.k_kernel = nnx.Param(
        k_kernel_init(
            rngs.params(),
            (in_features, self._num_kv_heads, units_per_head),
            param_dtype,
        )
    )
    self.v_kernel = nnx.Param(
        v_kernel_init(
            rngs.params(),
            (in_features, self._num_kv_heads, units_per_head),
            param_dtype,
        )
    )
    if use_bias:
      self.q_bias = nnx.Param(
          bias_init(
              rngs.params(),
              (num_heads, units_per_head),
              param_dtype,
          )
      )
      self.k_bias = nnx.Param(
          bias_init(
              rngs.params(),
              (self._num_kv_heads, units_per_head),
              param_dtype,
          )
      )
      self.v_bias = nnx.Param(
          bias_init(
              rngs.params(),
              (self._num_kv_heads, units_per_head),
              param_dtype,
          )
      )

    # Per-dim scale.
    self._has_per_dim_scale = per_dim_scale
    if per_dim_scale:
      self.per_dim_scale = nnx.Param(
          jnp.zeros([units_per_head], dtype=param_dtype)
      )

    # Dropout.
    self._attention_probabilities_dropout = simple_lib.Dropout(
        rate=attention_probabilities_dropout_rate,
        rngs=rngs if attention_probabilities_dropout_rate > 0 else None,
    )

  def _get_per_dim_scale(self) -> jax.Array | None:
    if self._has_per_dim_scale:
      return self.per_dim_scale[...]
    return None

  def _project_qkv(
      self, x: types.Sequence
  ) -> tuple[types.Sequence, types.Sequence, types.Sequence]:
    """Projects input to query, key, value sequences."""
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self._compute_dtype
    )

    def project(values, kernel, bias=None):
      k = kernel[...].astype(compute_dtype)
      v = values.astype(compute_dtype)
      result = jnp.einsum('...a,abc->...bc', v, k)
      if bias is not None:
        result += bias[...].astype(compute_dtype)
      return result

    q_bias = self.q_bias if self._use_bias else None
    k_bias = self.k_bias if self._use_bias else None
    v_bias = self.v_bias if self._use_bias else None

    queries = x.apply_values_masked(
        lambda v: project(v, self.q_kernel, q_bias)
    )
    keys = x.apply_values_masked(
        lambda v: project(v, self.k_kernel, k_bias)
    )
    values = x.apply_values_masked(
        lambda v: project(v, self.v_kernel, v_bias)
    )
    return queries, keys, values

  @property
  def supports_step(self) -> bool:
    return (
        self._max_future_horizon >= 0
        and self._max_past_horizon >= 0
    )

  @property
  def input_latency(self) -> int:
    return (
        self._max_future_horizon
        if self._max_future_horizon >= 0
        else 0
    )

  @property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    start = (
        -np.inf
        if self._max_past_horizon == -1
        else -self._max_past_horizon
    )
    end = (
        np.inf
        if self._max_future_horizon == -1
        else self._max_future_horizon
    )
    return {0: (start, end)}

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self._compute_dtype
    )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 1:
      raise ValueError(
          'DotProductSelfAttention requires rank 3 input, got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    return (self._num_heads, self._units_per_head)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    compute_dtype = utils.get_promoted_dtype(
        input_spec.dtype, self._param_dtype, dtype=self._compute_dtype
    )
    max_past = max(0, self._max_past_horizon)
    max_future = max(0, self._max_future_horizon)
    kv_buffer_size = max_past + max_future

    kv_zero_values = jnp.zeros(
        (batch_size, kv_buffer_size, self._num_kv_heads,
         self._units_per_head),
        dtype=compute_dtype,
    )
    kv_zero_mask = jnp.zeros(
        [batch_size, kv_buffer_size], dtype=types.MASK_DTYPE
    )

    kv_buffer_keys = kv_zero_values
    kv_buffer_values = kv_zero_values
    kv_buffer_mask = kv_zero_mask

    if max_future:
      query_delay_buffer = types.Sequence(
          jnp.zeros(
              (batch_size, max_future, self._num_heads,
               self._units_per_head),
              dtype=compute_dtype,
          ),
          jnp.zeros(
              [batch_size, max_future], dtype=types.MASK_DTYPE
          ),
      )
    else:
      query_delay_buffer = ()

    time_step = jnp.zeros((batch_size,), dtype=jnp.int32)

    return (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        time_step,
        query_delay_buffer,
    )

  @types.check_step_with_emits
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    if not self.supports_step:
      raise ValueError(f'{type(self).__name__} is not steppable.')

    x_values_time = x.shape[1]
    x_queries, x_keys, x_values = self._project_qkv(x)

    (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        time_step,
        query_delay_buffer,
    ) = state

    kv_buffer_size = kv_buffer_keys.shape[1]

    # Guard against NaN/Inf in values.
    x_values = x_values.mask_invalid()
    combined_mask = utils.combine_mask(x_keys.mask, x_values.mask)

    # Concatenate KV buffer with new KV.
    kv_buffer_keys = jnp.concatenate(
        [kv_buffer_keys, x_keys.values], axis=1
    )
    kv_buffer_values = jnp.concatenate(
        [kv_buffer_values, x_values.values], axis=1
    )
    kv_buffer_mask = jnp.concatenate(
        [kv_buffer_mask, combined_mask], axis=1
    )
    kv_buffer_time = x_values_time + kv_buffer_size

    # Handle query delay buffer.
    if query_delay_buffer:
      query_delay_buffer = types.Sequence.concatenate_sequences(
          [query_delay_buffer, x_queries]
      )
      x_queries = query_delay_buffer[:, :x_values_time]
      query_delay_buffer = query_delay_buffer[
          :, -self._max_future_horizon:
      ]

    valid_mask = kv_buffer_mask[:, jnp.newaxis, jnp.newaxis, :]
    visibility_mask = _self_attention_step_visibility_mask(
        self._max_past_horizon,
        self._max_future_horizon,
        x_values_time,
        kv_buffer_time,
    )
    if visibility_mask is not None:
      valid_mask = jnp.logical_and(valid_mask, visibility_mask)

    per_dim_scale = self._get_per_dim_scale()

    context_vectors, probabilities = _dot_product_attention(
        queries=x_queries.values,
        keys=kv_buffer_keys,
        values=kv_buffer_values,
        logit_visibility_mask=valid_mask,
        logit_bias=None,
        attention_logits_soft_cap=self._attention_logits_soft_cap,
        attention_probabilities_dropout=(
            self._attention_probabilities_dropout
        ),
        per_dim_scale=per_dim_scale,
        query_scale=self._query_scale,
        precision=self._precision,
        zero_fully_masked=self._zero_fully_masked,
        compute_dtype=self._compute_dtype,
    )

    # Preserve last kv_buffer_size timesteps.
    kv_buffer_keys = kv_buffer_keys[:, -kv_buffer_size:]
    kv_buffer_values = kv_buffer_values[:, -kv_buffer_size:]
    kv_buffer_mask = kv_buffer_mask[:, -kv_buffer_size:]

    state = (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        time_step + x_values_time,
        query_delay_buffer,
    )

    emits = SelfAttentionEmits(
        types.Sequence(probabilities, x_queries.mask)
    )
    context_vectors = types.Sequence(
        context_vectors, x_queries.mask
    )
    return context_vectors, state, emits

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    values_time = x.shape[1]

    queries, keys, values = self._project_qkv(x)

    # Guard against NaN/Inf in values.
    values = values.mask_invalid()

    valid_mask = x.mask[:, jnp.newaxis, jnp.newaxis, :]
    visibility_mask = _self_attention_layer_visibility_mask(
        self._max_past_horizon,
        self._max_future_horizon,
        values_time,
    )
    if visibility_mask is not None:
      valid_mask = jnp.logical_and(visibility_mask, valid_mask)

    per_dim_scale = self._get_per_dim_scale()

    context_vectors, probabilities = _dot_product_attention(
        queries=queries.values,
        keys=keys.values,
        values=values.values,
        logit_visibility_mask=valid_mask,
        logit_bias=None,
        attention_logits_soft_cap=self._attention_logits_soft_cap,
        attention_probabilities_dropout=(
            self._attention_probabilities_dropout
        ),
        per_dim_scale=per_dim_scale,
        query_scale=self._query_scale,
        precision=self._precision,
        zero_fully_masked=self._zero_fully_masked,
        compute_dtype=self._compute_dtype,
    )

    emits = SelfAttentionEmits(
        types.Sequence(probabilities, x.mask)
    )
    context_vectors = types.Sequence(context_vectors, x.mask)
    return context_vectors, emits


class CrossAttentionEmits(struct.PyTreeNode):
  """Emits produced by cross-attention layers."""
  probabilities: types.Sequence | types.ShapeDType


class DotProductAttention(types.Emitting):
  """A multi-headed dot-product cross attention layer.

  Queries come from the input sequence. Keys and values come from an
  external source sequence provided via constants.
  """

  def __init__(
      self,
      *,
      source_name: str,
      num_heads: int,
      units_per_head: int,
      num_kv_heads: int | None = None,
      attention_probabilities_dropout_rate: float = 0.0,
      use_bias: bool = False,
      per_dim_scale: bool = False,
      query_scale: float | None = None,
      attention_logits_soft_cap: float | None = None,
      zero_fully_masked: bool = False,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      precision: jax.lax.PrecisionLike = None,
      q_kernel_init: nnx.initializers.Initializer = (
          _input_projection_default_kernel_init
      ),
      k_kernel_init: nnx.initializers.Initializer = (
          _input_projection_default_kernel_init
      ),
      v_kernel_init: nnx.initializers.Initializer = (
          _input_projection_default_kernel_init
      ),
      bias_init: nnx.initializers.Initializer = (
          nnx.initializers.zeros_init()
      ),
      in_features: int,
      source_features: int,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    if not source_name:
      raise ValueError('source_name must be non-empty.')
    if num_heads <= 0:
      raise ValueError(f'num_heads must be positive, got {num_heads}.')
    if units_per_head <= 0:
      raise ValueError(
          f'units_per_head must be positive, got {units_per_head}.'
      )
    if attention_logits_soft_cap and attention_logits_soft_cap < 0.0:
      raise ValueError(
          f'{attention_logits_soft_cap=} should be None or non-negative.'
      )

    self._source_name = source_name
    self._num_heads = num_heads
    self._units_per_head = units_per_head
    self._num_kv_heads = num_kv_heads or num_heads
    self._use_bias = use_bias
    self._query_scale = query_scale
    self._attention_logits_soft_cap = attention_logits_soft_cap
    self._zero_fully_masked = zero_fully_masked
    self._compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self._precision = precision

    if num_heads % self._num_kv_heads != 0:
      raise ValueError(
          f'{num_heads=} must be divisible by '
          f'{self._num_kv_heads=}.'
      )

    # Query projection (from input features).
    self.q_kernel = nnx.Param(
        q_kernel_init(
            rngs.params(),
            (in_features, num_heads, units_per_head),
            param_dtype,
        )
    )
    # Key and value projections (from source features).
    self.k_kernel = nnx.Param(
        k_kernel_init(
            rngs.params(),
            (source_features, self._num_kv_heads, units_per_head),
            param_dtype,
        )
    )
    self.v_kernel = nnx.Param(
        v_kernel_init(
            rngs.params(),
            (source_features, self._num_kv_heads, units_per_head),
            param_dtype,
        )
    )
    if use_bias:
      self.q_bias = nnx.Param(
          bias_init(
              rngs.params(),
              (num_heads, units_per_head),
              param_dtype,
          )
      )
      self.k_bias = nnx.Param(
          bias_init(
              rngs.params(),
              (self._num_kv_heads, units_per_head),
              param_dtype,
          )
      )
      self.v_bias = nnx.Param(
          bias_init(
              rngs.params(),
              (self._num_kv_heads, units_per_head),
              param_dtype,
          )
      )

    # Per-dim scale.
    self._has_per_dim_scale = per_dim_scale
    if per_dim_scale:
      self.per_dim_scale = nnx.Param(
          jnp.zeros([units_per_head], dtype=param_dtype)
      )

    # Dropout.
    self._attention_probabilities_dropout = simple_lib.Dropout(
        rate=attention_probabilities_dropout_rate,
        rngs=rngs if attention_probabilities_dropout_rate > 0 else None,
    )

  def _get_source(
      self, constants: types.Constants | None
  ) -> types.Sequence:
    """Gets the source sequence from constants."""
    if constants is None:
      raise ValueError(
          f'{type(self).__name__} requires constants with '
          f'source_name={self._source_name!r}.'
      )
    source = constants.get(self._source_name)
    if source is None:
      raise ValueError(
          f'{type(self).__name__} expected {self._source_name!r} in '
          f'constants.'
      )
    if not isinstance(source, types.Sequence):
      raise ValueError(
          f'{type(self).__name__} expected a Sequence for '
          f'{self._source_name!r}, got: {type(source)}.'
      )
    return source

  def _project_q(
      self, x: types.Sequence
  ) -> types.Sequence:
    """Projects input to query."""
    compute_dtype = utils.get_promoted_dtype(
        x.dtype, self._param_dtype, dtype=self._compute_dtype
    )

    def project(values):
      k = self.q_kernel[...].astype(compute_dtype)
      v = values.astype(compute_dtype)
      result = jnp.einsum('...a,abc->...bc', v, k)
      if self._use_bias:
        result += self.q_bias[...].astype(compute_dtype)
      return result

    return x.apply_values_masked(project)

  def _project_kv(
      self, source: types.Sequence
  ) -> tuple[types.Sequence, types.Sequence]:
    """Projects source to key, value."""
    compute_dtype = utils.get_promoted_dtype(
        source.dtype, self._param_dtype, dtype=self._compute_dtype
    )

    def project_k(values):
      k = self.k_kernel[...].astype(compute_dtype)
      v = values.astype(compute_dtype)
      result = jnp.einsum('...a,abc->...bc', v, k)
      if self._use_bias:
        result += self.k_bias[...].astype(compute_dtype)
      return result

    def project_v(values):
      k = self.v_kernel[...].astype(compute_dtype)
      v = values.astype(compute_dtype)
      result = jnp.einsum('...a,abc->...bc', v, k)
      if self._use_bias:
        result += self.v_bias[...].astype(compute_dtype)
      return result

    keys = source.apply_values_masked(project_k)
    values = source.apply_values_masked(project_v)
    return keys, values

  def _get_per_dim_scale(self) -> jax.Array | None:
    if self._has_per_dim_scale:
      return self.per_dim_scale[...]
    return None

  @property
  def supports_step(self) -> bool:
    return True

  @property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    return {0: (0, 0)}

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self._param_dtype, dtype=self._compute_dtype
    )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 1:
      raise ValueError(
          'DotProductAttention requires rank 3 input, got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    return (self._num_heads, self._units_per_head)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    # Pre-project keys and values from the source.
    source = self._get_source(constants)
    keys, values = self._project_kv(source)
    values = values.mask_invalid()
    return (keys.values, values.values, source.mask)

  @types.check_step_with_emits
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    keys_values, values_values, source_mask = state
    queries = self._project_q(x)

    valid_mask = source_mask[:, jnp.newaxis, jnp.newaxis, :]

    per_dim_scale = self._get_per_dim_scale()

    context_vectors, probabilities = _dot_product_attention(
        queries=queries.values,
        keys=keys_values,
        values=values_values,
        logit_visibility_mask=valid_mask,
        logit_bias=None,
        attention_logits_soft_cap=self._attention_logits_soft_cap,
        attention_probabilities_dropout=(
            self._attention_probabilities_dropout
        ),
        per_dim_scale=per_dim_scale,
        query_scale=self._query_scale,
        precision=self._precision,
        zero_fully_masked=self._zero_fully_masked,
        compute_dtype=self._compute_dtype,
    )

    emits = CrossAttentionEmits(
        types.Sequence(probabilities, x.mask)
    )
    context_vectors = types.Sequence(context_vectors, x.mask)
    return context_vectors, state, emits

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    source = self._get_source(constants)
    queries = self._project_q(x)
    keys, values = self._project_kv(source)
    values = values.mask_invalid()

    valid_mask = source.mask[:, jnp.newaxis, jnp.newaxis, :]

    per_dim_scale = self._get_per_dim_scale()

    context_vectors, probabilities = _dot_product_attention(
        queries=queries.values,
        keys=keys.values,
        values=values.values,
        logit_visibility_mask=valid_mask,
        logit_bias=None,
        attention_logits_soft_cap=self._attention_logits_soft_cap,
        attention_probabilities_dropout=(
            self._attention_probabilities_dropout
        ),
        per_dim_scale=per_dim_scale,
        query_scale=self._query_scale,
        precision=self._precision,
        zero_fully_masked=self._zero_fully_masked,
        compute_dtype=self._compute_dtype,
    )

    emits = CrossAttentionEmits(
        types.Sequence(probabilities, x.mask)
    )
    context_vectors = types.Sequence(context_vectors, x.mask)
    return context_vectors, emits


# ---------------------------------------------------------------------------
# Helper: step-by-step with emits (used by GmmAttention monotonic mode)
# ---------------------------------------------------------------------------


def _step_by_step_with_emits(
    l: types.Emitting,
    x: types.Sequence,
    initial_state: types.State,
    *,
    constants: types.Constants | None = None,
) -> tuple[types.Sequence, types.State, types.Emits]:
  """Executes a layer one timestep at a time, collecting emits.

  Used by GmmAttention in monotonic mode to process multi-timestep inputs
  sequentially (since each step depends on the previous position state).

  Args:
    l: The Emitting layer to invoke step-by-step.
    x: The input Sequence to process.
    initial_state: The initial state.
    constants: Optional constants.

  Returns:
    Tuple of (output_sequence, final_state, stacked_emits).
  """
  state = initial_state
  output_blocks = []
  emits_list = []

  for t in range(x.shape[1]):
    x_t = x[:, t:t + 1]
    y_t, state, emits_t = l.step_with_emits(
        x_t, state, constants=constants
    )
    output_blocks.append(y_t)
    emits_list.append(emits_t)

  output = types.Sequence.concatenate_sequences(output_blocks)
  # Stack emits along the time dimension.
  emits = jax.tree.map(
      lambda *args: types.Sequence.concatenate_sequences(list(args))
      if isinstance(args[0], types.Sequence)
      else jnp.concatenate(args, axis=1),
      *emits_list,
  )
  return output, state, emits


# ---------------------------------------------------------------------------
# GmmAttention
# ---------------------------------------------------------------------------


class GmmAttention(types.PreservesType, types.Emitting):
  """A multi-headed Gaussian-mixture attention layer.

  Uses GMMs to model the probability distribution of where to focus
  attention in the source sequence.
  """

  def __init__(
      self,
      *,
      source_name: str,
      num_heads: int,
      units_per_head: int,
      num_components: int,
      monotonic: bool,
      in_features: int,
      precision: jax.lax.PrecisionLike = None,
      hidden_kernel_init: nnx.initializers.Initializer = (
          nnx.initializers.lecun_normal()
      ),
      output_kernel_init: nnx.initializers.Initializer = (
          nnx.initializers.lecun_normal()
      ),
      output_use_bias: bool = True,
      output_bias_init: nnx.initializers.Initializer = (
          nnx.initializers.zeros_init()
      ),
      init_offset_bias: float = 0.0,
      init_scale_bias: float = 0.0,
      max_offset: float = -1.0,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    if not source_name:
      raise ValueError('source_name must be non-empty.')
    if num_heads <= 0:
      raise ValueError(
          f'num_heads must be positive, got {num_heads}.'
      )
    if units_per_head <= 0:
      raise ValueError(
          f'units_per_head must be positive, got {units_per_head}.'
      )
    if num_components <= 0:
      raise ValueError(
          f'num_components must be positive, got {num_components}.'
      )

    self._source_name = source_name
    self._num_heads = num_heads
    self._units_per_head = units_per_head
    self._num_components = num_components
    self._monotonic = monotonic
    self._precision = precision
    self._init_offset_bias = init_offset_bias
    self._init_scale_bias = init_scale_bias
    self._max_offset = max_offset

    # Hidden MLP: ...c,cnu -> ...nu
    self._mlp_hidden = dense_lib.EinsumDense(
        equation='...c,cnu->...nu',
        input_shape=(in_features,),
        output_shape=(num_heads, units_per_head),
        precision=precision,
        kernel_init=hidden_kernel_init,
        activation=jax.nn.relu,
        rngs=rngs,
    )

    # Output MLP: ...nu,nucl -> ...ncl
    # Final axis holds 3 logits: prior, offset, scale.
    self._mlp_output = dense_lib.EinsumDense(
        equation='...nu,nucl->...ncl',
        input_shape=(num_heads, units_per_head),
        output_shape=(num_heads, num_components, 3),
        precision=precision,
        kernel_init=output_kernel_init,
        bias_axes='ncl' if output_use_bias else '',
        bias_init=output_bias_init,
        rngs=rngs,
    )

  def _get_source(
      self, constants: types.Constants | None
  ) -> types.Sequence:
    """Gets the source sequence from constants."""
    if constants is None:
      raise ValueError(
          f'{type(self).__name__} requires constants with '
          f'source_name={self._source_name!r}.'
      )
    source = constants.get(self._source_name)
    if source is None:
      raise ValueError(
          f'{type(self).__name__} expected {self._source_name!r} '
          f'in constants.'
      )
    if not isinstance(source, types.Sequence):
      raise ValueError(
          f'{type(self).__name__} expected a Sequence for '
          f'{self._source_name!r}, got: {type(source)}.'
      )
    return source

  @property
  def supports_step(self) -> bool:
    return True

  @property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    start = -np.inf if self._monotonic else 0
    return {0: (start, 0)}

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 1:
      raise ValueError(
          'GmmAttention requires rank 3 input, got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    source = self._get_source(constants)
    return (self._num_heads, source.shape[2])

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    if self._monotonic:
      return jnp.zeros(
          [batch_size, 1, self._num_heads, self._num_components],
          input_spec.dtype,
      )
    else:
      return ()

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    position = self.get_initial_state(
        x.shape[0], x.channel_spec, constants=constants
    )

    if self._monotonic and x.shape[1] != 1:
      context_vector, _, attention_weights = (
          _step_by_step_with_emits(
              self, x, position, constants=constants
          )
      )
      return context_vector, attention_weights

    source = self._get_source(constants)
    context_vector, _, attention_weights = self.attention(
        source=source, query=x, prev_position=position,
    )
    return context_vector, attention_weights

  @types.check_step_with_emits
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    source = self._get_source(constants)
    if self._monotonic and x.shape[1] != 1:
      return _step_by_step_with_emits(
          self, x, state, constants=constants
      )
    context_vector, new_position, attention_weights = self.attention(
        source=source, query=x, prev_position=state,
    )
    return context_vector, new_position, attention_weights

  def attention(
      self,
      source: types.Sequence,
      query: types.Sequence,
      prev_position: types.State,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    """Computes GMM attention from query and previous position.

    Args:
      source: [batch, source_time, source_dim].
      query: [batch, query_time, query_channels].
      prev_position: [batch, 1, num_heads, num_components].

    Returns:
      context_vector: [batch, query_time, num_heads, source_dim].
      position: [batch, 1, num_heads, num_components] or ().
      attention_emits: CrossAttentionEmits.
    """
    if query.values.ndim != 3:
      raise ValueError(
          f'Expected [b, t, d] inputs, got: {query.values.shape}.'
      )

    query_time = query.values.shape[1]
    if self._monotonic:
      assert query_time == 1, (
          f'Expected [b, 1, d] inputs, got: {query.values.shape}.'
      )

    # Project query through two-layer MLP.
    query = self._mlp_hidden.layer(query)
    query = self._mlp_output.layer(query)

    # Each is [batch, query_time, num_heads, num_components].
    prior_logits, offset_logits, scale_logits = utils.unstack(
        query.values, axis=4
    )

    offset_logits = offset_logits + self._init_offset_bias
    scale_logits = scale_logits + self._init_scale_bias

    # Evaluate GMM PDFs.
    attention_weights, new_position = self._eval_gmm_pdfs(
        source, prior_logits, offset_logits, scale_logits,
        prev_position,
    )

    # Mask invalid source positions.
    attention_weights = jnp.where(
        source.mask[:, jnp.newaxis, jnp.newaxis, :],
        attention_weights,
        jnp.zeros_like(attention_weights),
    )

    # Weighted sum over source:
    # [b, query_time, num_heads, source_time] @
    # [b, source_time, source_dim]
    # -> [b, query_time, num_heads, source_dim]
    context_vector = jnp.einsum(
        'BiNj,BjS->BiNS',
        attention_weights,
        source.values,
        precision=self._precision,
    )

    state = new_position if self._monotonic else ()
    attention_emits = CrossAttentionEmits(
        types.Sequence(attention_weights, query.mask)
    )
    return (
        types.Sequence(context_vector, query.mask),
        state,
        attention_emits,
    )

  def _eval_gmm_pdfs(
      self,
      source: types.Sequence,
      prior_logits: jax.Array,
      offset_logits: jax.Array,
      scale_logits: jax.Array,
      prev_position: types.State,
      normalize: bool = True,
  ) -> tuple[jax.Array, jax.Array]:
    """Evaluate the location GMMs on all encoder positions.

    Args:
      source: The source sequence.
      prior_logits: [b, query_time, num_heads, num_components].
      offset_logits: [b, query_time, num_heads, num_components].
      scale_logits: [b, query_time, num_heads, num_components].
      prev_position: [b, 1, num_heads, num_components].
      normalize: Whether to normalize attention probabilities.

    Returns:
      attention_weights: [b, query_time, num_heads, source_time].
      new_position: [b, query_time, num_heads, num_components].
    """
    priors = utils.run_in_at_least_fp32(jax.nn.softmax)(
        prior_logits
    )
    variances = jnp.square(jax.nn.softplus(scale_logits))
    if self._max_offset > 0:
      position_offset = (
          jax.nn.softplus(offset_logits)
          - jax.nn.softplus(offset_logits - self._max_offset)
      )
    else:
      position_offset = jax.nn.softplus(offset_logits)

    if self._monotonic:
      new_position = prev_position + position_offset
    else:
      new_position = position_offset

    # Expand to [b, query_time, num_heads, 1, num_components].
    priors = priors[:, :, :, jnp.newaxis, :]
    means = new_position[:, :, :, jnp.newaxis, :]
    variances = variances[:, :, :, jnp.newaxis, :]

    # [1, 1, 1, source_time, 1]
    source_length = source.values.shape[1]
    encoder_positions = jnp.asarray(
        jnp.arange(source_length), dtype=means.dtype
    )
    encoder_positions = encoder_positions[
        jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis
    ]

    if normalize:
      priors = priors * jnp.sqrt(
          2 * np.pi * variances + 1e-8
      )

    probabilities = priors * jnp.exp(
        -((encoder_positions - means) ** 2)
        / (2 * variances + 1e-9)
    )
    # Sum over components.
    probabilities = jnp.sum(probabilities, 4)
    return probabilities, new_position
