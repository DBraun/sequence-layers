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
"""NNX combinators."""

import fractions
import functools
import math
from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import utils
from sequence_layers.nnx import simple
from sequence_layers.nnx import types


class Serial(types.Emitting):
  """A combinator that processes SequenceLayers serially."""

  def __init__(self, layers: list[types.SequenceLayer]):
    super().__init__()
    self.layers = nnx.List(layers)

  @property
  def supports_step(self) -> bool:
    return all(l.supports_step for l in self.layers)

  @property
  def input_latency(self) -> int:
    return self.get_accumulated_input_latency(0)

  @property
  def output_latency(self) -> int:
    return self.get_accumulated_output_latency(0)

  def get_accumulated_input_latency(self, input_latency: int) -> int:
    return utils.serial_input_latency(self.layers, input_latency)

  def get_accumulated_output_latency(self, output_latency: int) -> int:
    return utils.serial_output_latency(self.layers, output_latency)

  @property
  def block_size(self) -> int:
    return utils.serial_block_size(self.layers)

  @property
  def output_ratio(self) -> fractions.Fraction:
    return utils.serial_output_ratio(self.layers)

  @functools.cached_property
  def receptive_field(self) -> types.ReceptiveField:
    return utils.reduce_receptive_field_per_step(
        self.receptive_field_per_step, self.output_ratio
    )

  @functools.cached_property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return utils.receptive_field_per_step_of_serial_layers(self.layers)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    spec = input_spec
    states = []
    for child_layer in self.layers:
      states.append(
          child_layer.get_initial_state(
              batch_size, spec, constants=constants
          )
      )
      spec = child_layer.get_output_spec(spec, constants=constants)
    return tuple(states)

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    dtype = input_dtype
    for child_layer in self.layers:
      dtype = child_layer.get_output_dtype(dtype, constants=constants)
    return dtype

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    shape = tuple(input_shape)
    for child_layer in self.layers:
      shape = child_layer.get_output_shape(shape, constants=constants)
    return shape

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    new_state = []
    emits = {}
    if len(self.layers) != len(state):
      raise ValueError(
          f'{type(self).__name__} received unexpected state structure:'
          f' {len(self.layers)=} != {len(state)=}'
      )
    for child_layer, state_i in zip(self.layers, state):
      x, state_i, emits_i = child_layer.step_with_emits(
          x, state_i, constants=constants
      )
      new_state.append(state_i)
      emits[child_layer.name] = emits_i
    return x, tuple(new_state), emits

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    emits = {}
    for child_layer in self.layers:
      x, emits_i = child_layer.layer_with_emits(x, constants=constants)
      emits[child_layer.name] = emits_i
    return x, emits


class Residual(types.Emitting):
  """A residual wrapper: ``y = body(x) + shortcut(x)``."""

  def __init__(
      self,
      layers: list[types.SequenceLayer],
      *,
      shortcut: types.SequenceLayer | None = None,
  ):
    super().__init__()
    self._body = Serial(layers)
    self._shortcut = shortcut if shortcut is not None else simple.Identity()

    if self._shortcut.output_ratio != self._body.output_ratio:
      raise ValueError(
          'Residual layers and shortcut must have the same output ratio'
          f' {self._body.output_ratio} !='
          f' {self._shortcut.output_ratio}.'
      )
    if self._shortcut.input_latency != self._body.input_latency:
      raise ValueError(
          'Residual layers and shortcut must have the same input latency'
          f' {self._body.input_latency} !='
          f' {self._shortcut.input_latency}.'
      )

  @property
  def supports_step(self) -> bool:
    return self._body.supports_step and self._shortcut.supports_step

  @property
  def input_latency(self) -> int:
    return self._body.input_latency

  @property
  def output_latency(self) -> int:
    return self._body.output_latency

  def get_accumulated_input_latency(self, input_latency: int) -> int:
    return self._body.get_accumulated_input_latency(input_latency)

  def get_accumulated_output_latency(self, output_latency: int) -> int:
    return self._body.get_accumulated_output_latency(output_latency)

  @property
  def block_size(self) -> int:
    return int(np.lcm(self._body.block_size, self._shortcut.block_size))

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self._body.output_ratio

  @functools.cached_property
  def receptive_field(self) -> types.ReceptiveField:
    return utils.reduce_receptive_field_per_step(
        self.receptive_field_per_step, self.output_ratio
    )

  @functools.cached_property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    body_rf = self._body.receptive_field_per_step
    shortcut_rf = self._shortcut.receptive_field_per_step
    body_ratio = self._body.output_ratio
    shortcut_ratio = self._shortcut.output_ratio
    steps = range(math.lcm(len(body_rf), len(shortcut_rf)))
    rf_per_step = {}
    for step in steps:
      rf_per_step[step] = utils.receptive_field_union(
          utils.receptive_field_at(body_rf, body_ratio, step),
          utils.receptive_field_at(shortcut_rf, shortcut_ratio, step),
      )
    return rf_per_step

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    body_state = self._body.get_initial_state(
        batch_size, input_spec, constants=constants
    )
    shortcut_state = self._shortcut.get_initial_state(
        batch_size, input_spec, constants=constants
    )
    return body_state, shortcut_state

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    body_dtype = self._body.get_output_dtype(
        input_dtype, constants=constants
    )
    shortcut_dtype = self._shortcut.get_output_dtype(
        input_dtype, constants=constants
    )
    return jnp.result_type(body_dtype, shortcut_dtype)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    body_shape = self._body.get_output_shape(
        input_shape, constants=constants
    )
    shortcut_shape = self._shortcut.get_output_shape(
        input_shape, constants=constants
    )
    if body_shape != shortcut_shape:
      raise ValueError(
          f'Residual body and shortcut must have same output shape:'
          f' body={body_shape} shortcut={shortcut_shape}'
      )
    return body_shape

  def residual_function(
      self, y_body: types.Sequence, y_shortcut: types.Sequence
  ) -> types.Sequence:
    y_values = y_body.values + y_shortcut.values
    y_mask = y_body.mask & y_shortcut.mask
    return types.Sequence(y_values, y_mask)

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    body_state, shortcut_state = state
    y_body, body_state, body_emits = self._body.step_with_emits(
        x, body_state, constants=constants
    )
    y_shortcut, shortcut_state, shortcut_emits = (
        self._shortcut.step_with_emits(
            x, shortcut_state, constants=constants
        )
    )
    y = self.residual_function(y_body, y_shortcut)
    return y, (body_state, shortcut_state), (body_emits, shortcut_emits)

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    y_body, body_emits = self._body.layer_with_emits(
        x, constants=constants
    )
    y_shortcut, shortcut_emits = self._shortcut.layer_with_emits(
        x, constants=constants
    )
    y = self.residual_function(y_body, y_shortcut)
    return y, (body_emits, shortcut_emits)


CombinationMode = utils.CombinationMode


class Parallel(types.Emitting):
  """Applies a sequence of layers in parallel and combines outputs."""

  def __init__(
      self,
      layers: list[types.SequenceLayer],
      *,
      combination: utils.CombinationMode = utils.CombinationMode.STACK,
  ):
    super().__init__()
    self.layers = nnx.List(layers)
    self.combination = combination

    if not self.layers:
      self._output_ratio = fractions.Fraction(1)
      return

    first, *rest = self.layers
    self._output_ratio = first.output_ratio
    for child_layer in rest:
      if child_layer.output_ratio != self._output_ratio:
        raise ValueError(
            'Output ratios must be equal for all layers:'
            f' {self._output_ratio} != {child_layer.output_ratio}'
            f' for {child_layer}'
        )

    input_latency = first.input_latency
    for child_layer in rest:
      if child_layer.input_latency != input_latency:
        raise ValueError(
            'Parallel layers must have the same input latency.'
            f' {child_layer.name} has'
            f' input_latency={child_layer.input_latency} !='
            f' {input_latency}.'
        )

  @property
  def supports_step(self) -> bool:
    return all(l.supports_step for l in self.layers)

  @property
  def input_latency(self) -> int:
    return self.get_accumulated_input_latency(0)

  @property
  def output_latency(self) -> int:
    return self.get_accumulated_output_latency(0)

  @functools.cached_property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    if not self.layers:
      return {0: (0, 0)}
    return utils.aggregate_layers_receptive_field_per_steps(
        self.layers
    )

  def get_accumulated_input_latency(self, input_latency: int) -> int:
    if not self.layers:
      return input_latency
    child_latencies = [
        l.get_accumulated_input_latency(input_latency)
        for l in self.layers
    ]
    if len(set(child_latencies)) > 1:
      raise ValueError(
          'Parallel layers must have the same accumulated input'
          f' latency. Got: {child_latencies}.'
      )
    return child_latencies[0]

  def get_accumulated_output_latency(self, output_latency: int) -> int:
    if not self.layers:
      return output_latency
    child_latencies = [
        l.get_accumulated_output_latency(output_latency)
        for l in self.layers
    ]
    if len(set(child_latencies)) > 1:
      raise ValueError(
          'Parallel layers must have the same accumulated output'
          f' latency. Got: {child_latencies}.'
      )
    return child_latencies[0]

  @property
  def block_size(self) -> int:
    block_size = 1
    for child_layer in self.layers:
      block_size = np.lcm(block_size, child_layer.block_size)
    return int(block_size)

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self._output_ratio

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    states = []
    for child_layer in self.layers:
      states.append(
          child_layer.get_initial_state(
              batch_size, input_spec, constants=constants
          )
      )
    return tuple(states)

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if not self.layers:
      return tuple(input_shape)
    output_shapes = [
        l.get_output_shape(input_shape, constants=constants)
        for l in self.layers
    ]
    return utils.sequence_broadcast_combine_output_channel_shape(
        self.combination, *output_shapes
    )

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    if not self.layers:
      return input_dtype
    dtype = self.layers[0].get_output_dtype(
        input_dtype, constants=constants
    )
    for child_layer in self.layers[1:]:
      dtype = jnp.result_type(
          dtype,
          child_layer.get_output_dtype(
              input_dtype, constants=constants
          ),
      )
    return dtype

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    new_state = []
    emits = {}
    if len(self.layers) != len(state):
      raise ValueError(
          f'{type(self).__name__} received unexpected state structure:'
          f' {len(self.layers)=} != {len(state)=}'
      )
    if not self.layers:
      return x, state, emits

    ys = []
    for child_layer, layer_state in zip(self.layers, state):
      y, layer_state, layer_emits = child_layer.step_with_emits(
          x, layer_state, constants=constants
      )
      ys.append(y)
      new_state.append(layer_state)
      emits[child_layer.name] = layer_emits
    y = utils.sequence_broadcast_combine(self.combination, *ys)
    return y, tuple(new_state), emits

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    emits = {}
    if not self.layers:
      return x, emits
    ys = []
    for child_layer in self.layers:
      y, layer_emits = child_layer.layer_with_emits(
          x, constants=constants
      )
      ys.append(y)
      emits[child_layer.name] = layer_emits
    y = utils.sequence_broadcast_combine(self.combination, *ys)
    return y, emits


class Bidirectional(types.Emitting):
  """Processes a sequence with forward and backward layers.

  The forward layer processes the sequence unmodified. The backward layer
  processes the time-reversed sequence. Outputs are combined according to
  the specified combination mode.

  Does not support step-wise processing.
  """

  def __init__(
      self,
      *,
      forward: types.SequenceLayer,
      backward: types.SequenceLayer,
      combination: utils.CombinationMode = utils.CombinationMode.STACK,
  ):
    super().__init__()
    self.forward = forward
    self.backward = backward
    self.combination = combination

    if self.forward.output_ratio != self.backward.output_ratio:
      raise ValueError(
          'Output ratios for forward and backward must be equal.'
          f' forward={self.forward.output_ratio}'
          f' backward={self.backward.output_ratio}.'
      )

  @property
  def supports_step(self) -> bool:
    return False

  @property
  def block_size(self) -> int:
    return 1

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self.forward.output_ratio

  @functools.cached_property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    rf_fwd = self.forward.receptive_field_per_step
    rf_bwd = self.backward.receptive_field_per_step
    types.validate_receptive_field_per_step(rf_fwd)
    types.validate_receptive_field_per_step(rf_bwd)
    num_steps = math.lcm(len(rf_fwd), len(rf_bwd))
    rf_per_step = {}
    for step in range(num_steps):
      fwd_step = step % len(rf_fwd)
      bwd_step = (num_steps - step) % len(rf_bwd)
      rf_f = rf_fwd.get(fwd_step)
      rf_b = rf_bwd.get(bwd_step)
      rf_b = (
          tuple(-r for r in reversed(rf_b))
          if rf_b is not None
          else None
      )
      if rf_f is None and rf_b is None:
        rf_per_step[step] = None
      elif rf_f is None:
        rf_per_step[step] = rf_b
      elif rf_b is None:
        rf_per_step[step] = rf_f
      else:
        rf_per_step[step] = (
            min(rf_f[0], rf_b[0]),
            max(rf_f[1], rf_b[1]),
        )
    return rf_per_step

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    fwd_shape = self.forward.get_output_shape(
        input_shape, constants=constants
    )
    bwd_shape = self.backward.get_output_shape(
        input_shape, constants=constants
    )
    return utils.sequence_broadcast_combine_output_channel_shape(
        self.combination, fwd_shape, bwd_shape
    )

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    fwd_dtype = self.forward.get_output_dtype(
        input_dtype, constants=constants
    )
    bwd_dtype = self.backward.get_output_dtype(
        input_dtype, constants=constants
    )
    return jnp.result_type(fwd_dtype, bwd_dtype)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    fwd_state = self.forward.get_initial_state(
        batch_size, input_spec, constants=constants
    )
    bwd_state = self.backward.get_initial_state(
        batch_size, input_spec, constants=constants
    )
    return (fwd_state, bwd_state)

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    raise ValueError(
        'Bidirectional does not support step-wise processing.'
    )

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    y_forward, fwd_emits = self.forward.layer_with_emits(
        x, constants=constants
    )
    x_reverse = x.reverse_time()
    y_backward, bwd_emits = self.backward.layer_with_emits(
        x_reverse, constants=constants
    )
    y_backward = y_backward.reverse_time()
    y = utils.sequence_broadcast_combine(
        self.combination, y_forward, y_backward
    )
    return y, (fwd_emits, bwd_emits)


class Repeat(types.Emitting):
  """A combinator that repeats a SequenceLayer N times using nnx.scan.

  Parameters are stacked with a leading ``num_repeats`` dimension (created
  via ``nnx.vmap`` at construction time). Execution uses ``nnx.scan`` so
  the layer logic is compiled only once.

  Requires that the child layer has ``output_ratio == 1`` and that the
  output dtype and shape equal the input dtype and shape, since the
  input and output to ``jax.lax.scan`` must match.
  """

  def __init__(
      self,
      layer_fn: Callable[[nnx.Rngs], types.SequenceLayer],
      *,
      num_repeats: int,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    if num_repeats <= 0:
      raise ValueError(
          f'Expected num_repeats > 0, got {num_repeats}.'
      )
    self.num_repeats = num_repeats

    # Create num_repeats copies with stacked params via vmap.
    @nnx.split_rngs(splits=num_repeats)
    @nnx.vmap(in_axes=(0,), out_axes=0)
    def create_layers(rngs: nnx.Rngs):
      return layer_fn(rngs)

    self.child_layer = create_layers(rngs)

    # Validate constraints.
    child_output_ratio = self._get_child_property(
        lambda l: l.output_ratio
    )
    if child_output_ratio != 1:
      raise ValueError(
          'Repeat child layer must have output_ratio == 1, got'
          f' {child_output_ratio}.'
      )

  def _get_child_property(self, property_fn):
    """Read a property from the child layer.

    Since the child layer was created with vmap, its parameters have a
    leading num_repeats dimension. For scalar properties (block_size,
    output_ratio, etc.) the value is the same for every repeat, so we
    just need to read it once. We use nnx.scan to enter the proper
    scope so the child layer sees correctly-shaped parameters.
    """
    results = []

    @nnx.scan(in_axes=(0,), out_axes=0, length=self.num_repeats)
    def read_property(layer):
      results.append(property_fn(layer))
      return layer  # dummy output, required by scan

    read_property(self.child_layer)
    return results[0]

  @property
  def supports_step(self) -> bool:
    return self._get_child_property(lambda l: l.supports_step)

  @property
  def input_latency(self) -> int:
    return self.get_accumulated_input_latency(0)

  @property
  def output_latency(self) -> int:
    return self.get_accumulated_output_latency(0)

  def get_accumulated_input_latency(self, input_latency: int) -> int:
    child_fn = lambda l: l.get_accumulated_input_latency(input_latency)
    child_latency = self._get_child_property(child_fn)
    # Each repeat accumulates on top of the previous.
    result = input_latency
    for _ in range(self.num_repeats):
      result = child_fn(
          # Use a dummy to get the per-layer contribution.
          type('_', (), {
              'get_accumulated_input_latency': lambda self, il: child_latency
          })()
      ) if False else result  # noqa
    # Actually: each child has the same latency contribution.
    # input_latency_per_child = child_latency - input_latency
    # total = input_latency + num_repeats * per_child
    per_child = child_latency - input_latency
    return input_latency + self.num_repeats * per_child

  def get_accumulated_output_latency(self, output_latency: int) -> int:
    child_fn = lambda l: l.get_accumulated_output_latency(output_latency)
    child_latency = self._get_child_property(child_fn)
    per_child = child_latency - output_latency
    return output_latency + self.num_repeats * per_child

  @property
  def block_size(self) -> int:
    child_block_size = self._get_child_property(
        lambda l: l.block_size
    )
    child_output_ratio = self._get_child_property(
        lambda l: l.output_ratio
    )
    block_size = fractions.Fraction(1)
    output_ratio = fractions.Fraction(1)
    for _ in range(self.num_repeats):
      block_size = (
          np.lcm(block_size * output_ratio, child_block_size)
          / output_ratio
      )
      output_ratio *= child_output_ratio
    assert block_size.denominator == 1
    return block_size.numerator

  @property
  def output_ratio(self) -> fractions.Fraction:
    child_output_ratio = self._get_child_property(
        lambda l: l.output_ratio
    )
    return child_output_ratio ** self.num_repeats

  @functools.cached_property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    child_rf = self._get_child_property(
        lambda l: l.receptive_field_per_step
    )
    child_output_ratio = self._get_child_property(
        lambda l: l.output_ratio
    )
    overall_rf = child_rf
    for _ in range(self.num_repeats - 1):
      overall_rf = utils.propagate_receptive_field_to_prev_layer(
          overall_rf, child_rf, child_output_ratio
      )
    return overall_rf

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return input_dtype

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return tuple(input_shape)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    @nnx.scan(in_axes=(0,), out_axes=0)
    def get_states(layer):
      return layer.get_initial_state(
          batch_size, input_spec, constants=constants
      )

    return get_states(self.child_layer)

  def _layer_internal(
      self,
      x: types.Sequence,
      constants: types.Constants | None,
      with_emits: bool,
  ) -> tuple[types.Sequence, types.Emits]:
    @nnx.scan(
        in_axes=(nnx.Carry, 0),
        out_axes=(nnx.Carry, 0) if with_emits else nnx.Carry,
    )
    def scan_fn(x_tuple, layer):
      seq = types.Sequence(x_tuple[0], x_tuple[1])
      if with_emits:
        y, emits = layer.layer_with_emits(seq, constants=constants)
      else:
        y = layer.layer(seq, constants=constants)
        emits = ()
      carry_out = (y.unmask().values, y.mask)
      if with_emits:
        return carry_out, emits
      return carry_out

    if with_emits:
      result_tuple, emits = scan_fn(
          (x.unmask().values, x.mask), self.child_layer
      )
      emits = utils.unstack_tree(emits)
    else:
      result_tuple = scan_fn(
          (x.unmask().values, x.mask), self.child_layer
      )
      emits = ()

    y = types.Sequence(result_tuple[0], result_tuple[1])
    return y, emits

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    return self._layer_internal(x, constants, with_emits=True)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    y, _ = self._layer_internal(x, constants, with_emits=False)
    return y

  def _step_internal(
      self,
      x: types.Sequence,
      state: types.State,
      constants: types.Constants | None,
      with_emits: bool,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    @nnx.scan(
        in_axes=(nnx.Carry, 0, 0),
        out_axes=(
            (nnx.Carry, 0, 0) if with_emits else (nnx.Carry, 0)
        ),
    )
    def scan_fn(x_tuple, layer, state):
      seq = types.Sequence(x_tuple[0], x_tuple[1])
      if with_emits:
        y, new_state, emits = layer.step_with_emits(
            seq, state, constants=constants
        )
      else:
        y, new_state = layer.step(
            seq, state, constants=constants
        )
        emits = ()
      carry_out = (y.unmask().values, y.mask)
      if with_emits:
        return carry_out, new_state, emits
      return carry_out, new_state

    if with_emits:
      result_tuple, new_state, emits = scan_fn(
          (x.unmask().values, x.mask), self.child_layer, state
      )
      emits = utils.unstack_tree(emits)
    else:
      result_tuple, new_state = scan_fn(
          (x.unmask().values, x.mask), self.child_layer, state
      )
      emits = ()

    y = types.Sequence(result_tuple[0], result_tuple[1])
    return y, new_state, emits

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    y, state, _ = self._step_internal(
        x, state, constants, with_emits=False
    )
    return y, state

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    return self._step_internal(
        x, state, constants, with_emits=True
    )


class Blockwise(types.SequenceLayer):
  """Processes the provided layer in blocks of a given size."""

  def __init__(
      self,
      *,
      child_layer: types.SequenceLayer,
      block_size: int,
  ):
    super().__init__()
    self.child_layer = child_layer
    self._block_size = block_size

    if block_size % child_layer.block_size != 0:
      raise ValueError(
          'Block size must be a multiple of the child layer block'
          f' size. {block_size=} % {child_layer.block_size=} =='
          f' {block_size % child_layer.block_size}'
      )

  @property
  def supports_step(self) -> bool:
    return self.child_layer.supports_step

  @property
  def input_latency(self) -> int:
    return self.child_layer.input_latency

  @property
  def output_latency(self) -> int:
    return self.child_layer.output_latency

  @property
  def block_size(self) -> int:
    return self._block_size

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self.child_layer.output_ratio

  @functools.cached_property
  def receptive_field_per_step(
      self,
  ) -> dict[int, types.ReceptiveField]:
    return self.child_layer.receptive_field_per_step

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return self.child_layer.get_output_shape(
        input_shape, constants=constants
    )

  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return self.child_layer.get_output_dtype(
        input_dtype, constants=constants
    )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      constants: types.Constants | None = None,
  ) -> types.State:
    return self.child_layer.get_initial_state(
        batch_size, input_spec, constants=constants
    )

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    return self.child_layer.step(x, state, constants=constants)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return self.child_layer.layer(x, constants=constants)
