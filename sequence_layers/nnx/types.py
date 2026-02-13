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
"""NNX base types for sequence layers."""

import abc
import functools

from flax import nnx
from sequence_layers.jax.types import ChannelSpec
from sequence_layers.jax.types import Constants
from sequence_layers.jax.types import DType
from sequence_layers.jax.types import Emits
from sequence_layers.jax.types import MaskedSequence
from sequence_layers.jax.types import MASK_DTYPE
from sequence_layers.jax.types import MaskT
from sequence_layers.jax.types import PaddingMode
from sequence_layers.jax.types import ReceptiveField
from sequence_layers.jax.types import Sequence
from sequence_layers.jax.types import Shape
from sequence_layers.jax.types import ShapeDType
from sequence_layers.jax.types import ShapeLike
from sequence_layers.jax.types import Sharding
from sequence_layers.jax.types import State
from sequence_layers.jax.types import validate_receptive_field_per_step
from sequence_layers.jax.types import ValuesT


# Re-export for convenience.
__all__ = (
    'ChannelSpec',
    'Constants',
    'DType',
    'Emits',
    'Emitting',
    'MASK_DTYPE',
    'MaskedSequence',
    'MaskT',
    'PaddingMode',
    'PreservesShape',
    'PreservesType',
    'ReceptiveField',
    'Sequence',
    'SequenceLayer',
    'Shape',
    'ShapeDType',
    'ShapeLike',
    'Sharding',
    'State',
    'Stateless',
    'StatelessEmitting',
    'StatelessPointwise',
    'StatelessPointwiseFunctor',
    'Steppable',
    'ValuesT',
    'check_layer',
    'check_step',
)


class Steppable(metaclass=abc.ABCMeta):
  """A sequence processing layer that can be executed layerwise or stepwise.

  Same contract as the Linen version, but method signatures drop the
  ``training`` parameter — use ``self.deterministic`` instead.
  """

  @property
  def name(self) -> str:
    """Returns the module name. Provided by nnx.Module."""
    return type(self).__name__

  @property
  def block_size(self) -> int:
    return 1

  @property
  def output_ratio(self):
    import fractions
    return fractions.Fraction(1)

  @property
  def supports_step(self) -> bool:
    return True

  @property
  def input_latency(self) -> int:
    return 0

  @property
  def output_latency(self) -> int:
    return int(self.input_latency * self.output_ratio)

  def get_accumulated_input_latency(self, input_latency: int) -> int:
    import math
    return math.ceil(input_latency / self.output_ratio) + self.input_latency

  def get_accumulated_output_latency(self, output_latency: int) -> int:
    output_ratio = self.output_ratio
    return int(output_latency * output_ratio) + self.output_latency

  @property
  def receptive_field(self) -> ReceptiveField:
    import numpy as np
    rf_per_step = self.receptive_field_per_step
    validate_receptive_field_per_step(rf_per_step)
    rf_list = {s: rf for s, rf in rf_per_step.items() if rf is not None}
    if not rf_list:
      return None
    min_start = np.inf
    max_end = -np.inf
    output_ratio = self.output_ratio
    for output_step, (start, end) in rf_list.items():
      input_step = output_step // output_ratio
      start -= input_step
      end -= input_step
      min_start = min(min_start, start)
      max_end = max(max_end, end)
    return min_start, max_end

  @property
  def receptive_field_per_step(self) -> dict[int, ReceptiveField]:
    layer_name = type(self).__name__
    raise NotImplementedError(
        f'receptive_field_per_step is not implemented by {layer_name}'
    )

  @abc.abstractmethod
  def layer(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> Sequence:
    """Process this layer layer-wise."""

  def layer_with_emits(
      self,
      x: Sequence,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    outputs = self.layer(x, constants=constants)
    return outputs, ()

  def __call__(
      self, x: Sequence, constants: Constants | None = None
  ) -> Sequence:
    return self.layer(x, constants=constants)

  @abc.abstractmethod
  def step(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    """Process this layer step-wise."""

  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    outputs, state = self.step(x, state, constants=constants)
    return outputs, state, ()

  @abc.abstractmethod
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
  ) -> State:
    """Returns the initial state for this SequenceLayer."""

  @abc.abstractmethod
  def get_output_shape(
      self, input_shape: ShapeLike, *, constants: Constants | None = None
  ) -> Shape:
    """Returns the output shape for an input shape (channel dims only)."""

  def get_output_shape_for_sequence(
      self,
      x: Sequence,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    return self.get_output_shape(x.channel_shape, constants=constants)

  @abc.abstractmethod
  def get_output_dtype(
      self, input_dtype: DType, *, constants: Constants | None = None
  ) -> DType:
    """Returns the layer's output dtype for the specified input dtype."""

  def get_output_spec(
      self,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
  ) -> ChannelSpec:
    shape = self.get_output_shape(input_spec.shape, constants=constants)
    dtype = self.get_output_dtype(input_spec.dtype, constants=constants)
    return ChannelSpec(shape, dtype)

  def get_output_spec_for_sequence(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> ChannelSpec:
    return self.get_output_spec(x.channel_spec, constants=constants)


# ---------------------------------------------------------------------------
# Validation decorators (no `training` param)
# ---------------------------------------------------------------------------

def _check_step_common(layer: Steppable, x: Sequence) -> None:
  if not layer.supports_step:
    raise ValueError(f'{layer.__class__.__name__} does not support step().')
  block_size = layer.block_size
  if x.shape[1] % block_size != 0:
    raise ValueError(
        f'{layer.__class__.__name__} received input with shape'
        f' {x.shape=} which is not a multiple of {block_size=}.'
    )


def _check_output_spec(
    layer: Steppable, x: Sequence, y: Sequence, constants: Constants | None
):
  expected_output_spec = layer.get_output_spec(
      x.channel_spec, constants=constants
  )
  if y.channel_shape != expected_output_spec.shape:
    raise ValueError(
        f'{layer.__class__.__name__} produced output ({y.channel_spec}) for'
        f' input ({x.channel_spec}), whose shape does not match'
        f' get_output_spec ({expected_output_spec}).'
    )


def _check_output_ratio(layer: Steppable, x: Sequence, y: Sequence):
  output_ratio = layer.output_ratio
  expected_output_length = x.shape[1] * output_ratio
  if y.shape[1] != expected_output_length:
    raise ValueError(
        f'{layer.__class__.__name__} produced output ({y.shape}) for input'
        f' ({x.shape}), whose length does not equal'
        f' {expected_output_length} ({output_ratio=}).'
    )


def check_layer(layer_fn):
  """Validates layer inputs and outputs."""

  @functools.wraps(layer_fn)
  def check_layer_fn(
      self,
      x: Sequence,
      *,
      constants: Constants | None = None,
  ) -> Sequence:
    y = layer_fn(self, x, constants=constants)
    _check_output_spec(self, x, y, constants)
    return y

  return check_layer_fn


def check_step(step_fn):
  """Validates step inputs and outputs."""

  @functools.wraps(step_fn)
  def check_step_fn(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    _check_step_common(self, x)
    y, state = step_fn(self, x, state, constants=constants)
    _check_output_spec(self, x, y, constants)
    _check_output_ratio(self, x, y)
    return y, state

  return check_step_fn


def check_layer_with_emits(layer_with_emits_fn):
  """Validates layer_with_emits inputs and outputs."""

  @functools.wraps(layer_with_emits_fn)
  def check_layer_with_emits_fn(
      self,
      x: Sequence,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    y, emits = layer_with_emits_fn(self, x, constants=constants)
    _check_output_spec(self, x, y, constants)
    return y, emits

  return check_layer_with_emits_fn


def check_step_with_emits(step_with_emits_fn):
  """Validates step_with_emits inputs and outputs."""

  @functools.wraps(step_with_emits_fn)
  def check_step_with_emits_fn(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    _check_step_common(self, x)
    y, state, emits = step_with_emits_fn(
        self, x, state, constants=constants
    )
    _check_output_spec(self, x, y, constants)
    _check_output_ratio(self, x, y)
    return y, state, emits

  return check_step_with_emits_fn


# ---------------------------------------------------------------------------
# SequenceLayer — the NNX base module
# ---------------------------------------------------------------------------

class SequenceLayer(nnx.Module, Steppable):
  """Base NNX Module for Sequence Layers.

  Uses ``self.deterministic`` instead of a ``training`` parameter.
  Call ``model.train()`` to set ``deterministic=False`` and
  ``model.eval()`` to set ``deterministic=True`` on all submodules.
  """

  def __init__(self):
    self.deterministic: bool = True  # eval mode by default


# ---------------------------------------------------------------------------
# Mix-ins
# ---------------------------------------------------------------------------

class PreservesType:
  """Mix-in for layers that do not change the input dtype."""

  def get_output_dtype(
      self, input_dtype: DType, *, constants: Constants | None = None
  ) -> DType:
    del constants
    return input_dtype


class PreservesShape:
  """Mix-in for layers that do not change the input shape."""

  def get_output_shape(
      self, input_shape: ShapeLike, *, constants: Constants | None = None
  ) -> Shape:
    del constants
    return tuple(input_shape)


# ---------------------------------------------------------------------------
# Abstract convenience base classes
# ---------------------------------------------------------------------------

class Stateless(SequenceLayer):
  """A SequenceLayer with no state required for step-wise processing."""

  @property
  def receptive_field_per_step(self) -> dict[int, ReceptiveField]:
    return {0: (0, 0)}

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
  ) -> State:
    return ()

  def step(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    return self.layer(x, constants=constants), state


class Emitting(SequenceLayer, metaclass=abc.ABCMeta):
  """A SequenceLayer that emits auxiliary tensors."""

  def step(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    output, state, _ = self.step_with_emits(x, state, constants=constants)
    return output, state

  @abc.abstractmethod
  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    pass

  def layer(
      self,
      x: Sequence,
      *,
      constants: Constants | None = None,
  ) -> Sequence:
    outputs, _ = self.layer_with_emits(x, constants=constants)
    return outputs

  @abc.abstractmethod
  def layer_with_emits(
      self,
      x: Sequence,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    pass


class StatelessEmitting(Emitting):
  """Stateless + emitting."""

  @property
  def receptive_field_per_step(self) -> dict[int, ReceptiveField]:
    return {0: (0, 0)}

  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    outputs, emits = self.layer_with_emits(x, constants=constants)
    return outputs, state, emits

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
  ) -> State:
    return ()


class StatelessPointwise(PreservesShape, Stateless):
  """A SequenceLayer that has no state and operates pointwise."""


class StatelessPointwiseFunctor(StatelessPointwise, metaclass=abc.ABCMeta):
  """A stateless SequenceLayer for simple pointwise processing fns."""

  @abc.abstractmethod
  def fn(self, values: ValuesT, mask: MaskT) -> tuple[ValuesT, MaskT]:
    """Transforms each scalar in values independently."""

  @property
  def mask_required(self):
    return True

  @check_layer
  def layer(
      self,
      x: Sequence,
      *,
      constants: Constants | None = None,
  ) -> Sequence:
    if self.mask_required:
      y = x.apply(self.fn)
    else:
      y = x.apply_masked(self.fn)
    # In Linen, the apply/bind mechanism converts MaskedSequence outputs
    # to Sequence. In NNX (no bind/apply), apply_masked preserves the
    # input type. We explicitly return Sequence so that downstream
    # mask_invalid() calls properly zero invalid values.
    if isinstance(y, MaskedSequence):
      y = Sequence(y.values, y.mask)
    return y
