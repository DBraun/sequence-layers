"""SequenceLayer type hierarchy for MLX."""

import abc
import fractions
import functools
import math
from typing import Callable

import mlx.nn as nn

from sequence_layers.mlx import basic_types as bt

Sequence = bt.Sequence
MaskedSequence = bt.MaskedSequence
ChannelSpec = bt.ChannelSpec
ShapeDType = bt.ShapeDType
Shape = bt.Shape
ShapeLike = bt.ShapeLike
DType = bt.DType
State = bt.State
Constants = bt.Constants
Emits = bt.Emits
ReceptiveField = bt.ReceptiveField
ValuesT = bt.ValuesT
MaskT = bt.MaskT


# ---------------------------------------------------------------------------
# Check decorators
# ---------------------------------------------------------------------------


def _check_output_spec(layer, x, y, constants):
  expected = layer.get_output_spec(x.channel_spec, constants=constants)
  if y.channel_shape != expected.shape:
    raise ValueError(
        f'{layer.__class__.__name__} produced output'
        f' ({y.channel_spec}) for input ({x.channel_spec}),'
        ' whose shape does not match get_output_spec'
        f' ({expected}).'
    )


def _check_output_ratio(layer, x, y):
  expected_length = x.shape[1] * layer.output_ratio
  if y.shape[1] != expected_length:
    raise ValueError(
        f'{layer.__class__.__name__} produced output ({y.shape})'
        f' for input ({x.shape}), whose length does not equal'
        f' {expected_length} (output_ratio={layer.output_ratio}).'
    )


def check_layer(layer_fn):
  """Validates layer inputs and outputs."""

  @functools.wraps(layer_fn)
  def wrapper(self, x, *, constants=None, **kwargs):
    y = layer_fn(self, x, constants=constants)
    _check_output_spec(self, x, y, constants)
    return y

  return wrapper


def check_step(step_fn):
  """Validates step inputs and outputs."""

  @functools.wraps(step_fn)
  def wrapper(self, x, state, *, constants=None, **kwargs):
    if not self.supports_step:
      raise ValueError(f'{self.__class__.__name__} does not support step().')
    block_size = self.block_size
    if x.shape[1] % block_size != 0:
      raise ValueError(
          f'{self.__class__.__name__} received input with'
          f' {x.shape=} not a multiple of {block_size=}.'
      )
    y, state = step_fn(self, x, state, constants=constants)
    _check_output_spec(self, x, y, constants)
    _check_output_ratio(self, x, y)
    return y, state

  return wrapper


# ---------------------------------------------------------------------------
# Steppable ABC
# ---------------------------------------------------------------------------


class Steppable(metaclass=abc.ABCMeta):
  """A sequence processing layer that supports layer and step modes."""

  @property
  def block_size(self) -> int:
    return 1

  @property
  def output_ratio(self) -> fractions.Fraction:
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
    """Returns the accumulated input latency of this layer."""
    return math.ceil(input_latency / self.output_ratio) + self.input_latency

  @abc.abstractmethod
  def layer(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> Sequence:
    """Process this layer layer-wise."""

  def layer_with_emits(
      self, x: Sequence, *, constants: Constants | None = None, **kwargs
  ) -> tuple[Sequence, Emits]:
    return self.layer(x, constants=constants), ()

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
      **kwargs,
  ) -> tuple[Sequence, State, Emits]:
    y, state = self.step(x, state, constants=constants)
    return y, state, ()

  @abc.abstractmethod
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
  ) -> State:
    """Returns the initial state for step-wise processing."""

  @abc.abstractmethod
  def get_output_shape(
      self,
      input_shape: ShapeLike,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    """Returns the output channel shape for an input channel shape."""

  @abc.abstractmethod
  def get_output_dtype(
      self,
      input_dtype: DType,
      *,
      constants: Constants | None = None,
  ) -> DType:
    """Returns the output dtype for an input dtype."""

  def get_output_spec(
      self,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
  ) -> ChannelSpec:
    shape = self.get_output_shape(input_spec.shape, constants=constants)
    dtype = self.get_output_dtype(input_spec.dtype, constants=constants)
    return ChannelSpec(shape, dtype)


# ---------------------------------------------------------------------------
# SequenceLayer — MLX base
# ---------------------------------------------------------------------------


class SequenceLayer(nn.Module, Steppable):
  """Base MLX Module for Sequence Layers."""


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------


class PreservesType:
  """Mix-in: layer does not change the input dtype."""

  def get_output_dtype(
      self, input_dtype: DType, *, constants: Constants | None = None
  ) -> DType:
    del constants
    return input_dtype


class PreservesShape:
  """Mix-in: layer does not change the input channel shape."""

  def get_output_shape(
      self,
      input_shape: ShapeLike,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    del constants
    return tuple(input_shape)


# ---------------------------------------------------------------------------
# Stateless variants
# ---------------------------------------------------------------------------


class Stateless(SequenceLayer):
  """A SequenceLayer with no step state."""

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
      **kwargs,
  ) -> State:
    return ()

  def step(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
      **kwargs,
  ) -> tuple[Sequence, State]:
    return self.layer(x, constants=constants), state


class StatelessPointwise(PreservesShape, Stateless):
  """Stateless layer that operates pointwise (preserves shape)."""


class StatelessPointwiseFunctor(StatelessPointwise, metaclass=abc.ABCMeta):
  """Stateless pointwise layer defined by a fn(values, mask)."""

  @abc.abstractmethod
  def fn(self, values: ValuesT, mask: MaskT) -> tuple[ValuesT, MaskT]:
    """Transforms each scalar in values independently."""

  @property
  def mask_required(self):
    return True

  @check_layer
  def layer(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> Sequence:
    if self.mask_required:
      y = x.apply(self.fn)
    else:
      y = x.apply_masked(self.fn)
    # Ensure MaskedSequence -> Sequence conversion for apply.
    if isinstance(y, MaskedSequence) and self.mask_required:
      y = Sequence(y.values, y.mask)
    return y


# ---------------------------------------------------------------------------
# Emitting variants
# ---------------------------------------------------------------------------


class Emitting(SequenceLayer, metaclass=abc.ABCMeta):
  """A SequenceLayer that emits auxiliary tensors."""

  def step(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
      **kwargs,
  ) -> tuple[Sequence, State]:
    y, state, _ = self.step_with_emits(x, state, constants=constants)
    return y, state

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
      self, x: Sequence, *, constants: Constants | None = None, **kwargs
  ) -> Sequence:
    y, _ = self.layer_with_emits(x, constants=constants)
    return y

  @abc.abstractmethod
  def layer_with_emits(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> tuple[Sequence, Emits]:
    pass


class StatelessEmitting(Emitting):
  """Stateless layer that emits auxiliary tensors."""

  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
      **kwargs,
  ) -> tuple[Sequence, State, Emits]:
    y, emits = self.layer_with_emits(x, constants=constants)
    return y, state, emits

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
      **kwargs,
  ) -> State:
    return ()
