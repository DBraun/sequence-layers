"""Basic sequence types for MLX."""

import enum
from typing import Callable, Generic, Iterable, TypeVar

import mlx.core as mx
import numpy as np


# Type aliases.
MASK_DTYPE = mx.bool_

ValuesT = TypeVar('ValuesT', bound=mx.array)
MaskT = TypeVar('MaskT', bound=mx.array)
LengthsT = TypeVar('LengthsT', bound=mx.array)
ExpandedMaskT = TypeVar('ExpandedMaskT', bound=mx.array)
SequenceSelf = TypeVar('SequenceSelf', bound='Sequence')

Shape = tuple[int, ...]
ShapeLike = list[int] | tuple[int, ...]
DType = np.dtype
State = object  # Any pytree.
Constants = dict[str, object]
Emits = object

# Receptive field.
ReceptiveField = tuple[float | int, float | int] | None


class ShapeDType:
  """Lightweight replacement for jax.ShapeDtypeStruct."""

  def __init__(self, shape: Shape, dtype: DType):
    self.shape = shape
    self.dtype = dtype

  def __repr__(self) -> str:
    return f'ShapeDType(shape={self.shape}, dtype={self.dtype})'

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, ShapeDType):
      return NotImplemented
    return self.shape == other.shape and self.dtype == other.dtype

  def __hash__(self) -> int:
    return hash((self.shape, self.dtype))


ChannelSpec = ShapeDType


class PaddingMode(enum.Enum):
  VALID = 'valid'
  SAME = 'same'
  CAUSAL_VALID = 'causal_valid'
  REVERSE_CAUSAL_VALID = 'reverse_causal_valid'
  CAUSAL = 'causal'
  REVERSE_CAUSAL = 'reverse_causal'
  SEMICAUSAL = 'semicausal'
  SEMICAUSAL_FULL = 'semicausal_full'


def sequence_mask(lengths: LengthsT, maxlen: int) -> MaskT:
  return mx.arange(maxlen)[None, :] < mx.array(lengths)[:, None]


class Sequence(Generic[ValuesT, MaskT]):
  """A generic sequence container that preserves masking information."""

  values: ValuesT
  mask: MaskT

  def __init__(self, values: ValuesT, mask: MaskT):
    self.values = values
    self.mask = mask

  @property
  def shape(self) -> Shape:
    """Returns the shape of the sequence values."""
    return self.values.shape

  @property
  def ndim(self) -> int:
    """Returns the rank of the sequence values."""
    return self.values.ndim

  @property
  def channel_shape(self) -> Shape:
    """Returns the channel shape (the shape without batch and time)."""
    return self.values.shape[2:]

  @property
  def channel_spec(self) -> ChannelSpec:
    """Returns a "spec" for this sequence (the channel shape and dtype)."""
    return ChannelSpec(self.channel_shape, self.dtype)

  @property
  def dtype(self) -> DType:
    """Returns the dtype of the sequence values."""
    return self.values.dtype

  @classmethod
  def from_values(cls, values: ValuesT) -> 'MaskedSequence':
    """Returns a MaskedSequence assuming every timestep is valid."""
    if values.ndim < 2:
      raise ValueError(f'Expected {values.ndim=} to be at least 2.')
    return MaskedSequence(values, mx.ones(values.shape[:2], dtype=mx.bool_))

  @classmethod
  def concatenate_sequences(cls, sequences: Iterable['Sequence']) -> 'Sequence':
    """Concatenates sequences and their masks on the time axis."""
    values = []
    masks = []
    all_masked = True
    for sequence in sequences:
      if not isinstance(sequence, MaskedSequence):
        all_masked = False
      values.append(sequence.values)
      masks.append(sequence.mask)
    seq_type = MaskedSequence if all_masked else Sequence
    return seq_type(
        mx.concatenate(values, axis=1),
        mx.concatenate(masks, axis=1),
    )

  def expanded_mask(self) -> ExpandedMaskT:
    """Returns the Sequence mask expanded to match values rank."""
    return self.mask.reshape(self.mask.shape + (1,) * (self.values.ndim - 2))

  def apply_values(
      self,
      values_fn: Callable[..., ValuesT],
      *args,
      **kwargs,
  ) -> 'Sequence':
    """Transforms values, assuming result is unmasked."""
    return Sequence(values_fn(self.values, *args, **kwargs), self.mask)

  def apply_values_masked(
      self: SequenceSelf,
      values_fn: Callable[..., ValuesT],
      *args,
      **kwargs,
  ) -> SequenceSelf:
    """Transforms values, preserving masked state."""
    return type(self)(values_fn(self.values, *args, **kwargs), self.mask)

  def apply(
      self,
      apply_fn: Callable[..., tuple[ValuesT, MaskT]],
      *args,
      **kwargs,
  ) -> 'Sequence':
    """Transforms values/mask, assuming result is unmasked."""
    values, mask = apply_fn(self.values, self.mask, *args, **kwargs)
    return Sequence(values, mask)

  def apply_masked(
      self: SequenceSelf,
      apply_fn: Callable[..., tuple[ValuesT, MaskT]],
      *args,
      **kwargs,
  ) -> SequenceSelf:
    """Transforms values/mask, preserving masked state."""
    values, mask = apply_fn(self.values, self.mask, *args, **kwargs)
    return type(self)(values, mask)

  def astype(self: SequenceSelf, dtype: DType | None) -> SequenceSelf:
    """Returns a copy with values cast to dtype."""
    if dtype is None:
      return self
    return type(self)(self.values.astype(dtype), self.mask)

  def lengths(self) -> mx.array:
    """Returns the number of valid timesteps per batch item."""
    return mx.sum(self.mask.astype(mx.int32), axis=1)

  def __getitem__(
      self: SequenceSelf,
      the_slice,
  ) -> SequenceSelf:
    """Slices the Sequence values and mask."""
    if isinstance(the_slice, slice):
      the_slice = (the_slice,)
    return type(self)(
        self.values.__getitem__(the_slice),
        self.mask.__getitem__(the_slice[:2]),
    )

  def pad_time(
      self: SequenceSelf,
      pad_left: int,
      pad_right: int,
      valid: bool,
      pad_value: float | None = None,
  ) -> SequenceSelf:
    """Pads this sequence with timesteps on the left and right."""
    if not pad_left and not pad_right:
      return self
    pad_val = 0.0 if pad_value is None else pad_value
    values_rank = self.values.ndim
    values = mx.pad(
        self.values,
        [(0, 0), (pad_left, pad_right)] + [(0, 0)] * (values_rank - 2),
        constant_values=pad_val,
    )
    mask = mx.pad(
        self.mask,
        [(0, 0), (pad_left, pad_right)],
        constant_values=valid,
    )
    return type(self)(values, mask)

  def concatenate(self, other: 'Sequence') -> 'Sequence':
    """Concatenates with other on the time dimension."""
    values = mx.concatenate([self.values, other.values], axis=1)
    mask = mx.concatenate([self.mask, other.mask], axis=1)
    return_type = type(self) if type(self) is type(other) else Sequence
    return return_type(values, mask)

  def mask_invalid(self, mask_value: complex | None = None) -> 'Sequence':
    """Returns a sequence with invalid timesteps replaced."""
    raise NotImplementedError('Replaced below.')

  def unmask(self) -> 'Sequence':
    """Returns an unmasked version with unchanged values."""
    return self


class MaskedSequence(Sequence[ValuesT, MaskT]):
  """Sequence whose invalid timesteps are masked to zero."""

  def mask_invalid(self, mask_value: complex | None = None) -> 'Sequence':
    if mask_value is None:
      return self
    return mask_invalid(self, mask_value)

  def unmask(self) -> Sequence:
    return Sequence(self.values, self.mask)


def mask_invalid(
    sequence: Sequence,
    mask_value: complex | None = None,
) -> 'Sequence':
  """Returns a sequence with invalid timesteps replaced."""
  expanded_mask = sequence.expanded_mask()
  if mask_value is None:
    masked_values = mx.zeros_like(sequence.values)
    result_type = MaskedSequence
  else:
    masked_values = mx.full(
        sequence.values.shape, mask_value, sequence.values.dtype
    )
    result_type = Sequence
  masked_values = mx.where(expanded_mask, sequence.values, masked_values)
  return result_type(masked_values, sequence.mask)


# Defined outside of Sequence so mask_invalid can return MaskedSequence.
Sequence.mask_invalid = mask_invalid
