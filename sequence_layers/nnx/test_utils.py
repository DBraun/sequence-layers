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
"""NNX test utilities."""

import random

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax.test_utils import random_sequence
from sequence_layers.jax import types as jax_types
from sequence_layers.nnx import types


def _pad_to_multiple(
    x: types.Sequence, block_size: int
) -> tuple[types.Sequence, int, int]:
  """Pads x to a multiple of block_size, returning (padded, original, blocks)."""
  t = x.shape[1]
  if t == 0:
    return x, t, 0
  remainder = t % block_size
  if remainder != 0:
    pad = block_size - remainder
    x = x.pad_time(0, pad, valid=False)
  padded_t = x.shape[1]
  num_blocks = padded_t // block_size
  return x, t, num_blocks


def step_by_step(
    l: types.Steppable,
    x: types.Sequence,
    *,
    constants: types.Constants | None = None,
    blocks_per_step: int = 1,
) -> tuple[types.Sequence, types.State]:
  """Executes a SequenceLayer step-by-step in a Python loop.

  Args:
    l: The SequenceLayer to invoke step-by-step.
    x: The input Sequence to process.
    constants: Optional constants to provide.
    blocks_per_step: Runs the layer with this multiple of l.block_size
      timesteps per step.

  Returns:
    The resulting sequence and final state.
  """
  if not l.supports_step:
    raise ValueError(f'{l} cannot be stepped.')

  input_block_size = l.block_size * blocks_per_step
  x, _, num_blocks = _pad_to_multiple(x, input_block_size)

  state = l.get_initial_state(
      batch_size=x.shape[0],
      input_spec=x.channel_spec,
      constants=constants,
  )

  output_blocks = []
  for b in range(num_blocks):
    start = b * input_block_size
    end = start + input_block_size
    x_block = x[:, start:end]
    pad_amount = input_block_size - x_block.shape[1]
    if pad_amount:
      x_block = x_block.pad_time(0, pad_amount, valid=False)

    y_block, state = l.step(x_block, state, constants=constants)
    output_blocks.append(y_block)

  output = types.Sequence.concatenate_sequences(output_blocks)
  return output, state


def _mask_and_pad_to_max_length(
    a: types.Sequence, b: types.Sequence
) -> tuple[types.Sequence, types.Sequence]:
  a = a.mask_invalid()
  b = b.mask_invalid()
  a_time = a.values.shape[1]
  b_time = b.values.shape[1]
  max_time = max(a_time, b_time)
  a = a.pad_time(0, max_time - a_time, valid=False)
  b = b.pad_time(0, max_time - b_time, valid=False)
  return a, b


class SequenceLayerTest(parameterized.TestCase):
  """Base class for NNX SequenceLayer tests."""

  def setUp(self):
    super().setUp()
    random.seed(123456789)
    np.random.seed(123456789)

  def verify_contract(
      self,
      l: types.SequenceLayer,
      x: types.Sequence,
      *,
      constants: types.Constants | None = None,
      rtol: float = 1e-6,
      atol: float = 1e-6,
      test_gradients: bool = True,
      grad_rtol: float | None = None,
      grad_atol: float | None = None,
      padding_invariance_pad_value: float = jnp.nan,
      test_2x_step: bool = True,
      test_padding_invariance: bool = True,
  ) -> types.Sequence:
    """Verifies that the provided NNX layer obeys the SequenceLayer contract.

    Tests:
    1. Layer-wise and step-wise equivalence of outputs and gradients.
    2. Step must support any multiple of block size.
    3. Padding invariance.

    Args:
      l: The NNX layer to test (must already be constructed).
      x: The sequence to use as input.
      constants: Optional constants.
      rtol: Relative tolerance.
      atol: Absolute tolerance.
      test_gradients: Whether to compare gradients.
      grad_rtol: Gradient relative tolerance (defaults to rtol).
      grad_atol: Gradient absolute tolerance (defaults to atol).
      padding_invariance_pad_value: Value for padding invariance test.
      test_2x_step: Whether to test 2x block stepping.
      test_padding_invariance: Whether to test padding invariance.

    Returns:
      The masked layer-wise output.
    """
    if grad_rtol is None:
      grad_rtol = rtol
    if grad_atol is None:
      grad_atol = atol

    # Ensure eval mode for contract testing.
    l.eval()

    output_latency = l.output_latency
    input_latency = l.input_latency

    # We use nnx.split/merge for creating fresh copies of the model.
    graphdef, param_state, other_state = nnx.split(l, nnx.Param, ...)

    def _fresh_model():
      return nnx.merge(graphdef, param_state, other_state)

    # ---- Layer-wise output ----
    def layer_fn(model, x):
      return model.layer(x, constants=constants).mask_invalid()

    # ---- Gradient testing uses nnx.grad ----
    # nnx.grad differentiates through nnx.Param leaves of the model
    # automatically, and we pass x_values separately to also get input
    # gradients (the mask is boolean and non-differentiable).
    if test_gradients:
      x_mask = x.mask

      @nnx.grad(argnums=(0, 1))
      def layer_grad_fn(model, x_values):
        x_seq = jax_types.Sequence(x_values, x_mask).mask_invalid()
        y = model.layer(x_seq, constants=constants).mask_invalid()
        return jnp.sum(y.values ** 2)

      model_copy = _fresh_model()
      layer_model_grad, layer_x_values_grad = layer_grad_fn(
          model_copy, x.values
      )
      y_layer = layer_fn(_fresh_model(), x)
    else:
      y_layer = layer_fn(_fresh_model(), x)

    # ---- Padding invariance ----
    pad_amount = 4 * l.block_size
    if test_padding_invariance:
      x_padded = x.pad_time(0, pad_amount, valid=False).mask_invalid(
          padding_invariance_pad_value
      )
      y_layer_padded = layer_fn(_fresh_model(), x_padded)
      self.assertSequencesClose(
          y_layer, y_layer_padded, rtol=rtol, atol=atol
      )

    if l.supports_step:
      # Extend x by the input latency.
      x_step = x.pad_time(0, input_latency, valid=False)

      # ---- Step-wise output ----
      if test_gradients:
        x_step_mask = x_step.mask

        @nnx.grad(argnums=(0, 1))
        def step_grad_fn(model, x_step_values):
          x_seq = jax_types.Sequence(
              x_step_values, x_step_mask
          ).mask_invalid()
          y_step, _ = step_by_step(model, x_seq, constants=constants)
          y_step = y_step[:, output_latency:]
          y_step = y_step.mask_invalid()
          return jnp.sum(y_step.values ** 2)

        model_copy = _fresh_model()
        step_model_grad, step_x_values_grad = step_grad_fn(
            model_copy, x_step.values
        )
        # Also compute the actual step output for comparison.
        y_step, _ = step_by_step(
            _fresh_model(), x_step, constants=constants
        )
        y_step = y_step[:, output_latency:].mask_invalid()
      else:
        y_step, _ = step_by_step(
            _fresh_model(), x_step, constants=constants
        )
        y_step = y_step[:, output_latency:].mask_invalid()

      # Property 1: Layer/step equivalence.
      self.assertSequencesClose(y_layer, y_step, rtol=rtol, atol=atol)

      # 2x step test.
      if test_2x_step:
        y_step_2x, _ = step_by_step(
            _fresh_model(), x_step, constants=constants, blocks_per_step=2
        )
        y_step_2x = y_step_2x[:, output_latency:].mask_invalid()
        self.assertSequencesClose(y_layer, y_step_2x, rtol=rtol, atol=atol)

      # Padding invariance for step.
      if test_padding_invariance:
        x_padded_step = x_padded.pad_time(0, input_latency, valid=False)
        y_step_padded, _ = step_by_step(
            _fresh_model(), x_padded_step, constants=constants
        )
        y_step_padded = y_step_padded[:, output_latency:].mask_invalid()
        self.assertSequencesClose(
            y_step, y_step_padded, rtol=rtol, atol=atol
        )

      # Gradient equivalence.
      if test_gradients:
        # Compare x value gradients (trim step to match layer length).
        layer_x_g = layer_x_values_grad
        step_x_g = step_x_values_grad[:, :layer_x_g.shape[1]]
        # Mask out invalid timesteps before comparison.
        layer_x_g = jax_types.Sequence(
            layer_x_g, x.mask
        ).mask_invalid().values
        step_x_g = jax_types.Sequence(
            step_x_g, x.mask
        ).mask_invalid().values
        chex.assert_trees_all_close(
            layer_x_g, step_x_g,
            rtol=grad_rtol, atol=grad_atol,
        )
        # Compare param gradients (extract Param state from model grads).
        _, layer_param_grad, _ = nnx.split(
            layer_model_grad, nnx.Param, ...
        )
        _, step_param_grad, _ = nnx.split(
            step_model_grad, nnx.Param, ...
        )
        chex.assert_trees_all_close(
            layer_param_grad, step_param_grad,
            rtol=grad_rtol, atol=grad_atol,
        )

    return y_layer

  def assertSequencesClose(  # pylint: disable=invalid-name
      self,
      a: types.Sequence,
      b: types.Sequence,
      atol: float = 1e-6,
      rtol: float = 1e-6,
  ):
    a, b = _mask_and_pad_to_max_length(a, b)
    chex.assert_trees_all_close(a.values, b.values, atol=atol, rtol=rtol)
    chex.assert_trees_all_equal(a.mask, b.mask)

  def assertSequencesEqual(  # pylint: disable=invalid-name
      self,
      a: types.Sequence,
      b: types.Sequence,
  ):
    a, b = _mask_and_pad_to_max_length(a, b)
    chex.assert_trees_all_equal(a.values, b.values)
    chex.assert_trees_all_equal(a.mask, b.mask)


main = absltest.main
