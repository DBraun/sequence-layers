# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sequence Layers is a neural network library (JAX and TensorFlow 2) for building sequence models that support both layer-by-layer (parallel) processing for training and step-by-step (streaming) processing for inference. Created by Google.

## Commands

### Install
```bash
pip install -e .          # Core (JAX only)
pip install -e .[dev]     # Development dependencies
pip install -e .[tensorflow]  # TensorFlow support
```

### Test
```bash
# Run all JAX tests
pytest sequence_layers/jax/

# Run all TensorFlow tests
pytest sequence_layers/tensorflow/

# Run a single test file
pytest sequence_layers/jax/simple_test.py

# Run a specific test
pytest sequence_layers/jax/simple_test.py -k "test_name"

# Run tests in parallel
pytest sequence_layers/jax/ -n auto
```

### Format & Lint
```bash
# Format with pyink (Google Python style)
pyink --line-length 80 --unstable --pyink-indentation 2 --pyink-use-majority-quotes sequence_layers/

# Lint
pylint sequence_layers/
```

## Code Style

- **Formatter:** pyink (Google Python style)
- **Line length:** 80 characters
- **Indentation:** 2 spaces
- **Quotes:** majority quotes (single preferred)

## Architecture

### Core Abstraction: `SequenceLayer`

Defined in `sequence_layers/jax/types.py`. Every layer implements two execution modes:

1. **`layer()`** — Process an entire sequence at once (parallel, for training)
2. **`step()`** — Process fixed-size blocks with state (streaming, for inference)

The contract guarantees that `step()` produces identical results to `layer()` for the same input (after accounting for latency).

### The `Sequence` Type

A container of `(values, mask)` where `values` has shape `[batch, time, ...channels]` and `mask` is `[batch, time]` (True = valid). All layers propagate masks to track invalid/padded timesteps.

### Key Properties of Layers

- **`block_size`** — Input block size for step-wise processing
- **`output_ratio`** — Ratio of output timesteps to input timesteps (for resampling layers)
- **`supports_step`** — Whether the layer supports streaming
- **`input_latency` / `output_latency`** — Buffering requirements for streaming

### Module Organization (JAX)

All JAX layers live under `sequence_layers/jax/`:

| Module | Contents |
|---|---|
| `types.py` | Core types: `Sequence`, `SequenceLayer`, `Steppable`, padding modes |
| `simple.py` | Stateless/pointwise layers (activations, reshaping, embeddings, 100+ types) |
| `dense.py` | Fully connected layers |
| `convolution.py` | 1D/2D/3D convolutions with causal/semicausal padding |
| `recurrent.py` | LSTM, RGLRU |
| `attention/` | Multi-head attention variants (dot-product, local, streaming, cross, GMM) |
| `normalization.py` | Batch/Layer/Group/Instance normalization |
| `pooling.py` | Max/Avg/L2Norm pooling, reduction layers |
| `combinators.py` | Composition: `Serial`, `Parallel`, `Residual`, `Bidirectional`, `Repeat` |
| `position.py` | Positional encodings |
| `dsp.py` / `signal.py` | Digital signal processing layers |
| `conditioning.py` | Conditional processing |
| `export.py` | Model export for deployment |
| `test_utils.py` | Test base class `SequenceLayerTest`, `random_sequence()`, `verify_contract()` |

The TensorFlow implementation mirrors this structure under `sequence_layers/tensorflow/`.

### NNX Port (Proof-of-Concept)

A Flax NNX port lives under `sequence_layers/nnx/`. It mirrors the JAX/Linen module structure but uses NNX conventions: eager parameter creation, `train()`/`eval()` mode switching, no `bind`/`apply` ceremony. See `sequence_layers/nnx/STATUS.md` for what has been ported and what is deferred.

```bash
# Run all NNX tests
pytest sequence_layers/nnx/
```

Key NNX differences from the Linen version:
- Direct construction: `Dense(in_features=5, features=3, rngs=nnx.Rngs(0))` — no Config.make() pattern
- `self.deterministic` replaces `training: bool` parameter (set by `train()`/`eval()`)
- Use `variable[...]` not `.value` to access `nnx.Param` / `nnx.BatchStat` arrays
- Use `nnx.List` (not plain `list`) for child module containers

### Padding Modes

Defined in `types.py`: `VALID`, `SAME`, `CAUSAL`, `CAUSAL_VALID`, `REVERSE_CAUSAL`, `REVERSE_CAUSAL_VALID`, `SEMICAUSAL`, `SEMICAUSAL_FULL`. Causal modes are essential for streaming.

### Testing Patterns

Tests use `absltest` + `parameterized` (not plain pytest classes). The key test utility is `verify_contract()` which validates that a layer's `step()` and `layer()` outputs match. Test files are `*_test.py` adjacent to their source files.

### Build System

Uses **Flit** (`flit_core`) as the build backend. Python >=3.11 required.
