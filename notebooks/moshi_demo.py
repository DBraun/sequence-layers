# %% [markdown]
# # Moshi Speech Generation with SequenceLayers
#
# This script loads **Moshi** (Kyutai's speech-to-speech model) pretrained
# weights into sequence-layers MLX transformer blocks and runs autoregressive
# generation to produce audible speech.
#
# The Moshi architecture is a Depthformer with two stages:
# - **Main Transformer** (32 layers, d=4096, 32 heads, RoPE, SwiGLU) —
#   processes combined text + audio embeddings along the time dimension
# - **Depth Transformer** (6 layers per slice, d=1024, 16 heads, 8 slices) —
#   generates audio tokens hierarchically across RVQ codebook levels
#
# Reference: `/Users/braun/GitHub/moshi/moshi_mlx/moshi_mlx/`
#
# **Requirements:**
# ```
# pip install sequence-layers[mlx] rustymimi huggingface-hub sentencepiece
# ```
#
# **Hardware:** Apple Silicon with ≥32GB unified memory for bf16 weights.

# %% [markdown]
# ## 1. Setup & Constants

# %%
import functools
import time

import jax.nn
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

import sequence_layers.jax as sl
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import export

Sequence = bt.Sequence
ShapeDType = bt.ShapeDType

# --- Moshi config_v0_1 (7B) ---

# Main transformer.
MAIN_DIM = 4096
MAIN_HEADS = 32
MAIN_UPH = MAIN_DIM // MAIN_HEADS  # 128
MAIN_LAYERS = 32
MAIN_HIDDEN = 11 * MAIN_DIM // 4  # 11264 (SwiGLU hidden)
MAIN_CONTEXT = 512  # Original: 3000. Reduced for demo.
ROPE_BASE = 10_000.0

# Depth transformer (per-slice).
DEPTH_DIM = 1024
DEPTH_HEADS = 16
DEPTH_UPH = DEPTH_DIM // DEPTH_HEADS  # 64
DEPTH_LAYERS = 6
DEPTH_HIDDEN = 11 * DEPTH_DIM // 4  # 2816
DEPTH_CONTEXT = 8
NUM_SLICES = 8

# Vocabulary.
TEXT_IN_VOCAB = 32001
TEXT_OUT_VOCAB = 32000
AUDIO_VOCAB = 2049
AUDIO_OUT_VOCAB = AUDIO_VOCAB - 1  # 2048
NUM_CODEBOOKS = 16  # 8 model output + 8 user input
AUDIO_DELAYS = ([0] + [1] * 7) * 2  # 16 values

# Generation.
BATCH_SIZE = 1
NUM_STEPS = 100  # ~8 seconds at 12.5 Hz
SAMPLE_RATE = 24000
FRAME_RATE = 12.5  # Hz

print(f'Moshi config_v0_1 (7B):')
print(f'  Main: d={MAIN_DIM}, h={MAIN_HEADS}, L={MAIN_LAYERS}, '
      f'hidden={MAIN_HIDDEN}, context={MAIN_CONTEXT}')
print(f'  Depth: d={DEPTH_DIM}, h={DEPTH_HEADS}, L={DEPTH_LAYERS}, '
      f'hidden={DEPTH_HIDDEN}, slices={NUM_SLICES}')
print(f'  Vocab: text_in={TEXT_IN_VOCAB}, text_out={TEXT_OUT_VOCAB}, '
      f'audio={AUDIO_VOCAB}, codebooks={NUM_CODEBOOKS}')
print(f'  Generation: {NUM_STEPS} steps = '
      f'{NUM_STEPS / FRAME_RATE:.1f}s of audio')

# %% [markdown]
# ## 2. Sequence-Layers Configs
#
# Both transformers use pre-norm residual blocks with SwiGLU FFN.
# The main transformer has RoPE; the depth transformer has **none**.
#
# Compared to the original Moshi handwritten transformer, we add an
# explicit `Dense` after `Flatten` in the attention residual block to
# serve as the output projection (W_o), since sequence-layers'
# `DotProductSelfAttention` does not include one.

# %%


def main_transformer_config():
  """Main causal transformer: 32 layers, RoPE, SwiGLU."""
  return sl.Repeat.Config(
      num_repeats=MAIN_LAYERS,
      layer=sl.Serial.Config([
          # Self-attention residual block.
          sl.Residual.Config([
              sl.RMSNormalization.Config(epsilon=1e-8),
              sl.DotProductSelfAttention.Config(
                  num_heads=MAIN_HEADS,
                  units_per_head=MAIN_UPH,
                  max_past_horizon=MAIN_CONTEXT,
                  max_future_horizon=0,
                  query_network=sl.ApplyRotaryPositionalEncoding.Config(
                      max_wavelength=ROPE_BASE,
                  ),
                  key_network=sl.ApplyRotaryPositionalEncoding.Config(
                      max_wavelength=ROPE_BASE,
                  ),
              ),
              sl.Flatten.Config(),
              # Output projection W_o (Moshi has this; SL attention doesn't).
              sl.Dense.Config(features=MAIN_DIM, use_bias=False),
          ]),
          # SwiGLU FFN residual block.
          sl.Residual.Config([
              sl.RMSNormalization.Config(epsilon=1e-8),
              sl.Dense.Config(features=2 * MAIN_HIDDEN, use_bias=False),
              sl.GatedUnit.Config(
                  feature_activation=jax.nn.silu,
                  gate_activation=None,
              ),
              sl.Dense.Config(features=MAIN_DIM, use_bias=False),
          ]),
      ]),
  )


def depth_transformer_config():
  """Depth transformer: 6 layers, NO positional encoding."""
  return sl.Repeat.Config(
      num_repeats=DEPTH_LAYERS,
      layer=sl.Serial.Config([
          sl.Residual.Config([
              sl.RMSNormalization.Config(epsilon=1e-8),
              sl.DotProductSelfAttention.Config(
                  num_heads=DEPTH_HEADS,
                  units_per_head=DEPTH_UPH,
                  max_past_horizon=DEPTH_CONTEXT,
                  max_future_horizon=0,
                  # No positional encoding for depth!
              ),
              sl.Flatten.Config(),
              sl.Dense.Config(features=DEPTH_DIM, use_bias=False),
          ]),
          sl.Residual.Config([
              sl.RMSNormalization.Config(epsilon=1e-8),
              sl.Dense.Config(features=2 * DEPTH_HIDDEN, use_bias=False),
              sl.GatedUnit.Config(
                  feature_activation=jax.nn.silu,
                  gate_activation=None,
              ),
              sl.Dense.Config(features=DEPTH_DIM, use_bias=False),
          ]),
      ]),
  )


main_config = main_transformer_config()
depth_config = depth_transformer_config()
print('Configs defined.')

# %% [markdown]
# ## 3. Model Definition
#
# `MoshiModel` wraps sequence-layers transformer stacks with multi-modal
# embeddings and the depth generation loop. The main transformer uses a
# single `Repeat` model. The depth transformer uses 8 **separate**
# `Repeat` models (one per codebook slice) with per-slice weights but
# a **shared KV cache** — passing state between `step()` calls.

# %%

# Padding token used for uninitialized audio codebooks.
AUDIO_PADDING_TOKEN = AUDIO_VOCAB - 1  # 2048


def _safe_embed(embedding, tokens):
  """Embed with zero masking for negative token indices (Moshi convention)."""
  safe_tokens = mx.maximum(tokens, 0)
  y = embedding(safe_tokens)
  is_zero = tokens < 0
  return mx.where(is_zero[..., None], 0.0, y)


class MoshiModel(nn.Module):
  """Moshi speech model backed by SequenceLayers transformer blocks.

  Architecture:
    - text_emb + 16 audio_embs → sum → main transformer → out_norm
    - text_head predicts next text token
    - 8 depth transformer slices generate audio codebook tokens
      with shared KV cache (reset per time step)
  """

  def __init__(self, main_config, depth_config):
    super().__init__()

    # --- Main transformer components ---
    self.text_emb = nn.Embedding(TEXT_IN_VOCAB, MAIN_DIM)
    self.audio_embs = [
        nn.Embedding(AUDIO_VOCAB, MAIN_DIM) for _ in range(NUM_CODEBOOKS)
    ]
    self.main_stack = main_config.make(backend='mlx')
    self.out_norm = nn.RMSNorm(MAIN_DIM, eps=1e-8)
    self.text_head = nn.Linear(MAIN_DIM, TEXT_OUT_VOCAB, bias=False)

    # --- Depth transformer components (8 slices) ---
    # Each slice has its own transformer, embedding, and projections.
    self.depth_stacks = [
        depth_config.make(backend='mlx') for _ in range(NUM_SLICES)
    ]
    # Slice 0 conditions on text token, slices 1-7 on previous audio token.
    self.depth_text_emb = nn.Embedding(TEXT_IN_VOCAB, DEPTH_DIM)
    self.depth_audio_embs = [
        nn.Embedding(AUDIO_VOCAB, DEPTH_DIM) for _ in range(NUM_SLICES - 1)
    ]
    self.depth_linear_ins = [
        nn.Linear(MAIN_DIM, DEPTH_DIM, bias=False) for _ in range(NUM_SLICES)
    ]
    self.depth_linear_outs = [
        nn.Linear(DEPTH_DIM, AUDIO_OUT_VOCAB, bias=False)
        for _ in range(NUM_SLICES)
    ]

  def _embed(self, text_tokens, audio_tokens_list):
    """Sum text + all audio embeddings -> [batch, time, main_dim]."""
    x = _safe_embed(self.text_emb, text_tokens)
    for tok, emb in zip(audio_tokens_list, self.audio_embs):
      x = x + _safe_embed(emb, tok)
    return x

  def get_initial_main_state(self, batch_size):
    """Create fresh state for the main transformer."""
    spec = ShapeDType((MAIN_DIM,), mx.float32)
    return self.main_stack.get_initial_state(batch_size, spec)

  def generate_step(self, text_token, audio_tokens, main_state, sampler):
    """One autoregressive step: main transformer + depth generation.

    Args:
      text_token: [batch, 1] int32 — text token from previous step.
      audio_tokens: list of 16 [batch, 1] int32 — delayed audio inputs.
      main_state: main transformer KV cache state.
      sampler: callable(logits) -> (token, logprobs).

    Returns:
      next_text: [batch, 1] int32.
      next_audio: list of 8 [batch, 1] int32 (model output codebooks).
      new_main_state: updated state.
    """
    # --- Main transformer step ---
    x = self._embed(text_token, audio_tokens)
    mask = mx.ones((x.shape[0], 1), dtype=mx.bool_)
    y, main_state = self.main_stack.step(Sequence(x, mask), main_state)
    h = self.out_norm(y.values)  # [batch, 1, main_dim]

    # Text prediction.
    text_logits = self.text_head(h)  # [batch, 1, text_out_vocab]
    next_text, _ = sampler(text_logits[:, 0, :])
    next_text = next_text[:, None]  # [batch, 1]

    # --- Depth generation (8 slices, shared KV cache) ---
    depth_spec = ShapeDType((DEPTH_DIM,), mx.float32)
    depth_state = self.depth_stacks[0].get_initial_state(
        x.shape[0], depth_spec
    )

    prev_token = next_text
    next_audio = []

    for i in range(NUM_SLICES):
      # Embed previous token (text for slice 0, audio for rest).
      if i == 0:
        tok_emb = _safe_embed(self.depth_text_emb, prev_token)
      else:
        tok_emb = _safe_embed(self.depth_audio_embs[i - 1], prev_token)

      # Project main output + token embedding.
      dx = self.depth_linear_ins[i](h) + tok_emb

      # Step this slice's transformer, accumulating shared KV cache.
      dy, depth_state = self.depth_stacks[i].step(
          Sequence(dx, mask), depth_state
      )

      # Predict audio token.
      logits = self.depth_linear_outs[i](dy.values)  # [b, 1, 2048]
      token, _ = sampler(logits[:, 0, :])
      prev_token = token[:, None]
      next_audio.append(prev_token)

    return next_text, next_audio, main_state


print('MoshiModel class defined.')

# %% [markdown]
# ## 4. Build Model
#
# Instantiate and materialize deferred layers. At this point we have
# randomly initialized weights — section 5 loads pretrained weights.

# %%
model = MoshiModel(main_config, depth_config)

# Materialize deferred layers (attention needs to know in_features).
main_input_spec = ShapeDType((MAIN_DIM,), mx.float32)
depth_input_spec = ShapeDType((DEPTH_DIM,), mx.float32)
export._materialize_deferred(model.main_stack, BATCH_SIZE, main_input_spec)
for ds in model.depth_stacks:
  export._materialize_deferred(ds, BATCH_SIZE, depth_input_spec)

nparams = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
print(f'Model built: {nparams:,} parameters '
      f'({nparams * 2 / 1e9:.1f} GB in bf16)')

# %% [markdown]
# ## 5. Weight Loading
#
# Load pretrained Moshi weights from the MLX safetensors file into our
# sequence-layers model. The key mapping converts between Moshi's
# attribute paths and sequence-layers' internal structure.
#
# Critical transforms:
# - **QKV split**: Moshi's combined `in_proj.weight` [3d, d] is split
#   into separate `q_proj`, `k_proj`, `v_proj` and transposed.
# - **RoPE permutation** (main only): Moshi uses traditional (interleaved)
#   RoPE while sequence-layers uses non-interleaved. We permute Q/K
#   projection columns to compensate.

# %%


def _rope_permutation(units_per_head):
  """Permutation converting interleaved→half-split RoPE pairing."""
  return list(range(0, units_per_head, 2)) + list(
      range(1, units_per_head, 2)
  )


def _get_layer(repeat_model, layer_idx):
  """Get the Serial layer at index layer_idx from a Repeat model."""
  return repeat_model.layers[layer_idx]


def _get_attn_residual(serial_layer):
  """Get the attention Residual (first child of Serial)."""
  return serial_layer.layers[0]


def _get_ffn_residual(serial_layer):
  """Get the FFN Residual (second child of Serial)."""
  return serial_layer.layers[1]


def _set_qkv(attn_module, q_proj, k_proj, v_proj):
  """Set Q/K/V projections on DotProductSelfAttention."""
  # Navigate through DeferredDotProductSelfAttention._inner.
  inner = attn_module._inner
  inner.q_proj = q_proj
  inner.k_proj = k_proj
  inner.v_proj = v_proj


def _set_dense_weight(dense_module, weight):
  """Set weight on DenseDeferred._inner._linear."""
  dense_module._inner._linear.weight = weight


def _set_norm_weight(norm_module, weight):
  """Set weight on RMSNormalization.

  from_config passes scale_init=ones (not None), so _ensure_initialized
  always creates _scale (not _rms_norm). Handle both paths for safety.
  """
  if norm_module._rms_norm is not None:
    norm_module._rms_norm.weight = weight
  else:
    norm_module._scale = weight


def _load_transformer_layer(
    serial_layer,
    weights,
    prefix,
    num_heads,
    units_per_head,
    apply_rope_permutation=False,
):
  """Load weights for one transformer layer (attention + FFN residual).

  Args:
    serial_layer: The Serial combinator containing [attn_residual, ffn_residual].
    weights: dict of safetensors weights.
    prefix: key prefix, e.g. 'transformer.layers.0'.
    num_heads: number of attention heads.
    units_per_head: dimension per head.
    apply_rope_permutation: if True, permute Q/K for RoPE compatibility.
  """
  attn_res = _get_attn_residual(serial_layer)
  ffn_res = _get_ffn_residual(serial_layer)

  # --- Attention residual: [RMSNorm, Attention, Flatten, Dense(out_proj)] ---
  body = attn_res._body.layers

  # norm1
  _set_norm_weight(body[0], weights[f'{prefix}.norm1.weight'])

  # QKV from combined in_proj.
  in_proj = weights[f'{prefix}.self_attn.in_proj.weight']  # [3*d, d]
  d = num_heads * units_per_head
  q_w, k_w, v_w = mx.split(in_proj, 3, axis=0)  # each [d, d]
  q_proj = q_w.T  # [d, d] Linen [in, out]
  k_proj = k_w.T
  v_proj = v_w.T

  if apply_rope_permutation:
    P = _rope_permutation(units_per_head)
    q_proj = q_proj.reshape(d, num_heads, units_per_head)[:, :, P].reshape(
        d, d
    )
    k_proj = k_proj.reshape(d, num_heads, units_per_head)[:, :, P].reshape(
        d, d
    )

  _set_qkv(body[1], q_proj, k_proj, v_proj)

  # out_proj (Dense after Flatten).
  _set_dense_weight(body[3], weights[f'{prefix}.self_attn.out_proj.weight'])

  # --- FFN residual: [RMSNorm, Dense(gate+up), GatedUnit, Dense(down)] ---
  ffn_body = ffn_res._body.layers

  # norm2
  _set_norm_weight(ffn_body[0], weights[f'{prefix}.norm2.weight'])

  # linear_in (gate + up projection).
  _set_dense_weight(ffn_body[1], weights[f'{prefix}.gating.linear_in.weight'])

  # linear_out (down projection).
  _set_dense_weight(ffn_body[3], weights[f'{prefix}.gating.linear_out.weight'])


def load_moshi_weights(model, weights_path):
  """Load Moshi MLX safetensors into the sequence-layers MoshiModel.

  Args:
    model: MoshiModel instance (with materialized deferred layers).
    weights_path: path to model.safetensors.
  """
  print(f'Loading weights from {weights_path}...')
  weights = mx.load(weights_path)
  print(f'  {len(weights)} weight tensors loaded.')

  consumed = set()

  def get(key):
    consumed.add(key)
    return weights[key]

  # --- Embeddings ---
  model.text_emb.weight = get('text_emb.weight')
  for cb in range(NUM_CODEBOOKS):
    model.audio_embs[cb].weight = get(f'audio_embs.{cb}.weight')

  # --- Main transformer (32 layers) ---
  for i in range(MAIN_LAYERS):
    serial = _get_layer(model.main_stack, i)
    prefix = f'transformer.layers.{i}'
    consumed.update([
        f'{prefix}.self_attn.in_proj.weight',
        f'{prefix}.self_attn.out_proj.weight',
        f'{prefix}.norm1.weight',
        f'{prefix}.norm2.weight',
        f'{prefix}.gating.linear_in.weight',
        f'{prefix}.gating.linear_out.weight',
    ])
    _load_transformer_layer(
        serial,
        weights,
        prefix,
        num_heads=MAIN_HEADS,
        units_per_head=MAIN_UPH,
        apply_rope_permutation=True,
    )
  print(f'  Main transformer: {MAIN_LAYERS} layers loaded.')

  # --- Output norm + text head ---
  model.out_norm.weight = get('out_norm.weight')
  model.text_head.weight = get('text_linear.weight')

  # --- Depth transformer (8 slices x 6 layers each) ---
  for s in range(NUM_SLICES):
    # Embeddings.
    if s == 0:
      model.depth_text_emb.weight = get(f'depformer.slices.{s}.emb.weight')
    else:
      model.depth_audio_embs[s - 1].weight = get(
          f'depformer.slices.{s}.emb.weight'
      )

    # Linear projections.
    model.depth_linear_ins[s].weight = get(
        f'depformer.slices.{s}.linear_in.weight'
    )
    model.depth_linear_outs[s].weight = get(
        f'depformer.slices.{s}.linear_out.weight'
    )

    # Transformer layers.
    for j in range(DEPTH_LAYERS):
      serial = _get_layer(model.depth_stacks[s], j)
      prefix = f'depformer.slices.{s}.transformer.layers.{j}'
      consumed.update([
          f'{prefix}.self_attn.in_proj.weight',
          f'{prefix}.self_attn.out_proj.weight',
          f'{prefix}.norm1.weight',
          f'{prefix}.norm2.weight',
          f'{prefix}.gating.linear_in.weight',
          f'{prefix}.gating.linear_out.weight',
      ])
      _load_transformer_layer(
          serial,
          weights,
          prefix,
          num_heads=DEPTH_HEADS,
          units_per_head=DEPTH_UPH,
          apply_rope_permutation=False,  # No RoPE in depformer.
      )

  print(f'  Depth transformer: {NUM_SLICES} slices x '
        f'{DEPTH_LAYERS} layers loaded.')

  # Check for unconsumed keys (potential mapping errors).
  unconsumed = set(weights.keys()) - consumed
  if unconsumed:
    print(f'  WARNING: {len(unconsumed)} unconsumed weight keys:')
    for k in sorted(unconsumed)[:20]:
      print(f'    {k}: {weights[k].shape}')
    if len(unconsumed) > 20:
      print(f'    ... and {len(unconsumed) - 20} more')

  mx.eval(model.parameters())
  print('  Weights loaded and evaluated.')


# %% [markdown]
# ## 6. Download & Load Weights
#
# Download from HuggingFace and load into the model. This requires
# ~14 GB of disk space and ≥32 GB of unified memory.

# %%
HF_REPO = 'kyutai/moshiko-mlx-bf16'

try:
  from huggingface_hub import hf_hub_download

  model_path = hf_hub_download(HF_REPO, 'model.safetensors')
  tokenizer_path = hf_hub_download(HF_REPO, 'tokenizer_spm_32k_3.model')
  mimi_path = hf_hub_download(
      HF_REPO, 'tokenizer-e351c8d8-checkpoint125.safetensors'
  )
  print(f'Model: {model_path}')
  print(f'Tokenizer: {tokenizer_path}')
  print(f'Mimi: {mimi_path}')

  load_moshi_weights(model, model_path)

except (ImportError, Exception) as e:
  print(f'Could not download/load weights: {e}')
  print('Continuing with random weights for testing.')
  model_path = None
  tokenizer_path = None
  mimi_path = None

# %% [markdown]
# ## 7. Sampling Utilities

# %%


def top_p_sample(logits, top_p=0.95, temperature=0.8):
  """Top-p (nucleus) sampling from logits.

  Args:
    logits: [batch, vocab] float32.
    top_p: cumulative probability threshold.
    temperature: softmax temperature.

  Returns:
    tokens: [batch] int32.
  """
  probs = mx.softmax(logits / temperature, axis=-1)
  sorted_indices = mx.argsort(probs, axis=-1)
  sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
  cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

  # Keep tokens with cumulative prob above (1 - top_p).
  top_probs = mx.where(cumulative_probs > 1.0 - top_p, sorted_probs, 0.0)
  sampled = mx.random.categorical(mx.log(top_probs + 1e-10), axis=-1)
  tokens = mx.take_along_axis(
      sorted_indices, sampled[..., None], axis=-1
  ).squeeze(-1)
  return tokens


def make_sampler(temperature=0.8, top_p=0.95):
  """Create a sampler callable matching Moshi's interface."""
  def sampler(logits):
    if temperature == 0:
      token = mx.argmax(logits, axis=-1)
    else:
      token = top_p_sample(logits, top_p=top_p, temperature=temperature)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return token.astype(mx.int32), logprobs
  return sampler


# %% [markdown]
# ## 8. Autoregressive Generation
#
# At each time step:
# 1. Gather delayed audio inputs
# 2. Embed text + audio, step main transformer
# 3. Predict text token
# 4. Depth generation: 8 slices, shared KV cache
# 5. Store generated tokens with delay accounting

# %%
print(f'\nGenerating {NUM_STEPS} steps '
      f'({NUM_STEPS / FRAME_RATE:.1f}s of audio)...\n')

sampler = make_sampler(temperature=0.8, top_p=0.95)

# Generation buffer: [batch, 1+16 codebooks, time].
# Index 0 = text, indices 1..16 = audio codebooks.
gen_seq = np.zeros((BATCH_SIZE, 1 + NUM_CODEBOOKS, NUM_STEPS + 4),
                   dtype=np.int32)

# Initialize main transformer state.
main_state = model.get_initial_main_state(BATCH_SIZE)

t0 = time.time()

for t in range(NUM_STEPS):
  # --- 1. Gather inputs with delays ---
  # Text: previous step's output.
  text_idx = max(0, t - 1)
  text_token = mx.array(gen_seq[:, 0:1, text_idx])  # [batch, 1]

  # Audio: each codebook delayed by audio_delays[cb].
  audio_inputs = []
  for cb in range(NUM_CODEBOOKS):
    delay = AUDIO_DELAYS[cb]
    gen_idx = t - 1 - delay
    if gen_idx >= 0:
      tok = mx.array(gen_seq[:, cb + 1:cb + 2, gen_idx])
    else:
      tok = mx.full((BATCH_SIZE, 1), AUDIO_PADDING_TOKEN, dtype=mx.int32)
    audio_inputs.append(tok)

  # --- 2-5. Model step ---
  next_text, next_audio, main_state = model.generate_step(
      text_token, audio_inputs, main_state, sampler
  )
  mx.eval(next_text, *next_audio)

  # --- Store generated tokens ---
  gen_seq[:, 0, t] = np.array(next_text[:, 0])
  for i in range(NUM_SLICES):
    gen_seq[:, i + 1, t] = np.array(next_audio[i][:, 0])
  # User codebooks (8-15) remain 0 (silence).

  if (t + 1) % 10 == 0:
    elapsed = time.time() - t0
    steps_per_sec = (t + 1) / elapsed
    realtime_factor = steps_per_sec / FRAME_RATE
    print(f'  Step {t+1}/{NUM_STEPS}  '
          f'{steps_per_sec:.1f} steps/s  '
          f'{realtime_factor:.2f}x realtime')

elapsed = time.time() - t0
print(f'\nGeneration complete: {NUM_STEPS} steps in {elapsed:.1f}s '
      f'({NUM_STEPS/elapsed:.1f} steps/s)')

# %% [markdown]
# ## 9. Decode Text (optional)

# %%
try:
  import sentencepiece as spm

  if tokenizer_path is not None:
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    text_tokens = gen_seq[0, 0, :NUM_STEPS].tolist()
    decoded_text = sp.DecodeIds(text_tokens)
    print(f'Generated text: {decoded_text}')
  else:
    text_tokens = gen_seq[0, 0, :NUM_STEPS].tolist()
    print(f'Text token IDs: {text_tokens[:20]}...')
except ImportError:
  text_tokens = gen_seq[0, 0, :NUM_STEPS].tolist()
  print(f'Text token IDs (sentencepiece not installed): '
        f'{text_tokens[:20]}...')

# %% [markdown]
# ## 10. Decode Audio & Save WAV
#
# Use rustymimi to decode the first 8 codebooks (model's speech output)
# into PCM audio and save as a WAV file.

# %%
output_path = 'moshi_output.wav'

try:
  import rustymimi

  if mimi_path is not None:
    print(f'Decoding audio with Mimi...')
    mimi = rustymimi.Tokenizer(mimi_path, num_codebooks=8)

    # Extract model output codebooks (first 8).
    codes = gen_seq[:, 1:9, :NUM_STEPS].astype(np.uint32)  # [1, 8, T]
    pcm = mimi.decode(codes)  # [1, 1, samples]
    audio_samples = pcm[0, 0]

    rustymimi.write_wav(output_path, audio_samples, SAMPLE_RATE)
    duration = len(audio_samples) / SAMPLE_RATE
    print(f'Saved {output_path}: {duration:.1f}s, {SAMPLE_RATE}Hz')
  else:
    print('No Mimi weights — skipping audio decode.')
except ImportError:
  print('rustymimi not installed — skipping audio decode.')
  print('Install with: pip install rustymimi')

# %% [markdown]
# ## 11. Comparison: Moshi vs SequenceLayers
#
# | Aspect | Moshi (handwritten) | SequenceLayers |
# |--------|-------------------|----------------|
# | Transformer | `Transformer`, `TransformerLayer`, `Attention`, `MlpGating` classes (~300 lines) | `Repeat(Serial(Residual(...), Residual(...)))` config (~30 lines) |
# | KV cache | Manual `KVCache`, `RotatingKVCache` classes | Automatic via `step()` with ring buffer |
# | Streaming | Custom `__call__` with cache threading | `model.step(x, state)` → updated state |
# | Depformer | `DepFormerSlice` with per-slice Transformer | 8 independent Repeat stacks, shared state passing |
# | RoPE | `nn.RoPE(traditional=True)` manually applied | `ApplyRotaryPositionalEncoding.Config(...)` in attention config |
# | Weight loading | `load_pytorch_weights()` (~80 lines) | `load_moshi_weights()` walks SL model tree |
#
# The model-specific code (embeddings, depth loop, delay logic) stays
# custom. SequenceLayers provides the transformer building blocks with
# automatic streaming, KV caching, and verified step/layer equivalence.

# %%
print('Done!')
