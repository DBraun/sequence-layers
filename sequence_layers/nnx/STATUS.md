# NNX Port Status

Proof-of-concept port of `sequence_layers` from Flax Linen to Flax NNX.

**520 tests passing** across all ported modules.

## Ported

### types.py — Core base classes
All base classes ported: `SequenceLayer`, `Steppable`, `Stateless`,
`StatelessPointwise`, `StatelessPointwiseFunctor`, `Emitting`,
`StatelessEmitting`, `PreservesType`, `PreservesShape`.

### simple.py — Stateless / pointwise layers
| Ported | Status |
|--------|--------|
| Abs, Add, Affine, Argmax, Cast | Done |
| Downsample1D, Dropout, EinopsRearrange | Done |
| Elu, Embedding, EmbeddingTranspose | Done |
| Emit, NamedEmit | Done |
| ExpandDims, Exp, Flatten | Done |
| GatedLinearUnit, GatedTanhUnit, GatedUnit, Gelu | Done |
| Identity, Lambda, LeakyRelu, Log | Done |
| MaskInvalid, Max, Maximum, Mean, Min, Minimum, Mod | Done |
| MoveAxis, OneHot, Power, PRelu, Relu, Reshape | Done |
| Scale, Sigmoid, Slice, Snake, Softmax, Softplus | Done |
| Squeeze, Sum, SwapAxes, Swish, Tanh | Done |
| Transpose, Upsample1D, Upsample2D | Done |

### dense.py
| Ported | Status |
|--------|--------|
| Dense | Done |
| DenseShaped | Done |
| EinsumDense | Done |

### convolution.py
| Ported | Status |
|--------|--------|
| Conv1D, Conv1DTranspose | Done |
| Conv2D, Conv2DTranspose | Done |
| Conv3D, DepthwiseConv1D | Done |

### normalization.py
| Ported | Status |
|--------|--------|
| LayerNormalization, RMSNormalization | Done |
| GroupNormalization, BatchNormalization | Done |
| L2Normalize | Done |

### pooling.py
| Ported | Status |
|--------|--------|
| MaxPooling1D/2D/3D | Done |
| MinPooling1D/2D/3D | Done |
| AveragePooling1D/2D/3D | Done |

### recurrent.py
| Ported | Status |
|--------|--------|
| LSTM | Done |
| RGLRU | Done |

### position.py
| Ported | Status |
|--------|--------|
| AddTimingSignal | Done |
| ApplyRotaryPositionalEncoding | Done |

### attention.py
| Ported | Status |
|--------|--------|
| DotProductSelfAttention | Done |
| DotProductAttention (cross) | Done |
| GmmAttention | Done |

### combinators.py
| Ported | Status |
|--------|--------|
| Serial | Done |
| Residual | Done |
| Parallel | Done |
| Bidirectional | Done |
| Repeat | Done |
| Blockwise | Done |

### dsp.py
| Ported | Status |
|--------|--------|
| Delay | Done |
| Lookahead | Done |
| Window | Done |
| Frame | Done |
| OverlapAdd | Done |

---

## Deferred

Items below were evaluated and deferred due to deep Linen dependencies
(`nn.scan`, `nn.compact`, `nn.share_scope`, `FlaxEinsumDense`) or
infrastructure-specific patterns that don't have straightforward NNX
equivalents.

### simple.py
| Layer | Reason |
|-------|--------|
| GradientClipping | Uses `jax.custom_gradient`; low priority utility |
| ApplySharding | Requires `sharding_lib` infrastructure |
| OptimizationBarrier | Low priority utility (`jax.lax.optimization_barrier`) |
| CheckpointName | Low priority utility (`jax.ad_checkpoint.checkpoint_name`) |
| Logging | Debug utility; uses `jax.debug.callback` |
| GlobalEinopsRearrange | Does not support step; complex mask handling with time dim |
| GlobalReshape | Does not support step; complex mask aggregation logic |

### dense.py
| Layer | Reason |
|-------|--------|

### normalization.py
| Layer | Reason |
|-------|--------|
| WeightNormalization | Wraps another layer's kernel; complex `nn.compact` pattern |
| L2WeightNormalization | Same as WeightNormalization |
| SpectralWeightNormalization | Same; additionally uses spectral norm estimation |

### combinators.py
| Layer | Reason |
|-------|--------|
| SerialModules | Variant of Serial using `nn.share_scope`; NNX doesn't need this |
| ParallelChannels | Splits channels across branches; medium complexity |
| CheckpointGradient | Uses `nn.remat`; NNX has `nnx.remat` but API differs |

### attention/
| Layer | Reason |
|-------|--------|
| DotProductSelfAttentionV2 | Flash attention backend; complex `AttentionInputProjectionHelper` |
| LocalDotProductSelfAttention | Local windowed attention; relative position embeddings |
| StreamingDotProductAttention | Streaming cross-attention with KV cache ring buffers |
| StreamingLocalDotProductAttention | Combines streaming + local windowing |
| BlockwiseDotProductSelfAttention | Flash attention + block-based segment masking |
| MultiSourceDotProductAttention | Multi-source cross-attention with flash attention |
| ShawRelativePositionEmbedding | Relative position embedding variant |
| T5RelativePositionEmbedding | Relative position embedding variant |
| TransformerXLRelativePositionEmbedding | Relative position embedding variant |

### dsp.py
| Layer | Reason |
|-------|--------|
| FFT, IFFT, RFFT, IRFFT | FFT wrappers; depend on Frame/OverlapAdd (now ported) |
| STFT, InverseSTFT | Depend on Frame, Window, FFT layers |
| LinearToMelSpectrogram | Depends on STFT infrastructure |

### conditioning.py (entire module)
| Layer | Reason |
|-------|--------|
| Conditioning | Uses `nn.share_scope` for parameter sharing across branches |

---

## Suggested Next Steps (by priority)

1. **FFT/IFFT/RFFT/IRFFT** — FFT wrappers; Frame/OverlapAdd now ported.
   Unlocks STFT/InverseSTFT.
2. **STFT, InverseSTFT** — Depend on Frame, Window, FFT layers (all
   prerequisites now ported except FFT).
3. **CheckpointGradient** — NNX has `nnx.remat`; should be a thin wrapper.
4. **ParallelChannels** — medium complexity, useful combinator.
5. **Relative position embeddings** (Shaw, T5, TransformerXL) — easy to
   moderate; prerequisite for LocalDotProductSelfAttention.
6. **Advanced attention variants** — require porting
   `AttentionInputProjectionHelper` and attention common utilities first.
   `EinsumDense` and `GmmAttention` are now available.
   This is a large effort that unlocks all 6+ remaining attention variants.
