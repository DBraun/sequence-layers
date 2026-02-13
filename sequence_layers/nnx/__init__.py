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
"""NNX proof-of-concept port of sequence_layers."""

# Core types.
from sequence_layers.nnx.types import ChannelSpec
from sequence_layers.nnx.types import Constants
from sequence_layers.nnx.types import DType
from sequence_layers.nnx.types import Emits
from sequence_layers.nnx.types import Emitting
from sequence_layers.nnx.types import MaskedSequence
from sequence_layers.nnx.types import MASK_DTYPE
from sequence_layers.nnx.types import MaskT
from sequence_layers.nnx.types import PaddingMode
from sequence_layers.nnx.types import PreservesShape
from sequence_layers.nnx.types import PreservesType
from sequence_layers.nnx.types import ReceptiveField
from sequence_layers.nnx.types import Sequence
from sequence_layers.nnx.types import SequenceLayer
from sequence_layers.nnx.types import Shape
from sequence_layers.nnx.types import ShapeDType
from sequence_layers.nnx.types import ShapeLike
from sequence_layers.nnx.types import Sharding
from sequence_layers.nnx.types import State
from sequence_layers.nnx.types import Stateless
from sequence_layers.nnx.types import StatelessEmitting
from sequence_layers.nnx.types import StatelessPointwise
from sequence_layers.nnx.types import StatelessPointwiseFunctor
from sequence_layers.nnx.types import Steppable
from sequence_layers.nnx.types import ValuesT

# Simple layers.
from sequence_layers.nnx.simple import Abs
from sequence_layers.nnx.simple import Add
from sequence_layers.nnx.simple import Affine
from sequence_layers.nnx.simple import Argmax
from sequence_layers.nnx.simple import Cast
from sequence_layers.nnx.simple import Downsample1D
from sequence_layers.nnx.simple import Dropout
from sequence_layers.nnx.simple import EinopsRearrange
from sequence_layers.nnx.simple import Elu
from sequence_layers.nnx.simple import Embedding
from sequence_layers.nnx.simple import EmbeddingTranspose
from sequence_layers.nnx.simple import Emit
from sequence_layers.nnx.simple import ExpandDims
from sequence_layers.nnx.simple import Exp
from sequence_layers.nnx.simple import Flatten
from sequence_layers.nnx.simple import GatedLinearUnit
from sequence_layers.nnx.simple import GatedTanhUnit
from sequence_layers.nnx.simple import GatedUnit
from sequence_layers.nnx.simple import Gelu
from sequence_layers.nnx.simple import Identity
from sequence_layers.nnx.simple import Lambda
from sequence_layers.nnx.simple import LeakyRelu
from sequence_layers.nnx.simple import Log
from sequence_layers.nnx.simple import MaskInvalid
from sequence_layers.nnx.simple import Max
from sequence_layers.nnx.simple import Maximum
from sequence_layers.nnx.simple import Mean
from sequence_layers.nnx.simple import Min
from sequence_layers.nnx.simple import Minimum
from sequence_layers.nnx.simple import Mod
from sequence_layers.nnx.simple import MoveAxis
from sequence_layers.nnx.simple import NamedEmit
from sequence_layers.nnx.simple import OneHot
from sequence_layers.nnx.simple import Power
from sequence_layers.nnx.simple import PRelu
from sequence_layers.nnx.simple import Relu
from sequence_layers.nnx.simple import Reshape
from sequence_layers.nnx.simple import Scale
from sequence_layers.nnx.simple import Sigmoid
from sequence_layers.nnx.simple import Slice
from sequence_layers.nnx.simple import Snake
from sequence_layers.nnx.simple import Softmax
from sequence_layers.nnx.simple import Softplus
from sequence_layers.nnx.simple import Squeeze
from sequence_layers.nnx.simple import Sum
from sequence_layers.nnx.simple import SwapAxes
from sequence_layers.nnx.simple import Swish
from sequence_layers.nnx.simple import Tanh
from sequence_layers.nnx.simple import Transpose
from sequence_layers.nnx.simple import Upsample1D
from sequence_layers.nnx.simple import Upsample2D

# Dense layers.
from sequence_layers.nnx.dense import Dense
from sequence_layers.nnx.dense import DenseShaped

# Convolution layers.
from sequence_layers.nnx.convolution import Conv1D
from sequence_layers.nnx.convolution import Conv1DTranspose
from sequence_layers.nnx.convolution import Conv2D
from sequence_layers.nnx.convolution import Conv2DTranspose
from sequence_layers.nnx.convolution import Conv3D
from sequence_layers.nnx.convolution import DepthwiseConv1D

# Normalization layers.
from sequence_layers.nnx.normalization import BatchNormalization
from sequence_layers.nnx.normalization import GroupNormalization
from sequence_layers.nnx.normalization import L2Normalize
from sequence_layers.nnx.normalization import LayerNormalization
from sequence_layers.nnx.normalization import RMSNormalization

# Pooling layers.
from sequence_layers.nnx.pooling import AveragePooling1D
from sequence_layers.nnx.pooling import AveragePooling2D
from sequence_layers.nnx.pooling import AveragePooling3D
from sequence_layers.nnx.pooling import MaxPooling1D
from sequence_layers.nnx.pooling import MaxPooling2D
from sequence_layers.nnx.pooling import MaxPooling3D
from sequence_layers.nnx.pooling import MinPooling1D
from sequence_layers.nnx.pooling import MinPooling2D
from sequence_layers.nnx.pooling import MinPooling3D

# Recurrent layers.
from sequence_layers.nnx.recurrent import LSTM
from sequence_layers.nnx.recurrent import RGLRU

# Position layers.
from sequence_layers.nnx.position import AddTimingSignal
from sequence_layers.nnx.position import ApplyRotaryPositionalEncoding

# Attention layers.
from sequence_layers.nnx.attention import CrossAttentionEmits
from sequence_layers.nnx.attention import DotProductAttention
from sequence_layers.nnx.attention import DotProductSelfAttention
from sequence_layers.nnx.attention import SelfAttentionEmits

# Combinators.
from sequence_layers.nnx.combinators import Bidirectional
from sequence_layers.nnx.combinators import Blockwise
from sequence_layers.nnx.combinators import CombinationMode
from sequence_layers.nnx.combinators import Parallel
from sequence_layers.nnx.combinators import Residual
from sequence_layers.nnx.combinators import Serial
