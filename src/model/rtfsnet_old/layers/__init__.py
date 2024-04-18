import torch.nn as nn

from .attention import (
    CBAMBlock,
    CoTAttention,
    GlobalAttention,
    GlobalAttention2D,
    MultiHeadSelfAttention,
    MultiHeadSelfAttention2D,
    ShuffleAttention,
)
from .conv_layers import (
    ConvActNorm,
    ConvNormAct,
    ConvolutionalRNN,
    DepthwiseSeparableConvolution,
    FeedForwardNetwork,
)
from .fusion import (
    ATTNFusionCell,
    ConvGRUFusionCell,
    ConvLSTMFusionCell,
    InjectionMultiSum,
)
from .mlp import MLP
from .permutator import Permutator
from .rnn_layers import (
    BiLSTM2D,
    DualPathRNN,
    GlobalAttentionRNN,
    GlobalGALR,
    RNNProjection,
)


def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)

        if cls is None:
            raise ValueError(
                "Could not interpret normalization identifier: " + str(identifier)
            )
        return cls
    else:
        raise ValueError(
            "Could not interpret normalization identifier: " + str(identifier)
        )
