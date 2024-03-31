import torch.nn as nn

from src.model.rtfsnet.layers.attention import (
    CBAMBlock,
    CoTAttention,
    GlobalAttention,
    GlobalAttention2D,
    MultiHeadSelfAttention,
    MultiHeadSelfAttention2D,
    ShuffleAttention,
)
from src.model.rtfsnet.layers.conv_layers import (
    ConvActNorm,
    ConvNormAct,
    ConvolutionalRNN,
    DepthwiseSeparableConvolution,
    FeedForwardNetwork,
)
from src.model.rtfsnet.layers.fusion import (
    ATTNFusionCell,
    ConvGRUFusionCell,
    ConvLSTMFusionCell,
    InjectionMultiSum,
)
from src.model.rtfsnet.layers.mlp import MLP
from src.model.rtfsnet.layers.permutator import Permutator
from src.model.rtfsnet.layers.rnn_layers import (
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
