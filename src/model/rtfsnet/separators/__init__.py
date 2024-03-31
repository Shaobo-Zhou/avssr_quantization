import torch.nn as nn

from src.model.rtfsnet.separators.dpt import DPTNet
from src.model.rtfsnet.separators.frcnn import FRCNN
from src.model.rtfsnet.separators.tdanet import TDANet


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
