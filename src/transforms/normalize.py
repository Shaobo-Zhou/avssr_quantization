from torch import nn


class Normalize(nn.Module):
    """
    Batch-version of Normalize
    """

    def __init__(self, mean, std):
        super().__init__()

        self.mean = mean
        self.std = std

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x
