import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from src.loss.asr_loss import ASRLoss
from src.loss.ss_loss import PairwiseNegSDR


class AVSSRLoss(_Loss):
    def __init__(self, asr_coef=1, ss_coef=1):
        super().__init__()

        self.asr_loss = ASRLoss()
        self.ss_loss = PairwiseNegSDR(sdr_type="snr")

        self.asr_coef = 1
        self.ss_coef = 1

    def forward(self, **batch):
        asr_loss = self.asr_loss(**batch)
        ss_loss = self.ss_loss(**batch)

        loss = self.asr_coef * asr_loss + self.ss_coef * ss_loss

        return {"loss": loss, "asr_loss": asr_loss, "ss_loss": ss_loss}
