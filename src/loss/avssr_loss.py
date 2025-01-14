import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from src.loss.asr_loss import ASRLoss
from src.loss.ss_loss import PairwiseNegSDR


class AVSSRLoss(_Loss):
    def __init__(self, asr_coef=1, ss_coef=1, kd_coef=0):
        super().__init__()

        self.asr_loss = ASRLoss()
        self.train_ss_loss = PairwiseNegSDR(sdr_type="snr")
        self.inference_ss_loss = PairwiseNegSDR(sdr_type="sisdr")
        self.kd_loss = nn.MSELoss()

        self.asr_coef = asr_coef
        self.ss_coef = ss_coef
        self.kd_coef = kd_coef

        self.ss_mode = "train"

    def set_ss_mode(self, mode):
        self.ss_mode = mode

    def forward(self, **batch):
        asr_loss = self.asr_loss(**batch)

        if self.ss_mode == "train":
            ss_loss = self.train_ss_loss(**batch)
        else:
            ss_loss = self.inference_ss_loss(**batch)

        loss = self.asr_coef * asr_loss + self.ss_coef * ss_loss

        if self.kd_coef > 0:
            kd_loss = self.kd_loss(batch["kd_embedding"], batch["t_kd_embedding"])
            loss = loss + self.kd_coef * kd_loss
        else:
            kd_loss = torch.zeros(1)

        return {
            "loss": loss,
            "asr_loss": asr_loss,
            "ss_loss": ss_loss,
            "kd_loss": kd_loss,
        }
