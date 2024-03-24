import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class ASRLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.loss = nn.CTCLoss(zero_infinity=True)

    def forward(
        self,
        tokens_logits: torch.Tensor,
        s_tokens: torch.Tensor,
        s_tokens_length: torch.Tensor,
        s_audio_length: torch.Tensor,
        **batch
    ):
        log_probs = nn.functional.log_softmax(tokens_logits, dim=-1)

        log_probs = log_probs.transpose(0, 1)  # time first

        loss = self.loss(
            log_probs=log_probs,
            targets=s_tokens,
            input_lengths=s_audio_length,
            target_lengths=s_tokens_length,
        )

        return loss
