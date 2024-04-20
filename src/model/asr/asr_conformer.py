import torch
from torch import nn

from src.model.asr.conformer import ConformerEncoder


class ASRConformer(nn.Module):
    def __init__(
        self,
        res_reduce,
        n_tokens,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_encoder_layers: int = 17,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
        do_subsample: bool = True,
        subsampling_factor: int = 4,
        pre_conv_channels: int = 0,
        **kwargs
    ) -> None:
        super().__init__()

        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            do_subsample=do_subsample,
            subsampling_factor=subsampling_factor,
        )

        self.fc = nn.Linear(encoder_dim, n_tokens)

        if pre_conv_channels != 0:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(pre_conv_channels, 1, 1), nn.PReLU()
            )
        else:
            self.pre_conv = None

        self.n_tokens = n_tokens
        self.res_reduce = res_reduce

    def forward(self, fused_feats, s_audio_length):
        if self.pre_conv is not None:
            fused_feats = self.pre_conv(fused_feats)  # -> B x 1 x C x T
        fused_feats = fused_feats.squeeze(1).transpose(1, 2)  # B x T x C
        encoder_output, s_audio_length = self.encoder(
            fused_feats, (s_audio_length // self.res_reduce).to(torch.int32)
        )
        tokens_logits = self.fc(encoder_output)
        # tokens_logits = tokens_logits.view(tokens_logits.shape[0], -1, self.n_tokens)

        return {"tokens_logits": tokens_logits, "s_audio_length": s_audio_length}
