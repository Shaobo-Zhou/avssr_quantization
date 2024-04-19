from torch import nn


class ASRExample(nn.Module):
    def __init__(
        self,
        input_channels,
        n_repeats,
        n_input_feats,
        n_hidden_channels,
        res_reduce,
        n_tokens,
        **kwargs
    ) -> None:
        super().__init__()
        if n_repeats == 1:
            n_hidden_channels = 1
        conv_part = [
            nn.Conv2d(input_channels, n_hidden_channels, 4, 2, 1),
            nn.ReLU(),
        ]
        for _ in range(n_repeats - 2):
            conv_part.extend(
                [nn.Conv2d(n_hidden_channels, n_hidden_channels, 4, 2, 1), nn.ReLU()]
            )
        if n_repeats >= 2:
            conv_part.extend(
                [
                    nn.Conv2d(n_hidden_channels, 1, 4, 2, 1),
                    nn.ReLU(),
                ]
            )
        self.conv_part = nn.Sequential(*conv_part)
        self.fc_part = nn.Linear(n_input_feats, n_tokens)

        self.n_tokens = n_tokens
        self.res_reduce = res_reduce

    def forward(self, fused_feats, s_audio_length):
        conv_output = self.conv_part(fused_feats)  # B x C x F x T -> B x 1 x F x T

        conv_output = conv_output.squeeze(1).transpose(1, 2)  # B x T x C
        tokens_logits = self.fc_part(conv_output)
        # tokens_logits = tokens_logits.view(tokens_logits.shape[0], -1, self.n_tokens)

        s_audio_length = (
            s_audio_length // self.res_reduce
        )  # we changed the resolution to x self.res_reduce lower

        return {"tokens_logits": tokens_logits, "s_audio_length": s_audio_length}
