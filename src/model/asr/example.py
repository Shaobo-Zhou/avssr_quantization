from torch import nn


class ASRExample(nn.Module):
    def __init__(
        self, n_input_feats, n_hidden_channels, res_reduce, n_tokens, **kwargs
    ) -> None:
        super().__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(1, n_hidden_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_hidden_channels, n_hidden_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_hidden_channels, n_hidden_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_hidden_channels, n_hidden_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_hidden_channels, 1, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc_part = nn.Linear(n_input_feats, n_tokens)

        self.n_tokens = n_tokens
        self.res_reduce = res_reduce

    def forward(self, fused_feats, s_audio_length):
        conv_output = self.conv_part(fused_feats)

        conv_output = conv_output.squeeze(1).transpose(1, 2)
        tokens_logits = self.fc_part(conv_output)
        # tokens_logits = tokens_logits.view(tokens_logits.shape[0], -1, self.n_tokens)

        s_audio_length = (
            s_audio_length // self.res_reduce
        )  # we changed the resolution to x self.res_reduce lower

        return tokens_logits, s_audio_length
