from torch import nn


class ASRExample(nn.Module):
    def __init__(
        self, n_input_feats, n_audio_feats, fc_hidden, n_tokens, **kwargs
    ) -> None:
        super().__init__()
        self.asr_net = nn.Sequential(
            nn.Linear(n_input_feats, fc_hidden),
            nn.ReLU(),
            nn.Linear(
                fc_hidden, n_audio_feats * n_tokens // 128
            ),  # decrease resolution
        )

        self.n_tokens = n_tokens

    def forward(self, fused_feats, s_audio_length):
        tokens_logits = self.asr_net(fused_feats)
        tokens_logits = tokens_logits.view(tokens_logits.shape[0], -1, self.n_tokens)

        s_audio_length = (
            s_audio_length // 128
        )  # we changed the resolution to x128 lower

        return tokens_logits, s_audio_length
