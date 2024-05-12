from sru import SRU
from torch import nn


class ASRRNN(nn.Module):
    def __init__(
        self,
        input_channels,
        rnn_num_layers,
        rnn_hidden_size,
        rnn_bidirectional,
        rnn_input_size,
        res_reduce,
        n_tokens,
        **kwargs
    ) -> None:
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1), nn.ReLU()
        )
        self.rnn = SRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            bidirectional=rnn_bidirectional,
            num_layers=rnn_num_layers,
        )

        bi_coef = 2 if rnn_bidirectional else 1
        self.fc_part = nn.Linear(bi_coef * rnn_hidden_size, n_tokens)

        self.n_tokens = n_tokens
        self.res_reduce = res_reduce

    def forward(self, fused_feats, s_audio_length, aug=None):
        conv_output = self.pre_conv(fused_feats)  # B x C x F x T -> B x 1 x F x T

        conv_output = conv_output.squeeze(1).permute(2, 0, 1)  # T x B x F
        rnn_output = self.rnn(conv_output)[0]
        rnn_output = rnn_output.permute(1, 0, 2)  # B x T x F
        tokens_logits = self.fc_part(rnn_output)
        # tokens_logits = tokens_logits.view(tokens_logits.shape[0], -1, self.n_tokens)

        s_audio_length = (
            s_audio_length // self.res_reduce
        )  # we changed the resolution to x self.res_reduce lower

        return {"tokens_logits": tokens_logits, "s_audio_length": s_audio_length}
