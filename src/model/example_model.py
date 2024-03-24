import numpy as np
import torch
from torch import nn
from torch.nn import Sequential


class ExampleModel(nn.Module):
    def __init__(self, n_audio_feats, n_video_feats, n_video_channels, n_hidden_channels,
                 n_tokens, fc_hidden=512):
        super().__init__()
        self.audio_net = Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_hidden_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_hidden_channels, out_channels=n_hidden_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_hidden_channels, out_channels=1,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=n_audio_feats // 8,
                      out_features=fc_hidden),
        )
        self.video_net = Sequential(
            nn.Conv2d(in_channels=n_video_channels, out_channels=n_hidden_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_hidden_channels, out_channels=n_hidden_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_hidden_channels, out_channels=1,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=(n_video_feats // 8) ** 2,
                      out_features=fc_hidden),
        )

        self.decoder = Sequential(
            nn.Linear(in_features=2 * fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_audio_feats),
        )

        self.ctc_net = Sequential(
            nn.Linear(2 * fc_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_audio_feats * n_tokens // 128) # decrease resolution
        )

        self.n_tokens = n_tokens


    def forward(self, mix_audio, s_video, s_audio_length, **batch):
        audio_feats = self.audio_net(mix_audio.unsqueeze(1))
        video_feats = self.video_net(s_video)

        feats = torch.cat([audio_feats, video_feats], dim=-1)

        predicted_audio = self.decoder(feats)
        tokens_logits = self.ctc_net(feats)

        tokens_logits = tokens_logits.view(tokens_logits.shape[0], -1, self.n_tokens)

        return {
            "predicted_audio": predicted_audio,
            "tokens_logits": tokens_logits,
            "s_audio_length": s_audio_length // 128, # we changed the resolution to x128 lower
        }


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
