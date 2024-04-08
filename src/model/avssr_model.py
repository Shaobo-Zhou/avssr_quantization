import torch
from torch import nn


class AVSSRModel(nn.Module):
    """
    General class for AVSSR Models
    """

    def __init__(
        self,
        ss_model,  # CTCNet or RTFSNet
        video_model,  # ResNet18 or ShuffleNet
        asr_model,  # model from src.model.asr
        train_video_model=False,
        **kwargs
    ) -> None:
        super().__init__()

        self.ss_model = ss_model
        self.video_model = video_model
        self.asr_model = asr_model
        self.train_video_model = train_video_model

    def forward(self, mix_audio, s_video, s_audio_length, **batch):
        # get predicted_audio, fused_features, etc.
        if self.video_model is None:
            ss_batch = self.ss_model(mix_audio)
        else:
            if not self.train_video_model:
                with torch.no_grad():
                    mouth_emb = self.video_model(s_video)
            else:
                mouth_emb = self.video_model(s_video)
            ss_batch = self.ss_model(mix_audio, mouth_emb)

        # get tokens_logits, s_audio_length
        asr_batch = self.asr_model(ss_batch["fused_feats"], s_audio_length)

        # join keys
        ss_batch.update(asr_batch)

        return ss_batch

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        full_params = sum([p.numel() for p in self.parameters()])
        video_params = sum([p.numel() for p in self.video_model.parameters()])
        asr_params = sum([p.numel() for p in self.asr_model.parameters()])
        ss_params = full_params - video_params - asr_params

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([p.numel() for p in model_parameters])

        result_str = super().__str__()
        result_str = result_str + "\nAll parameters: {}".format(full_params)
        result_str = result_str + "\nVideo parameters: {}".format(video_params)
        result_str = result_str + "\nASR parameters: {}".format(asr_params)
        result_str = result_str + "\nSS parameters: {}".format(ss_params)
        result_str = result_str + "\nTrainable parameters: {}".format(params)

        return result_str
