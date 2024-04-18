import torch
from torch import nn

from src.utils.io_utils import ROOT_PATH


class AVSSModel(nn.Module):
    """
    General class for AV Source Separation Models

    Used for KD teachers
    """

    def __init__(
        self,
        ss_model,  # CTCNet or RTFSNet
        video_model,  # ResNet18 or ShuffleNet
        train_video_model=False,
        ss_pretrain_path=None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.ss_model = ss_model
        self.video_model = video_model
        self.train_video_model = train_video_model

        if ss_pretrain_path is not None:
            ss_pretrain_path = str(ROOT_PATH / "data" / "pretrain" / ss_pretrain_path)
            print(f"Loading SS weights from {ss_pretrain_path}...")
            self.ss_model.init_from(ss_pretrain_path)

    def forward(self, mix_audio, s_video, **batch):
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

        return ss_batch

    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """

    #     full_params = sum([p.numel() for p in self.parameters()])
    #     video_params = sum([p.numel() for p in self.video_model.parameters()])
    #     ss_params = full_params - video_params

    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([p.numel() for p in model_parameters])

    #     result_str = super().__str__()
    #     result_str = result_str + "\nAll parameters: {}".format(full_params)
    #     result_str = result_str + "\nVideo parameters: {}".format(video_params)
    #     result_str = result_str + "\nSS parameters: {}".format(ss_params)
    #     result_str = result_str + "\nTrainable parameters: {}".format(params)

    #     return result_str
