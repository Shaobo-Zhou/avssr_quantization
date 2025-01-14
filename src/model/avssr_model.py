import torch
from torch import nn
import time
from src.utils.io_utils import ROOT_PATH


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
        train_ss_model=False,
        ss_pretrain_path=None,
        ss_teacher=None,  # AVSS teacher used for distillation on the fly
        ss_teacher_proj=None,
        asr_aug=None,
        skip_ASR=False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.ss_model = ss_model
        self.video_model = video_model
        self.asr_model = asr_model
        self.train_video_model = train_video_model
        self.train_ss_model = train_ss_model

        self.ss_teacher = ss_teacher
        self.ss_teacher_proj = None
        if ss_teacher_proj is not None:
            self.ss_teacher_proj = nn.Sequential(
                nn.Conv2d(ss_teacher_proj[0], ss_teacher_proj[1], kernel_size=1),
                nn.PReLU(),
            )

        self.asr_aug = asr_aug
        if ss_pretrain_path is not None:
            ss_pretrain_path = str(ROOT_PATH / "data" / "pretrain" / ss_pretrain_path)
            print(f"Loading SS weights from {ss_pretrain_path}...")
            self.ss_model.init_from(ss_pretrain_path)
        self.skip_ASR = skip_ASR

    def train(self, mode: bool = True):
        super().train(mode)

        # these models should always be in eval mode
        if not self.train_video_model:
            self.video_model.eval()

        if not self.train_ss_model:
            self.ss_model.eval()

        if self.ss_teacher is not None:
            self.ss_teacher.eval()

        return self

    def forward(self, mix_audio, s_video, s_audio_length, **batch):
        
        #start_time = time.time()
        if self.video_model is None:
            ss_batch = self.ss_model(mix_audio)
        else:
            if not self.train_video_model:
                with torch.no_grad():
                    mouth_emb = self.video_model(s_video)
            else:
                mouth_emb = self.video_model(s_video)
            if not self.train_ss_model:
                with torch.no_grad():
                    ss_batch = self.ss_model(mix_audio, mouth_emb)
            else:
                ss_batch = self.ss_model(mix_audio, mouth_emb)
        
        if self.skip_ASR == False:
            asr_batch = self.asr_model(
                ss_batch["fused_feats"], s_audio_length, self.asr_aug
            )  
            # join keys
            ss_batch.update(asr_batch)

        # teacher for KD
        if self.ss_teacher is not None:
            with torch.no_grad():
                teacher_ss_batch = self.ss_teacher(mix_audio, s_video)
                ss_batch["t_kd_embedding"] = teacher_ss_batch["kd_embedding"]
            if self.ss_teacher_proj is not None:
                ss_batch["kd_embedding"] = self.ss_teacher_proj(
                    ss_batch["kd_embedding"]
                )

        return ss_batch

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        full_params = sum([p.numel() for p in self.parameters()])

        if self.video_model is not None:
            video_params = sum([p.numel() for p in self.video_model.parameters()])
        else:
            video_params = 0

        asr_params = sum([p.numel() for p in self.asr_model.parameters()])

        if self.ss_teacher is not None:
            teacher_params = sum([p.numel() for p in self.ss_teacher.parameters()])
            full_params = full_params - teacher_params

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
