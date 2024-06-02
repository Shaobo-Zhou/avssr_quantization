import nemo.collections.asr as nemo_asr
import torch
from torch import nn


class ASRNemo(nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=model_name
        )

    def forward(self, audio, s_audio_length, asr_aug=None, **batch):
        outputs = self.asr_model(input_signal=audio, input_signal_length=s_audio_length)

        tokens_logits = outputs[0]
        s_audio_length = outputs[1]

        return {"tokens_logits": tokens_logits, "s_audio_length": s_audio_length}
