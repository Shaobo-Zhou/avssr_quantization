import torch
from torchmetrics.text import CharErrorRate, WordErrorRate

from src.metrics.base_metric import BaseMetric


class ASRMetric(BaseMetric):
    def __init__(self, metric_type, device, text_encoder, decode_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if metric_type == "CER":
            self.metric = CharErrorRate().to(device)
        elif metric_type == "WER":
            self.metric = WordErrorRate().to(device)
        else:
            raise NotImplementedError()
        
        self.text_encoder = text_encoder
        self.decode_type = decode_type

        if self.decode_type == "beam_search" or self.decode_type == "lm_beam_search":
            self.beam_size = kwargs["beam_size"]

    @torch.no_grad()
    def __call__(self, tokens_logits: torch.Tensor, 
                 s_audio_length: torch.Tensor, s_text: list[str], **kwargs):
        if self.decode_type == "argmax":
            text_list = self.text_encoder.ctc_argmax(tokens_logits, s_audio_length)
        elif self.decode_type == "beam_search":
            text_list = self.text_encoder.ctc_beam_search(tokens_logits, s_audio_length,
                                                          beam_size=self.beam_size, use_lm=False)
        elif self.decode_type == "lm_beam_search":
            text_list = self.text_encoder.ctc_beam_search(tokens_logits, s_audio_length,
                                                          beam_size=self.beam_size, use_lm=True)

        return self.metric(text_list, s_text)
    