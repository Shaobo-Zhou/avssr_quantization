import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, SignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class SSMetric(BaseMetric):
    def __init__(self, metric_type, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if metric_type == "SDRi":
            self.metric = SignalDistortionRatio().to(device)
        elif metric_type == "SI-SNRi":
            self.metric = ScaleInvariantSignalNoiseRatio().to(device)
        elif metric_type == "PESQ":
            self.metric = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb").to(
                device
            )
        elif metric_type == "STOI":
            self.metric = ShortTimeObjectiveIntelligibility(fs=16000).to(device)
        else:
            raise NotImplementedError()

        self.metric_type = metric_type

    @torch.no_grad()
    def __call__(
        self,
        s_audio: torch.Tensor,
        mix_audio: torch.Tensor,
        predicted_audio: torch.Tensor,
        **kwargs
    ):
        if self.metric_type == "PESQ" or self.metric_type == "STOI":
            return self.metric(predicted_audio, s_audio)
        else:
            metric_mix = self.metric(mix_audio, s_audio)
            metric_pred = self.metric(predicted_audio, s_audio)

            improvement = metric_pred - metric_mix

            return improvement
