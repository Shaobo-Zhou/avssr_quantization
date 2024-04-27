import argparse
import os
from collections import defaultdict
from pathlib import Path

import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, SignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.text import CharErrorRate, WordErrorRate
from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent
DATA_PATH = ROOT_PATH / "data" / "saved"


@torch.no_grad()
def get_metrics_from_file(
    file,
    metric_sdr,
    metric_si_snr,
    metric_pesq,
    metric_stoi,
    metric_cer,
    metric_wer,
    device,
):
    data = torch.load(file)

    s_audio = data["s_audio"].to(device)
    mix_audio = data["mix_audio"].to(device)
    predicted_audio = data["predicted_audio"].to(device)
    s_text = data["s_text"]
    argmax_text = data["argmax_text"]
    bs_text = data["bs_text"]
    lm_text = data["lm_text"]

    metric_mix = metric_sdr(mix_audio, s_audio)
    metric_pred = metric_sdr(predicted_audio, s_audio)
    sdr_i = metric_pred - metric_mix

    metric_mix = metric_si_snr(mix_audio, s_audio)
    metric_pred = metric_si_snr(predicted_audio, s_audio)
    si_snr_i = metric_pred - metric_mix

    pesq = metric_pesq(predicted_audio, s_audio)
    stoi = metric_stoi(predicted_audio, s_audio)

    argmax_cer = metric_cer(argmax_text, s_text)
    argmax_wer = metric_wer(argmax_text, s_text)

    bs_cer = metric_cer(bs_text, s_text)
    bs_wer = metric_wer(bs_text, s_text)

    lm_cer = metric_cer(lm_text, s_text)
    lm_wer = metric_wer(lm_text, s_text)

    return {
        "SDRi": sdr_i,
        "SI-SNRi": si_snr_i,
        "PESQ": pesq,
        "STOI": stoi,
        "CER (Argmax)": argmax_cer,
        "WER (Argmax)": argmax_wer,
        "CER (BS)": bs_cer,
        "WER (BS)": bs_wer,
        "CER (LM)": lm_cer,
        "WER (LM)": lm_wer,
    }


def calculate_metrics(dataset_name, device):
    data_path = DATA_PATH / dataset_name

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # define metrics
    metric_sdr = SignalDistortionRatio().to(device)
    metric_si_snr = ScaleInvariantSignalNoiseRatio().to(device)
    metric_pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb").to(device)
    metric_stoi = ShortTimeObjectiveIntelligibility(fs=16000).to(device)
    metric_cer = CharErrorRate().to(device)
    metric_wer = WordErrorRate().to(device)

    splits = os.listdir(str(data_path))

    splits = filter(lambda x: not x.endswith("pth"), splits)
    for split in splits:
        print(f"Calculating metrics for split: {split} ...")

        split_path = data_path / split

        result_metrics = defaultdict(float)
        amount = 0
        for filename in tqdm(os.listdir(str(split_path))):
            file_path = split_path / filename
            file_metrics = get_metrics_from_file(
                file_path,
                metric_sdr=metric_sdr,
                metric_si_snr=metric_si_snr,
                metric_pesq=metric_pesq,
                metric_stoi=metric_stoi,
                metric_cer=metric_cer,
                metric_wer=metric_wer,
                device=device,
            )
            for k, v in file_metrics.items():
                result_metrics[k] += v
            amount += 1

        for k, v in result_metrics.items():
            result_metrics[k] = v / amount
            print(f"Metric: {k}, Value: {result_metrics[k]}")

        torch.save(result_metrics, data_path / f"{split}_metrics.pth")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Calculate all metrics on a given saved dataset"
    )
    args.add_argument(
        "-d",
        "--dataset_name",
        default=None,
        type=str,
        help="Dataset name inside data dir (default: None)",
    )
    args.add_argument(
        "--device",
        default="auto",
        type=str,
        help="Device: cpu, cuda or auto (default: auto)",
    )

    args = args.parse_args()

    calculate_metrics(args.dataset_name, args.device)
