import random
import warnings
import os
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.ao.quantization.observer import *
from torch.ao.quantization.qconfig import QConfig
from src.trainer import Inferencer
from src.utils.data_utils import *
from quantization import *
from src.utils.io_utils import ROOT_PATH
from quant_utils.quantization import *
from quant_utils.equalization import *
from nemo_new.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
import argparse
import pickle

warnings.filterwarnings("ignore", category=UserWarning)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_quantization_config(quant_config_name, qscheme, percentile=None):
    with initialize(config_path="src/configs"):
        quant_config = compose(config_name=quant_config_name).get("quantization", {})

    scheme = torch.per_tensor_symmetric if qscheme == "symmetric" else torch.per_tensor_affine

    activation_args = HistogramObserver.with_args(bins=2048, qscheme=scheme)
    if percentile is not None:
        activation_args = PercentileHistogramObserver.with_args(percentile=percentile, qscheme=scheme)

    custom_qconfig = {
        "feedforward": QConfig(activation=activation_args, weight=HistogramObserver.with_args(bins=2048, dtype=torch.qint8, qscheme=scheme)),
        "conv": QConfig(activation=activation_args, weight=HistogramObserver.with_args(bins=2048, dtype=torch.qint8, qscheme=scheme)),
        "attn": QConfig(activation=activation_args, weight=HistogramObserver.with_args(bins=2048, dtype=torch.qint8, qscheme=scheme)),
        "pre_encode": QConfig(activation=activation_args, weight=HistogramObserver.with_args(bins=2048, dtype=torch.qint8, qscheme=scheme)),
        "layer_norm": QConfig(activation=activation_args, weight=HistogramObserver.with_args(bins=2048, dtype=torch.qint8, qscheme=scheme))
    }

    return quant_config, custom_qconfig

@hydra.main(version_base=None, config_path="src/configs", config_name="inference_orig")
def main(config):
    parser = argparse.ArgumentParser(description="Quantization Setup")
    parser.add_argument("--quant_config_name", type=str, required=True, help="Name of the quantization configuration.")
    parser.add_argument("--percentile", type=float, default=None, help="Percentile for quantization.")
    parser.add_argument("--qscheme", type=str, choices=["symmetric", "affine"], default="symmetric",
                        help="Quantization scheme: 'symmetric' or 'affine'.")
    parser.add_argument("--equalization", action="store_true", help="Flag to apply weight-activation equalization.")
    parser.add_argument(
    "--use_SS_outputs",
    action="store_true",
    help="Specify whether to use SS outputs. Default is False.",)

    args = parser.parse_args()

    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device
    print(f'Device is {device}')

    # Get Quantization Configurations
    quant_config, custom_qconfig = get_quantization_config(args.quant_config_name, args.qscheme, args.percentile)
    dynamic_modules = quant_config.get("dynamic_modules", [])
    static_modules = quant_config.get("static_modules", [])

    # Prepare Model
    text_encoder = instantiate(config.text_encoder)
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder)

    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    preprocessor = model.asr_model.asr_model.preprocessor
    cfg_dict = OmegaConf.to_container(preprocessor._cfg, resolve=True)
    cfg_dict.pop('_target_', None)
    preprocessor = AudioToMelSpectrogramPreprocessor(**cfg_dict)
    model.asr_model.asr_model.preprocessor = preprocessor

    calibration_samples = get_calibration_samples(config, text_encoder, 1)

    if args.equalization:
        print("Applying weight-activation equalization.")
        activation_collector = ActivationCollector(model=model)

        # Specify the layers for which to collect input activations (only applicable for linear layers)
        layer_keys = [
            f"encoder.layers.{i}.feed_forward1.linear1" for i in range(16)
        ] + [
            f"encoder.layers.{i}.feed_forward1.linear2" for i in range(16)
        ] + [
            f"encoder.layers.{i}.feed_forward2.linear1" for i in range(16)
        ] + [
            f"encoder.layers.{i}.feed_forward2.linear2" for i in range(16)
        ] + [
            f"encoder.layers.{i}.self_attn.linear_out" for i in range(16)
        ] + [
            f"encoder.layers.{i}.self_attn.linear_pos" for i in range(16)
        ] + [
            f"encoder.layers.{i}.self_attn.linear_q" for i in range(16)
        ] + [
            f"encoder.layers.{i}.self_attn.linear_k" for i in range(16)
        ] + [
            f"encoder.layers.{i}.self_attn.linear_v" for i in range(16)
        ]

        # Register hooks
        activation_collector.register_hooks(layer_keys)

        # Run calibration data to collect activations
        activation_collector.collect_activations(calibration_samples=calibration_samples, device=device)

        # Retrieve the collected activation statistics
        activation_stats = activation_collector.get_activation_stats()

        # Save the activation statistics to a file
        with open("activation_stats.pkl", "wb") as f:
            pickle.dump(activation_stats, f)
        print("Activation statistics saved to 'activation_stats.pkl'")

        alpha = 0.5
        equalizer = OfflineModelEqualizer(model=model.asr_model, activation_stats=activation_stats, alpha=alpha)
        equalizer.apply_equalization()
        activation_collector.remove_hooks()

    # Apply Quantization
    model = QuantizedASRModel(model, dynamic_modules=dynamic_modules, static_modules=static_modules, qconfig_dict=custom_qconfig, calibration_samples=calibration_samples, device=device)

    quant_checkpoint = quant_config.get("checkpoint_path", None)
    if quant_checkpoint is not None and os.path.exists(quant_checkpoint):
        state_dict = torch.load(quant_checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded quantized model from {quant_checkpoint}")
    else:
        torch.save(model.state_dict(), quant_checkpoint)
        print(f"Quantized model saved at {quant_checkpoint}")

    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        quantization=True,
        SS_targets=args.use_SS_outputs,
        save_path=save_path,
        load_path=ROOT_PATH / "data" / "saved" / "lrs2_SS",
        text_encoder=text_encoder,
    )
    inferencer.run_inference(max_samples=None)

if __name__ == "__main__":
    main()