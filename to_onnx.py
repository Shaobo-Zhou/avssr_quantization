import random
import warnings
import os
import hydra
import argparse
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.ao.quantization.observer import *
from torch.ao.quantization.qconfig import QConfig
from src.utils.data_utils import *
from quantization import *
from quant_utils.quantization import *
from quant_utils.equalization import *
from nemo_new.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor

warnings.filterwarnings("ignore", category=UserWarning)


def set_random_seed(seed):
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class NormOut(nn.Module):
    def __init__(self, original_layer):
        super(NormOut, self).__init__()
        self.norm_out = original_layer.norm_out
        self.dropout = original_layer.dropout
        self.fc_factor = original_layer.fc_factor
    def forward(self, x):
        residual = x
        residual = residual + self.dropout(x) * self.fc_factor
        x = self.norm_out(residual)

        return x

class IdentityLinear(nn.Module):
    def __init__(self, input_dim, fac=1):
        super(IdentityLinear, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=False)
        # Initialize weights to identity matrix
        with torch.no_grad():
            self.linear.weight.copy_(fac*torch.eye(input_dim))

    def forward(self, x):
        return self.linear(x)
    

### Identity layers are inserted to avoid TensorRT run PointwiseFusion, which causes an segmentation fault 
class ManualLayerNorm(torch.nn.Module):
    def __init__(self, original_layer_norm):
        super(ManualLayerNorm, self).__init__()
        # Copy weights and bias from the original LayerNorm
        self.weight = torch.nn.Parameter(original_layer_norm.weight.clone())
        self.bias = torch.nn.Parameter(original_layer_norm.bias.clone())
        self.eps = original_layer_norm.eps  # Keep the same epsilon value
        self.identity1 = IdentityLinear(176, fac=2)
        self.identity2 = IdentityLinear(176, fac=0.5)
        self.identity3 = IdentityLinear(176)


    def forward(self, x):
        # Compute mean and std along the last dimension while broadcasting to preserve shape
        mean = x.mean(dim=-1, keepdim=True)  # Shape: [1, 26, 1]
        std = x.std(dim=-1, keepdim=True)    # Shape: [1, 26, 1]

        # Broadcast mean and std to match the original tensor shape
        mean = mean.expand_as(x)            # Shape: [1, 26, 176]
        std = std.expand_as(x)              # Shape: [1, 26, 176]

        # Subtract mean and apply first identity
        x = x - mean
        x = self.identity1(x)

        # Apply weights and the second identity
        x = x * self.weight
        x = self.identity2(x)

        # Apply standard deviation, epsilon adjustment, and identities
        std = self.identity3(std)
        std = std + self.eps
        std = self.identity3(std)
        x = x / std
        x = self.identity3(x)

        # Add bias
        return x + self.bias

    
class FF(nn.Module):
    def __init__(self, original_ffn):
        super(FF, self).__init__()

        # Copy original modules
        self.quant = original_ffn.quant
        self.dequant = original_ffn.dequant
        self.linear1 = original_ffn.linear1
        self.sigmoid = nn.Sigmoid()
        self.dropout = original_ffn.dropout
        self.linear2 = original_ffn.linear2
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.quant(x)
        x = self.linear1(x)

        x = self.dequant(x)
        x = x * self.sigmoid(x)  ### Compute Swish manually instead of nn.Swish
        x = self.quant(x) 

        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dequant(x)  
        
        return x


class Manual_Layer(nn.Module):
    def __init__(self, original_layer):
        super(Manual_Layer, self).__init__()
        self.norm_feed_forward1 = original_layer.norm_feed_forward1
        self.feed_forward1 = original_layer.feed_forward1
        self.norm_self_att = original_layer.norm_self_att
        self.self_attn = original_layer.self_attn
        self.norm_conv = original_layer.norm_conv
        self.conv = original_layer.conv
        self.norm_feed_forward2 = original_layer.norm_feed_forward2
        self.feed_forward2 = FF(original_layer.feed_forward2)
        self.norm_out = original_layer.norm_out
        self.dropout = original_layer.dropout
        self.fc_factor = original_layer.fc_factor
    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        residual=x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor
        x = self.norm_self_att(residual)
        x = self.self_attn(x,x,x,None,pos_emb=pos_emb)
        residual = residual + self.dropout(x)
        x = self.norm_conv(residual)
        x = self.conv(x)
        residual = residual + self.dropout(x)
        x = self.norm_feed_forward2(residual)

        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)
        

        return x

class Manual_Layer1(nn.Module):
    def __init__(self, original_layer):
        super(Manual_Layer1, self).__init__()
        self.norm_feed_forward1 = original_layer.norm_feed_forward1
        self.feed_forward1 = original_layer.feed_forward1
        self.norm_self_att = original_layer.norm_self_att
        self.self_attn = original_layer.self_attn
        self.norm_conv = original_layer.norm_conv
        self.conv = original_layer.conv
        self.norm_feed_forward2 =  ManualLayerNorm(original_layer.norm_feed_forward2)
        self.feed_forward2 = FF(original_layer.feed_forward2)
        #self.norm_out = original_layer.norm_out
        self.norm_out = ManualLayerNorm(original_layer.norm_out)
        self.dropout = original_layer.dropout
        self.fc_factor = original_layer.fc_factor
        
    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        residual=x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor
        x = self.norm_self_att(residual)
        x = self.self_attn(x,x,x,None,pos_emb=pos_emb)
        residual = residual + self.dropout(x)
        x = self.norm_conv(residual)
        x = self.conv(x)
        residual = residual + self.dropout(x)
        x = self.norm_feed_forward2(residual)

        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)
        """ x = self.identity1(x)
        x += 1e-9
        x = self.identity2(x) """
        return x

class SS_Wrapper(nn.Module):
    def __init__(self, model):
        super(SS_Wrapper, self).__init__()
        self.model = model

    def forward(self, mix_audio, s_video, s_audio_length):
        if self.model.model.video_model is None:
            ss_batch = self.model.model.ss_model(mix_audio)
        else:
            mouth_emb = self.model.model.video_model(s_video)
            ss_batch = self.model.model.ss_model(mix_audio, mouth_emb)
        outputs = self.model.asr_model(
            input_signal=ss_batch["fused_feats"], input_signal_length=s_audio_length
        )
        tokens_logits, s_audio_length = outputs[0], outputs[1]
        ss_batch["tokens_logits"] = tokens_logits
        ss_batch["s_audio_length"] = s_audio_length

        # Flatten the dict into keys and values for tracing
        keys = list(ss_batch.keys())
        values = [ss_batch[key] for key in keys]
        return values
    
class ASR_Wrapper(nn.Module):
    def __init__(self, asr_model):
        super(ASR_Wrapper, self).__init__()
        self.asr_model = asr_model
    def forward(self, audio, s_audio_length):
        outputs = self.asr_model(input_signal=audio, input_signal_length=s_audio_length)

        tokens_logits = outputs[0]
        s_audio_length = outputs[1]

        return tokens_logits, s_audio_length
    

"""
Use this if you have TensorRT < 8.6, as the STFT operations in SS and ASR.preprocessor are not supported.
Manual implementation of the original conformer encoder as workaround as otherwise a segmentation fault is caused during TensorRT optimization
Involving manual implementation of Swish, adding identity layers in norm_out to prevent TensorRT applying PointwiseFusion
"""

class ASR_Nopre_Wrapper(nn.Module):
    def __init__(self, asr_model):
        super(ASR_Nopre_Wrapper, self).__init__()
        self.asr_model = asr_model
        self.asr_encoder = self.asr_model.encoder
        self.asr_decoder = self.asr_model.decoder

        # Replace each encoder layer with Manual_Layer
        new_layers = []
        for i, layer in enumerate(self.asr_encoder.layers):
            # Wrap each encoder layer with Manual_Layer
            if i < 15:
                new_layer = Manual_Layer(layer)
            else:
                new_layer = Manual_Layer1(layer) ###Identity layers need to be inserted only for the last layer
            new_layers.append(new_layer)

        # Replace the original encoder's layers with the new layers
        self.asr_encoder.layers = nn.ModuleList(new_layers)

    def forward(self, preprocessed_feats, s_audio_length):
        # Run the preprocessed features through the ASR encoder
        encoder_output = self.asr_encoder(audio_signal=preprocessed_feats, length=s_audio_length)
        
        # Run the encoder output through the decoder
        decoder_output = self.asr_decoder(encoder_output=encoder_output[0])
        
        # Extract tokens logits from decoder output for CTC decoding
        tokens_logits = decoder_output
        
        return tokens_logits

def get_quantization_config(quant_config_name, qscheme, percentile=None):
    with initialize(config_path="src/configs"):
        quant_config = compose(config_name=quant_config_name).get("quantization", {})

    scheme = torch.per_tensor_symmetric if qscheme == "symmetric" else torch.per_tensor_affine

    activation_args = HistogramObserver.with_args(bins=2048, qscheme=scheme)
    if percentile is not None:
        activation_args = HistogramObserver.with_args(bins=2048, qscheme=scheme, percentile=percentile)

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
    parser = argparse.ArgumentParser(description="Quantized Model Export")
    parser.add_argument("--quant_config_name", type=str, required=True, help="Name of the quantization configuration.")
    parser.add_argument("--percentile", type=float, default=None, help="Percentile for quantization.")
    parser.add_argument("--qscheme", type=str, choices=["symmetric", "affine"], default="symmetric", help="Quantization scheme.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for inference (auto, cpu, cuda).")
    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Determine device
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    print(f"Device: {device}")

    # Setup data and model
    text_encoder = instantiate(config.text_encoder)

    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    preprocessor = model.asr_model.asr_model.preprocessor
    cfg_dict = OmegaConf.to_container(preprocessor._cfg, resolve=True)
    cfg_dict.pop('_target_', None)
    preprocessor = AudioToMelSpectrogramPreprocessor(**cfg_dict)
    model.asr_model.asr_model.preprocessor = preprocessor

    calibration_samples = get_calibration_samples(config, text_encoder, 1)

    # Collect activations for equalization
    activation_collector = ActivationCollector(model=model)
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

    activation_collector.register_hooks(layer_keys)
    activation_collector.collect_activations(calibration_samples=calibration_samples, device=device)
    activation_stats = activation_collector.get_activation_stats()

    equalizer = OfflineModelEqualizer(model=model.asr_model, activation_stats=activation_stats)
    equalizer.apply_equalization()
    activation_collector.remove_hooks()

    # Get quantization configuration
    quant_config, custom_qconfig = get_quantization_config(args.quant_config_name, args.qscheme, args.percentile)
    dynamic_modules = quant_config.get("dynamic_modules", [])
    static_modules = quant_config.get("static_modules", [])


    # Apply quantization
    model = QuantizedASRModel(model, dynamic_modules=dynamic_modules, static_modules=static_modules, qconfig_dict=custom_qconfig, calibration_samples=calibration_samples, device=device)

    quant_checkpoint = quant_config.get("checkpoint_path", None)
    if quant_checkpoint is not None and os.path.exists(quant_checkpoint):
        state_dict = torch.load(quant_checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded quantized model from {quant_checkpoint}")

    # Export to ONNX
    example_input = (
        torch.randn(1, 80, 201).to('cpu'),
        torch.tensor([201], dtype=torch.int32).to('cpu')
    )
    asr_model = model.asr_model

    wrapped_asr_model = ASR_Nopre_Wrapper(asr_model)
    onnx_path = f"{args.quant_config_name}_nopre_id.onnx"
    torch.onnx.export(
        wrapped_asr_model,
        example_input,
        onnx_path,
        opset_version=16,
        input_names=["input_signal", "input_signal_length"],
        output_names=["tokens_logits"],
        dynamic_axes=None
    )

    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    main()
