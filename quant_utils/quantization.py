import hydra
import torch
import warnings
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.quantization.observer import HistogramObserver
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerFeedForward, ConformerConvolution
from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttention


def get_module(model, module_string):
    """Retrieve submodule using its string reference."""
    curr = model  # Start from the model itself
    for attr in module_string.split("."):
        if attr.isnumeric():
            curr = curr[int(attr)]
        else:
            curr = getattr(curr, attr)
    return curr

def set_module(model, module_string, new_module):
    """Set a submodule using its string reference."""
    curr = model  # Start from the model itself
    attrs = module_string.split(".")
    for attr in attrs[:-1]:
        if attr.isnumeric():
            curr = curr[int(attr)]
        else:
            curr = getattr(curr, attr)
    if attrs[-1].isnumeric():
        curr[int(attrs[-1])] = new_module
    else:
        setattr(curr, attrs[-1], new_module)

# Static Quantization Wrapper (for models that can be statically quantized directly)
class StaticQuant(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.__dict__['_modules']:
            return self.__dict__['_modules'][name]
        else:
            return getattr(self.__dict__['_modules']['model'], name)

    def forward(self, x=None, *args, **kwargs):

        if x is None and 'encoder_output' in kwargs:      ### The decoder input is passed by keyword argument
            x = kwargs['encoder_output']
            x = self.quant(x)
            kwargs['encoder_output'] = x
            x = self.model(**kwargs)
            
        else:
            x = self.quant(x)
            x = self.model(x, *args, **kwargs)
        if isinstance(x, tuple):
            return tuple(self.dequant(output) for output in x)
        else:
            return self.dequant(x)

        
### Define custom quantized modules
class CustomFeedForward(nn.Module):
    """Custom implementation of a quantized ConformerFeedForward module."""
    def __init__(self, original_ffn):
        super(CustomFeedForward, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Copy original modules
        self.linear1 = original_ffn.linear1
        self.activation = original_ffn.activation  # Non-ReLU activation(Swish) not supported for quantization
        #self.activation = nn.Hardswish() # Can be approximated by a HardSwish()
        self.dropout = original_ffn.dropout
        self.linear2 = original_ffn.linear2

    def forward(self, x):
        x = self.quant(x)
        x = self.linear1(x)
        
        if isinstance(self.activation, nn.SiLU):  # Check if activation is Swish (SiLU in PyTorch)
            x = self.dequant(x)
            x = self.activation(x)  # Handle non-supported activation (Swish)
            x = self.quant(x)
        else:
            x = self.activation(x)  # Directly apply activation if supported

        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dequant(x)  
        
        return x

class CustomAttention(nn.Module):
    """Custom implementation of a quantized RelPositionMultiHeadAttention module."""
    def __init__(self, original_attn):
        super().__init__()
        self.quant = QuantStub()
        self.attn = original_attn
        self.dequant = DeQuantStub()
    def forward(self, query, key, value, mask, pos_emb, cache=None):
        query = self.quant(query)
        key = self.quant(key)
        value = self.quant(value)
        if pos_emb is not None:
            pos_emb = self.quant(pos_emb)
        if mask is not None:
            if mask.dtype == torch.float32:
                mask = self.quant(mask)
            
        q, k, v = self.attn.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.attn.linear_pos(pos_emb).view(n_batch_pos, -1, self.attn.h, self.attn.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        q = self.dequant(q)
        k = self.dequant(k)
        v = self.dequant(v)
        p = self.dequant(p)

        q_with_bias_u = (q + self.attn.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.attn.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.attn.rel_shift(matrix_bd)
        # drops extra elements in the matrix_bd to match the matrix_ac's size
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

        scores = (matrix_ac + matrix_bd) / self.attn.s_d_k  # (batch, head, time1, time2)

        n_batch = v.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.attn.dropout(attn)
        x = torch.matmul(p_attn, v)  # Keep matmul in full precision

        # Keep in full precision until reshaping is complete
        x = x.transpose(1, 2).reshape(n_batch, -1, self.attn.h * self.attn.d_k)  # (batch, time1, d_model)

        x = self.quant(x)  
        x = self.attn.linear_out(x)  
        x = self.dequant(x)  

        return x
    
class QuantizedCausalConv1D(nn.Module):
    """Custom implementation of a quantized causal Conv1D module."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True,):
        super().__init__()
        # Define padding as per causal needs
        self._left_padding = kernel_size - 1 if padding is None else padding[0]
        self._right_padding = 0 if padding is None else padding[1]
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Quantized Convolution layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  
            dilation=dilation,
            groups=groups,
            bias=bias,
        )


    def forward(self, x):
        x = torch.nn.functional.pad(x, (self._left_padding, self._right_padding))
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x
    

class CustomConv(nn.Module):
    def __init__(self, original_conv):
        super(CustomConv, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.pointwise_conv1 = original_conv.pointwise_conv1
        self.depthwise_conv = QuantizedCausalConv1D(in_channels=original_conv.depthwise_conv.in_channels,
            out_channels=original_conv.depthwise_conv.out_channels,
            kernel_size=original_conv.depthwise_conv.kernel_size[0],
            stride=original_conv.depthwise_conv.stride[0],
            padding=[original_conv.depthwise_conv._left_padding, original_conv.depthwise_conv._right_padding],
            dilation=original_conv.depthwise_conv.dilation[0],
            groups=original_conv.depthwise_conv.groups,
            bias=original_conv.depthwise_conv.bias is not None)
        self.depthwise_conv.conv.weight.data = original_conv.depthwise_conv.weight.data.clone()
        if original_conv.depthwise_conv.bias is not None:
            self.depthwise_conv.conv.bias.data = original_conv.depthwise_conv.bias.data.clone()
        self.batch_norm = original_conv.batch_norm
        self.activation = original_conv.activation
        self.pointwise_conv2 = original_conv.pointwise_conv2

    def forward(self, x, pad_mask=None, cache=None):
        x = self.quant(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.dequant(x)

        x = nn.functional.glu(x, dim=1) # Not supported activation
        x = self.depthwise_conv(x)  #Self defined module, not supported for quantization
        x = self.batch_norm(x)
        x = self.activation(x)
        
        x = self.quant(x)  
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        x = self.dequant(x)  
        
        return x

class PercentileHistogramObserver(torch.quantization.HistogramObserver):
    def __init__(self, percentile=0.999, bins=2048, **kwargs):
        super().__init__(bins=bins, **kwargs)
        self.percentile = percentile

    def _calculate_percentile_min_max(self):
        """
        Calculates min and max values based on the specified percentile.
        """
        hist = self.histogram
        bin_width = (self.max_val - self.min_val) / self.bins
        cumulative_hist = torch.cumsum(hist, dim=0)

        total = cumulative_hist[-1]

        # Compute the lower and upper bounds
        lower_bound = total * (1 - self.percentile)
        upper_bound = total * self.percentile

        # Find the bin indices corresponding to the bounds
        min_bin = torch.searchsorted(cumulative_hist, lower_bound).item()
        max_bin = torch.searchsorted(cumulative_hist, upper_bound).item()

        # Compute min_val and max_val based on bin indices
        min_val = self.min_val + min_bin * bin_width
        max_val = self.min_val + max_bin * bin_width

        return min_val, max_val

    def calculate_qparams(self):
        is_uninitialized = self.min_val == float("inf") and self.max_val == float("-inf")
        if is_uninitialized:
            warnings.warn("Must run observer before calling calculate_qparams.")
            return torch.tensor([1.0]), torch.tensor([0])

        # Use percentile-based min and max calculation
        min_val, max_val = self._calculate_percentile_min_max()

        return self._calculate_qparams(min_val, max_val)

# Quantized ASR model class
class QuantizedASRModel(nn.Module):
    # Wrapper for quantizing an ASR model with static and dynamic quantization
    def __init__(self, model, dynamic_modules, static_modules, qconfig_dict, load_SS=False, calibration_samples=None, device='auto', qat=False):
        super(QuantizedASRModel, self).__init__()
        self.model = model
        self.asr_model = model.asr_model.asr_model
        
        self.device = device
        self.load_SS = load_SS # If true, can load and use the pre-saved SS outputs directly, as SS model wasn't modified
        self.qat = qat 

        # qconfig_dict should be a dictionary containing qconfig for each module type
        self.qconfig_dict = qconfig_dict
        self.dynamic_modules = dynamic_modules
        self.static_modules = static_modules
        self.calibration_samples = calibration_samples

        # Quantize the original model using the helper functions
        self.custom_quantize()

    def fuse_conv_relu(self, model):
        # Fuse Conv2d and ReLU layers in the ConvSubsampling part
        torch.quantization.fuse_modules(
            model.encoder.pre_encode.conv, 
            [['0', '1'], ['2', '3']], 
            inplace=True
        )

    def custom_quantize(self, dynamic_targets=None, dynamic_dtype=torch.qint8):
        """Performs in-place quantization of an ASR model using specific qconfigs for each module type."""
        self.fuse_conv_relu(self.asr_model)
        
        ##################################################
        # Dynamic Quantization                           #
        ##################################################
        if self.dynamic_modules:
            if dynamic_targets is None:
                dynamic_targets = {
                    torch.nn.LSTM,
                    torch.nn.GRU,
                    torch.nn.RNNCell,
                    torch.nn.GRUCell,
                    torch.nn.LSTMCell,
                    torch.nn.Linear
                }

            for module in self.dynamic_modules:
                torch.quantization.quantize_dynamic(
                    get_module(self.asr_model, module),
                    dynamic_targets,
                    dtype=dynamic_dtype,
                    inplace=True,
                )
        
        ##################################################
        # Static Quantization with Specific qconfig      #
        ##################################################
        if self.static_modules:
            for module in self.static_modules:
                target_module = get_module(self.asr_model, module)
                
                
                # Apply module-specific qconfig
                if isinstance(target_module, ConformerFeedForward):
                    set_module(self.asr_model, module, CustomFeedForward(target_module))
                    get_module(self.asr_model, module).qconfig = self.qconfig_dict.get('feedforward', torch.ao.quantization.default_qconfig)


                elif isinstance(target_module, ConformerConvolution):
                    set_module(self.asr_model, module, CustomConv(target_module))
                    get_module(self.asr_model, module).qconfig = self.qconfig_dict.get('conv', torch.ao.quantization.default_qconfig)

                elif isinstance(target_module, RelPositionMultiHeadAttention):
                    set_module(self.asr_model, module, CustomAttention(target_module))
                    get_module(self.asr_model, module).qconfig = self.qconfig_dict.get('attn', torch.ao.quantization.default_qconfig)

                elif module == "encoder.pre_encode" or module == "decoder":
                    set_module(self.asr_model, module, StaticQuant(target_module))
                    get_module(self.asr_model, module).qconfig = self.qconfig_dict.get('pre_encode', torch.ao.quantization.default_qconfig)
                
                elif isinstance(target_module,nn.LayerNorm) :
                    set_module(self.asr_model, module, StaticQuant(target_module))
                    get_module(self.asr_model, module).qconfig = self.qconfig_dict.get('layer_norm', torch.ao.quantization.default_qconfig)
                else: 
                    print(f"Module {module} does not exist!")

        if self.qat: #Quantization-Awared-Training (not implemented yet)
            self.asr_model.to(self.device)
            self.asr_model.train()
            torch.ao.quantization.prepare_qat(self.asr_model, inplace=True)

        else: # Calibration
            self.asr_model.eval()
            torch.ao.quantization.prepare(self.asr_model, inplace=True)

            # Calibration step: pass a few batches through the model
            if self.load_SS and self.calibration_samples:
                for test_batch in tqdm(self.calibration_samples, desc="Calibrating", unit="batch"):
                    fused_feats = test_batch['fused_feats'].to(self.device)
                    s_audio_length = test_batch["s_audio_length"].squeeze().to(self.device)
                    self.asr_model(input_signal=fused_feats, input_signal_length=s_audio_length)
            elif self.calibration_samples:
                for test_batch in tqdm(self.calibration_samples, desc="Calibrating", unit="batch"):
                    self.model(**test_batch)
            observer_dict = torch.quantization.get_observer_state_dict(self.asr_model)
            torch.ao.quantization.convert(self.asr_model, remove_qconfig=False, inplace=True)

        
    def plot_quantization_histogram(self, observer_dict, layer_key, layer_index, plot_percentiles=False, output_dir="."):
        """
        Plots the histogram of the specified layer's weights or activations with quantization bounds.

        Parameters:
        - observer_dict: Dictionary containing observer data.
        - layer_key: The name of the layer to plot.
        - layer_index: Index of the layer to include in the plot filename.
        - plot_percentiles: Whether to include percentile bounds in the plot.
        - output_dir: Directory where the plot should be saved.
        """
        # Get the specified layer's histogram and quantization parameters
        layer = dict(self.asr_model.named_modules()).get(layer_key)
        scale = layer.scale.item() if isinstance(layer.scale, torch.Tensor) else layer.scale
        zero_point = layer.zero_point.item() if isinstance(layer.zero_point, torch.Tensor) else layer.zero_point
        hist_key = f"{layer_key}.activation_post_process.histogram"
        min_key = f"{layer_key}.activation_post_process.min_val"
        max_key = f"{layer_key}.activation_post_process.max_val"

        # Retrieve histogram, min, and max values
        histogram = observer_dict.get(hist_key)
        min_val = observer_dict.get(min_key)
        max_val = observer_dict.get(max_key)

        if histogram is None or min_val is None or max_val is None:
            print(f"No histogram data found for layer: {layer_key}")
            return

        bin_count = len(histogram)
        bin_edges = np.linspace(min_val, max_val, bin_count + 1)

        # Prepare for percentile plotting if enabled
        percentile_values = []
        if plot_percentiles:
            cumulative_histogram = np.cumsum(histogram)
            total = cumulative_histogram[-1]
            percentile_pairs = [(0.001, 0.999), (0.0001, 0.9999), (0.00001, 0.99999)]

            for lower_percentile, upper_percentile in percentile_pairs:
                lower_idx = np.searchsorted(cumulative_histogram, total * lower_percentile)
                upper_idx = np.searchsorted(cumulative_histogram, total * upper_percentile)

                lower_value = min_val + (lower_idx / bin_count) * (max_val - min_val)
                upper_value = min_val + (upper_idx / bin_count) * (max_val - min_val)
                percentile_values.append((lower_value, upper_value))

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.bar(bin_edges[:-1], histogram, width=(max_val - min_val) / bin_count, align='edge', alpha=0.5, label="Histogram")
        plt.axvline(min_val, color='g', linestyle='--', label="Min Value")
        plt.axvline(max_val, color='r', linestyle='--', label="Max Value")

        if plot_percentiles:
            colors = ['brown', 'olive', 'gray']
            labels = [f"Central {(upper - lower) * 100}%" for lower, upper in percentile_pairs]
            for i, (lower_value, upper_value) in enumerate(percentile_values):
                plt.axvline(lower_value, color=colors[i], linestyle='-', label=f"{labels[i]} Lower Bound")
                plt.axvline(upper_value, color=colors[i], linestyle='-', label=f"{labels[i]} Upper Bound")

        # Calculate and plot quantized levels as vertical markers
        quant_levels = (np.arange(0, 256) - zero_point) * scale
        quant_zero_point = quant_levels[0] + zero_point * scale
        plt.axvline(quant_levels[0], color='blue', linestyle='-', linewidth=0.5, label="Quantization Lower Bound")
        plt.axvline(quant_levels[255], color='yellow', linestyle='-', linewidth=0.5, label="Quantization Upper Bound")
        plt.axvline(quant_zero_point, color='black', linestyle='-', label="Quantization Zero Point")

        # Calculate mean squared quantization error (MSQE)
        quant_error_squared = np.zeros_like(histogram)
        bin_width = (max_val - min_val) / bin_count
        for i in range(bin_count):
            bin_center = min_val + (i + 0.5) * bin_width
            bin_center = bin_center.item()
            closest_quant_level = quant_levels[np.argmin(np.abs(quant_levels - bin_center))]
            quant_error_squared[i] = ((bin_center - closest_quant_level) ** 2) * histogram[i]

        msqe = quant_error_squared.sum() / histogram.sum() if histogram.sum() > 0 else 0
        plt.text(0.7, 0.9, f"mse={msqe:.2f}", transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

        # Set titles and labels
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.yscale('log')
        plt.title("Observer Histogram with Quantization Steps and MSQE")
        plt.legend(loc='upper left')

        # Save the plot
        output_path = os.path.join(output_dir, f"{layer_key}.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()     

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

