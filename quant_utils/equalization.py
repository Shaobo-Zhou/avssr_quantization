import random
import warnings
import os
import pickle
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
from pruning import prune_layer, check_sparsity
from quantization import get_module, set_module
from nemo_new.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor


class ActivationCollector:
    def __init__(self, model):
        self.model = model
        self.asr_model = model.asr_model.asr_model
        self.activation_stats = {}
        self.hook_handles = []  # Store hook handles for later removal

    def register_hooks(self, layer_keys):
        """Register hooks to capture input activations for specified layers."""
        for layer_key in layer_keys:
            layer = dict(self.asr_model.named_modules()).get(layer_key)
            if layer is not None:
                hook_handle = layer.register_forward_pre_hook(self._hook_fn(layer_key))
                #hook_handle = layer.register_forward_hook(self._hook_fn(layer_key))
                self.hook_handles.append(hook_handle)  # Save the handle
                #print(f"Registered hook for layer: {layer_key}")
            else:
                print(f"Layer {layer_key} not found.")

    def _hook_fn(self, layer_key):
        #Hook function to record per-channel min and max values of input activations for each layer.
        def hook(module, input):
            # `input` is a tuple, as PyTorch hooks receive inputs as a tuple
            input_data = input[0]

            # Check if the input is quantized and dequantize if necessary
            if input_data.is_quantized:
                input_data = input_data.dequantize()

            # Move the data to CPU and detach from the computation graph
            input_data = input_data.cpu().detach()

            # Compute per-channel min and max values
            min_vals = input_data.amin(dim=(0, 1)).numpy()  # Per-channel min
            max_vals = input_data.amax(dim=(0, 1)).numpy()  # Per-channel max

            if layer_key not in self.activation_stats:
                self.activation_stats[layer_key] = {'min': min_vals, 'max': max_vals}
            else:
                # Update per-channel min/max values
                self.activation_stats[layer_key]['min'] = np.minimum(self.activation_stats[layer_key]['min'], min_vals)
                self.activation_stats[layer_key]['max'] = np.maximum(self.activation_stats[layer_key]['max'], max_vals)

        return hook
    """ def _hook_fn(self, layer_key):
        
        def hook(module, input, output):
            # 'output' is the tensor produced by this module
            if output.is_quantized:
                output = output.dequantize()

            # Move the data to CPU and detach from the computation graph
            output_data = output.cpu().detach()

            # For example, if the output shape is (B, T, Channels),
            # we reduce over dimensions (0,1). Modify as needed for your shape.
            min_vals = output_data.amin(dim=(0, 1)).numpy()  # Per-channel min
            max_vals = output_data.amax(dim=(0, 1)).numpy()  # Per-channel max

            if layer_key not in self.activation_stats:
                self.activation_stats[layer_key] = {'min': min_vals, 'max': max_vals}
            else:
                # Update per-channel min/max values
                self.activation_stats[layer_key]['min'] = np.minimum(
                    self.activation_stats[layer_key]['min'], min_vals
                )
                self.activation_stats[layer_key]['max'] = np.maximum(
                    self.activation_stats[layer_key]['max'], max_vals
                )
        return hook """

    def collect_activations(self, calibration_samples, device):
        """Iterate over calibration data to collect activation statistics."""
        with torch.no_grad():
            for test_batch in tqdm(calibration_samples, desc="Gathering statistics", unit="batch"):
                self.model(**test_batch)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        print("All hooks have been removed.")

    def get_activation_stats(self):
        """Return the collected activation statistics."""   
        return self.activation_stats

class OfflineModelEqualizer:
    def __init__(self, model, activation_stats, alpha=0.5):
        """
        Offline equalizer for a model's weights using activation statistics.

        Args:
            model (nn.Module): The target model for equalization.
            activation_stats (dict): Per-layer activation statistics with min/max values.
            alpha (float): Balancing factor between activations and weights. Default: 0.5.
        """
        self.model = model
        self.activation_stats = activation_stats
        self.alpha = alpha
        self.activation_stats_before = {}
        self.activation_stats_after = {}

    def compute_scaling_factor(self, layer, layer_key):
        """
        Compute per-channel scaling factor based on activations and weights.

        Args:
            layer (nn.Module): The current layer (e.g., nn.Linear).
            layer_key (str): Key to access the activation stats for the layer.

        Returns:
            torch.Tensor: Per-channel scaling factors.
        """
        if "self.attn" in layer_key:
            layer_qkey = f"{layer_key}_linear_q"
            layer_kkey = f"{layer_key}_linear_k"
            layer_vkey = f"{layer_key}_linear_v"
            try:
                # Per-channel max activations for query, key, and value
                max_qactivation = np.maximum(
                    np.abs(self.activation_stats[layer_qkey]['min']),
                    np.abs(self.activation_stats[layer_qkey]['max'])
                )
                max_kactivation = np.maximum(
                    np.abs(self.activation_stats[layer_kkey]['min']),
                    np.abs(self.activation_stats[layer_kkey]['max'])
                )
                max_vactivation = np.maximum(
                    np.abs(self.activation_stats[layer_vkey]['min']),
                    np.abs(self.activation_stats[layer_vkey]['max'])
                )

                # Take the maximum across query, key, and value channels
                max_activation = np.maximum.reduce([max_qactivation, max_kactivation, max_vactivation])

                # Compute per-channel max weight
                max_weight = layer.weight.abs().max(dim=0).values.cpu().detach().numpy()  # Assuming per-channel scaling

                # Compute scaling factor
                scale_factor = (max_activation ** self.alpha) / (max_weight ** (1 - self.alpha))
                return torch.tensor(scale_factor, device=layer.weight.device)
            except KeyError as e:
                raise ValueError(f"Missing activation stats for layer {layer_key}.") from e

        else:
            try:
                # Per-channel max activation values
                max_activation = np.maximum(
                    np.abs(self.activation_stats[layer_key]['min']),
                    np.abs(self.activation_stats[layer_key]['max'])
                )

                # Compute per-channel max weight
                max_weight = layer.weight.abs().max(dim=0).values.cpu().detach().numpy()  # Assuming per-channel scaling

                # Compute scaling factor
                scale_factor = (max_activation ** self.alpha) / (max_weight ** (1 - self.alpha))
                return torch.tensor(scale_factor, device=layer.weight.device)
            except KeyError as e:
                raise ValueError(f"Missing activation stats for layer {layer_key}.") from e

    def apply_equalization(self):
        """
        Perform offline equalization for all target layers.
        """
        layer_keys = [
            f"encoder.layers.{i}.feed_forward1.linear1" for i in range(16)
        ] + [
            f"encoder.layers.{i}.self_attn" for i in range(16)
        ] + [
            f"encoder.layers.{i}.feed_forward2.linear1" for i in range(16)
        ]
        """ for layer_key in layer_keys:
            layer = self.get_layer_by_key(layer_key)
            if layer is None:
                print(f"Layer {layer_key} not found in the model.")
                continue

            if isinstance(layer, nn.Linear):
                print(f"Equalizing layer: {layer_key}")
                scale_factor = self.compute_scaling_factor(layer, layer_key)
                print(f"{layer_key} scale factor", scale_factor)
                #print(scale_factor.shape)
                self.scale_weights(layer, scale_factor)
                self.adjust_previous_layer(layer_key, scale_factor) """
        for layer_key in layer_keys:
            layer = self.get_layer_by_key(layer_key)
            if layer is None:
                print(f"Layer {layer_key} not found in the model.")
                continue

            if isinstance(layer, nn.Linear):
                # Save activation stats before equalization
                self.activation_stats_before[layer_key] = self.activation_stats.get(layer_key, {})

                # Perform equalization
                #print(f"Equalizing layer: {layer_key}")
                scale_factor = self.compute_scaling_factor(layer, layer_key)
                self.scale_weights(layer, scale_factor)
                self.adjust_previous_layer(layer_key, scale_factor)

                # Collect activation stats after equalization
                self.activation_stats_after[layer_key] = self.activation_stats.get(layer_key, {})


    def scale_weights(self, layer, scale_factor):
        """
        Scale the weights of the current layer.

        Args:
            layer (nn.Module): Current layer to be scaled.
            scale_factor (torch.Tensor): Scaling factors for the layer's weights.
        """
        scale_factor = scale_factor.view(1,-1)  # Shape: (out_features, 1)
        layer.weight.data *= scale_factor
        """ if layer.bias is not None:
            layer.bias.data /= scale_factor.squeeze() """ 

    def adjust_previous_layer(self, layer_key, scale_factor):
        """
        Adjust the weights of the previous layer to apply inverse scaling.

        Args:
            layer_key (str): Key of the current layer.
            scale_factor (torch.Tensor): Scaling factors to adjust the previous layer.
        """
        prev_layer_key = self.get_previous_layer_key(layer_key)
        prev_layer = self.get_layer_by_key(prev_layer_key)

        if prev_layer is None:
            print(f"Previous layer not found for {layer_key}. Skipping adjustment.")
            return

        if isinstance(prev_layer, nn.Linear):
            scale_factor = scale_factor.squeeze()  # Ensure scale_factor matches bias dimensions
            prev_layer.weight.data /= scale_factor.view(1,-1)
            if prev_layer.bias is not None:
                prev_layer.bias.data /= scale_factor
        elif isinstance(prev_layer, nn.LayerNorm):
            # No need to squeeze for LayerNorm
            prev_layer.weight.data /= scale_factor
            if prev_layer.bias is not None:
                prev_layer.bias.data /= scale_factor

    def get_previous_layer_key(self, layer_key):
        """
        Determine the key of the previous layer.

        Args:
            layer_key (str): Key of the current layer.

        Returns:
            str or None: Key of the previous layer, or None if not found.
        """
        # Add rules for identifying previous layers
        if "feed_forward1.linear2" in layer_key:
            return layer_key.replace("feed_forward1.linear2", "norm_conv")
        elif "feed_forward2.linear2" in layer_key:
            return layer_key.replace("feed_forward2.linear2", "norm_out")
        elif "feed_forward1.linear1" in layer_key:
            return layer_key.replace("feed_forward1.linear1", "norm_feed_forward1")
        elif "feed_forward2.linear1" in layer_key:
            return layer_key.replace("feed_forward2.linear1", "norm_feed_forward2")
        return None

    def get_layer_by_key(self, layer_key):
        """
        Retrieve a layer by its key in the model.

        Args:
            layer_key (str): Key of the layer to retrieve.

        Returns:
            nn.Module: The corresponding layer, or None if not found.
        """
        return dict(self.model.asr_model.named_modules()).get(layer_key)
    
    def plot_activation_distribution(self, layer_key):
        """
        Plot the channel-wise activation distribution before and after equalization.

        Args:
            layer_key (str): The specific layer to visualize.
        """
        before = self.activation_stats_before.get(layer_key, {})
        after = self.activation_stats_after.get(layer_key, {})

        min_before = before.get('min', [])
        max_before = before.get('max', [])
        min_after = after.get('min', [])
        max_after = after.get('max', [])

        channels = np.arange(len(min_before))

        plt.figure(figsize=(12, 6))
        # Plot before equalization
        plt.plot(channels, min_before, label='Min (Before)', color='blue', linestyle='--')
        plt.plot(channels, max_before, label='Max (Before)', color='red', linestyle='--')
        # Plot after equalization
        plt.plot(channels, min_after, label='Min (After)', color='blue', linestyle='-')
        plt.plot(channels, max_after, label='Max (After)', color='red', linestyle='-')

        # Add labels, legend, and title
        plt.xlabel('Channels')
        plt.ylabel('Activation Value')
        plt.title(f'Activation Distribution for {layer_key}')
        plt.legend()
        plt.grid()
        output_path = "Activation_new.png"
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()


class ModelEqualizer:
    def __init__(self, model, activation_stats, alpha=0.5):
        self.model = model
        self.activation_stats = activation_stats  # Collected activation stats from pre-calculated file
        self.alpha = alpha

    def get_layer_activation_stats(self, layer_key):

        for key, value in self.activation_stats.items():
            print(key, ":", value) 
        """Retrieve activation stats for the given layer key from the activation stats dictionary."""
        if layer_key not in self.activation_stats:
            raise ValueError(f"Activation stats not found for layer {layer_key}")
        
        min_val = self.activation_stats[layer_key]['min']
        max_val = self.activation_stats[layer_key]['max']

        return min_val, max_val

    def equalize_layer(self, layer, min_activation, max_activation):
        """Applies weight-activation equalization to a given layer."""
        # Get weights and compute the max of weights
        weight = layer.weight.data
        max_weight = torch.max(weight.abs())

        # Compute scaling factor s_j based on min/max of activations
        max_activation = max(abs(min_activation), abs(max_activation))
        scale_factor = (max_activation ** self.alpha) / (max_weight ** (1 - self.alpha))

        # Apply scaling to weights and return the inverse scale for input activation
        layer.weight.data *= scale_factor  # Scale weights
        return 1 / scale_factor  # Return the inverse scaling factor for input scaling

    def apply_equalization(self):
        # List of encoder layers you want to equalize
        layer_keys =[
            f"encoder.layers.{i}.feed_forward1.linear2" for i in range(1)
        ] 
        """+ [
            f"encoder.layers.{i}.feed_forward1.linear1" for i in range(1)
        ] + [
            f"encoder.layers.{i}.feed_forward2.linear1" for i in range(16)
        ] + [
            f"encoder.layers.{i}.feed_forward2.linear2" for i in range(16)
        ] 
         + [
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
        """
        # Apply equalization on each layer
        for layer_key in layer_keys:
            layer = dict(self.model.asr_model.named_modules()).get(layer_key)
            if layer is None:
                print(f"Layer {layer_key} not found.")
                continue

            # Get activation stats from the pre-collected activation stats
            min_activation, max_activation = self.get_layer_activation_stats(layer_key)

            # Perform equalization for the current layer
            input_scale_factor = self.equalize_layer(layer, min_activation, max_activation)
            
            # Scale input activation by input_scale_factor before it enters the layer
            self.scale_input(layer_key, input_scale_factor)
            print(f"Applied equalization on {layer_key} with scale factor: {1/input_scale_factor:.4f}")

    def scale_input(self, layer_key, input_scale_factor):
        """Scales the input to a layer by the inverse scale factor."""
        # Define a small module to scale the input before it enters the target layer
        class InputScaler(nn.Module):
            def __init__(self, scale):
                super(InputScaler, self).__init__()
                self.scale = scale

            def forward(self, x):
                return x * self.scale

        # Insert the input scaling layer before the target layer in the model
        parent_module, layer_name = self._get_parent_module(layer_key)
        input_scaler = InputScaler(input_scale_factor)

        # Replace the original layer with a sequential container that includes input scaling
        setattr(parent_module, layer_name, nn.Sequential(input_scaler, getattr(parent_module, layer_name)))

    def _get_parent_module(self, layer_key):
        """Helper function to get the parent module of a given layer in a nested model."""
        module_hierarchy = layer_key.split('.')
        parent_module = self.model.asr_model
        for module_name in module_hierarchy[:-1]:  # Traverse to the parent module
            parent_module = getattr(parent_module, module_name)
        return parent_module, module_hierarchy[-1]  # Return the parent and final layer name


