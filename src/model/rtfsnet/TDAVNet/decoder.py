import inspect

import torch
import torch.nn as nn
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from librosa import util as librosa_util
from librosa.util import pad_center, tiny
from scipy.signal import get_window
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F



class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, max_frames, filter_length, hop_length, win_length, window):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )
        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window
        window_sum = self.window_sumsquare(
            self.window,
            max_frames,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.filter_length,
            dtype=np.float32,
        )
        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("window_sum", torch.from_numpy(window_sum))
        self.tiny = tiny(window_sum)

    def forward(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )
    
        if self.window is not None:
            win_dim = inverse_transform.size(-1)
            window_sum = self.window_sum[:win_dim]
            # remove modulation effects
            #inverse_transform = inverse_transform.squeeze()
            approx_nonzero_indices = (window_sum > self.tiny).nonzero()

            #inverse_transform[approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            inverse_transform[..., approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            #inverse_transform = inverse_transform.unsqueeze(0).unsqueeze(1)
            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length
        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]
        return inverse_transform

    @staticmethod
    def window_sumsquare(
        window,
        n_frames,
        hop_length=200,
        win_length=800,
        n_fft=800,
        dtype=np.float32,
        norm=None,
    ):
        """
        # from librosa 0.6
        Compute the sum-square envelope of a window function at a given hop length.
        This is used to estimate modulation effects induced by windowing
        observations in short-time fourier transforms.
        Parameters
        ----------
        window : string, tuple, number, callable, or list-like
            Window specification, as in `get_window`
        n_frames : int > 0
            The number of analysis frames
        hop_length : int > 0
            The number of samples to advance between frames
        win_length : [optional]
            The length of the window function.  By default, this matches `n_fft`.
        n_fft : int > 0
            The length of each analysis frame.
        dtype : np.dtype
            The data type of the output
        Returns
        -------
        wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
            The sum-squared envelope of the window function
        """
        if win_length is None:
            win_length = n_fft

        n = n_fft + hop_length * (n_frames - 1)
        x = np.zeros(n, dtype=dtype)

        # Compute the squared window at the desired length
        win_sq = get_window(window, win_length, fftbins=True)
        win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
        win_sq = librosa_util.pad_center(win_sq, size=n_fft)

        # Fill the envelope
        for i in range(n_frames):
            sample = i * hop_length
            x[sample : min(n, sample + n_fft)] += win_sq[
                : max(0, min(n_fft, n - sample))
            ]
        return x


class ExportableISTFTModule(nn.Module):
    def __init__(
        self, max_frames, filter_length, hop_length, win_length, window="hann"
    ):
        super().__init__()
        self.stft = STFT(
            max_frames,
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

    def forward(self, mag, x, y):
        phase = torch.atan2(y, x)
        return self.stft(mag, phase)




class BaseDecoder(nn.Module):
    def pad_to_input_length(self, separated_audio, input_frames):
        output_frames = separated_audio.shape[-1]
        return nn.functional.pad(separated_audio, [0, input_frames - output_frames])

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class ConvolutionalDecoder(BaseDecoder):
    def __init__(
        self,
        in_chan: int,
        n_src: int,
        kernel_size: int,
        stride: int,
        bias=False,
        *args,
        **kwargs,
    ):
        super(ConvolutionalDecoder, self).__init__()

        self.in_chan = in_chan
        self.n_src = n_src
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

        self.padding = (self.kernel_size - 1) // 2
        self.output_padding = ((self.kernel_size - 1) // 2) - 1

        self.decoder = nn.ConvTranspose1d(
            in_channels=self.in_chan,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            bias=self.bias,
        )

        torch.nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, separated_audio_embedding: torch.Tensor, input_shape: torch.Size):
        # B, n_src, N, T
        batch_size, length = input_shape[0], input_shape[-1]

        separated_audio_embedding = separated_audio_embedding.view(
            batch_size * self.n_src, self.in_chan, -1
        )

        separated_audio = self.decoder(
            separated_audio_embedding
        )  # B * n_src, N, T -> B*n_src, 1, L
        separated_audio = self.pad_to_input_length(separated_audio, length)
        separated_audio = separated_audio.view(batch_size, self.n_src, -1)

        return separated_audio


class STFTDecoder(BaseDecoder):
    def __init__(
        self,
        win: int,
        hop_length: int,
        in_chan: int,
        n_src: int,
        kernel_size: int = -1,
        stride: int = 1,
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super(STFTDecoder, self).__init__()
        self.win = win
        self.hop_length = hop_length
        self.in_chan = in_chan
        self.n_src = n_src
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.stride = stride
        self.bias = bias


        if self.kernel_size > 0:
            self.decoder = nn.ConvTranspose2d(
                in_channels=self.in_chan,
                out_channels=2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            )
            torch.nn.init.xavier_uniform_(self.decoder.weight)
        else:
            self.decoder = nn.Identity()

        self.register_buffer("window", torch.hann_window(self.win), False)

        self.istft = ExportableISTFTModule(
            max_frames=5200,  # Set as needed
            filter_length=self.win,
            hop_length=hop_length,
            win_length=self.win,
            window="hann",
        )

    def forward(self, x: torch.Tensor, input_shape: torch.Size):

        # B, n_src, N, T, F

        batch_size, length = input_shape[0], input_shape[-1]

        x = x.view(
            batch_size * self.n_src, self.in_chan, *x.shape[-2:]
        )  # B, n_src, N, T, F -> # B * n_src, N, T, F
        decoded_separated_audio = self.decoder(
            x
        )  # B * n_src, N, T, F - > B * n_src, 2, T, F
        mag = decoded_separated_audio.norm(dim=1)  # Magnitude: B * n_src, T, F
        real = decoded_separated_audio[:, 0]  # Real part
        imag = decoded_separated_audio[:, 1]  # Imaginary part
        real = real.transpose(1, 2).contiguous()  # B*n_src, F, T
        imag = imag.transpose(1, 2).contiguous()  # B*n_src, F, T
        mag = mag.transpose(1, 2).contiguous()  # B*n_src, F, T

        """ spec = torch.complex(
            decoded_separated_audio[:, 0], decoded_separated_audio[:, 1]
        )  # B*n_src, T, F
        # spec = torch.stack([spec.real, spec.imag], dim=-1)  # B*n_src, T, F
        spec = spec.transpose(1, 2).contiguous()  # B*n_src, F, T
        output = torch.istft(
            spec,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            length=length,
        )  # B*n_src, L  """
        
        output = self.istft(mag, real, imag)  # B * n_src, L
        output = output.view(batch_size, self.n_src, length)  # B, n_src, L

        return output


def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError(
                "Could not interpret normalization identifier: " + str(identifier)
            )
        return cls
    else:
        raise ValueError(
            "Could not interpret normalization identifier: " + str(identifier)
        )
