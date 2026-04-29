"""PerceptualAdaptiveSpectrogram — torch port.

Two public modules:
- ``PerceptualAdaptiveSpectrogramStreaming`` — stateful, per-chunk; matches the C++.
- ``PerceptualAdaptiveSpectrogram`` — stateless wrapper for full-clip / training use.

Spec: docs/superpowers/specs/2026-04-27-torch-modules-perceptual-adaptive-spectrogram-design.md
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from torch_modules.perceptual_spectral_analysis.log_scale import LogScale
from torch_modules.perceptual_spectral_analysis.moving_max_min import MovingMaxMinVertical
from torch_modules.perceptual_spectral_analysis.spectrogram_adaptive_moving import (
    SpectrogramAdaptiveMoving,
)


class PerceptualAdaptiveSpectrogramStreaming(nn.Module):
    """Streaming (per-buffer-size-chunk) port of ``PerceptualAdaptiveSpectrogram``.

    Forward expects ``(..., bufferSize)`` and returns ``(..., n_bands, 2^(n_spectrograms - 1))`` in dB.

    State lifecycle (lazy allocation):
    - First forward allocates state buffers sized to the input's flattened batch shape ``B*``,
      device, and dtype.
    - Subsequent calls must match the bound ``B*``; mismatches raise ``ValueError``.
    - ``reset()`` drops state and re-allocates on the next forward.
    - ``.to(device)`` works after first forward; before first forward, do ``module.to(device)``
      then run forward on input on the same device.
    - ``reset()`` always disconnects autograd (overwrites state with sentinels, no requires_grad).
    """

    def __init__(
        self,
        *,
        buffer_size: int,
        n_bands: int,
        n_spectrograms: int = 3,
        n_folds: int = 1,
        nonlinearity: int = 0,
        sample_rate: float = 48000.0,
        frequency_min: float = 20.0,
        frequency_max: float = 20000.0,
        spectral_tilt: bool = True,
        method: str = "ADAPTIVE",
    ):
        super().__init__()
        if method.upper() != "ADAPTIVE":
            raise ValueError(f"only method='ADAPTIVE' is supported in v1, got {method!r}")
        if n_folds != 1:
            raise ValueError(f"only nFolds=1 is supported in v1, got {n_folds}")

        self.buffer_size = buffer_size
        self.n_bands = n_bands

        n_bands_internal = 2 * buffer_size + 1
        self.spectrogram = SpectrogramAdaptiveMoving(
            buffer_size=buffer_size, n_bands=n_bands_internal,
            n_spectrograms=n_spectrograms, n_folds=n_folds, nonlinearity=nonlinearity,
        )
        self.log_scale = LogScale(
            n_inputs=n_bands_internal, n_outputs=n_bands,
            output_start=frequency_min, output_end=frequency_max,
            input_end=sample_rate / 2,
        )
        self.moving_max_min = MovingMaxMinVertical(filter_length=max(1, n_bands // 500))

        # Spectral tilt: always allocated as register_buffer so forward stays branch-free
        # (TorchScript / torch.compile can trace through). Zero when disabled.
        if spectral_tilt:
            # 3 dB boost per octave. Compute as 10*log10(freq/1000); replace leading -inf with 0
            # to avoid NaN/inf at DC.
            # Spec choice: the C++ produces -inf at DC (log10(0)); the torch port substitutes 0.0
            # so that DC bins receive a 0 dB tilt (no shift) instead of -inf.
            freq = np.linspace(0.0, sample_rate / 2, n_bands_internal, dtype=np.float32)
            with np.errstate(divide="ignore"):
                tilt = 10.0 * np.log10(freq / 1000.0)
            tilt = np.nan_to_num(tilt, neginf=0.0)
        else:
            tilt = np.zeros(n_bands_internal, dtype=np.float32)
        self.register_buffer(
            "spectral_tilt_vector",
            torch.from_numpy(tilt).unsqueeze(-1),
            persistent=False,
        )

    def reset(self) -> None:
        """Reset all stateful sub-modules (FFT cascade buffers, per-level history, moving filters)."""
        self.spectrogram.reset()
        # log_scale and moving_max_min are stateless.

    def forward(self, x: torch.Tensor, *, detach_state: bool = True) -> torch.Tensor:
        if not x.is_floating_point():
            raise ValueError(f"input must be a floating dtype, got {x.dtype}")
        if x.shape[-1] != self.buffer_size:
            raise ValueError(
                f"expected last dim {self.buffer_size}, got {x.shape[-1]} (shape {tuple(x.shape)})"
            )

        spectrogram_db = self.spectrogram(x, detach_state=detach_state)  # (..., n_bands_internal, frames)
        spectrogram_db = spectrogram_db + self.spectral_tilt_vector
        # log_scale wants (..., n_inputs); transpose (n_inputs, frames) -> (frames, n_inputs), then back.
        log_scaled = self.log_scale(spectrogram_db.transpose(-1, -2)).transpose(-1, -2)
        return self.moving_max_min(log_scaled)

    def forward_fullclip(self, x: torch.Tensor) -> torch.Tensor:
        """Process an entire clip ``(..., T)`` in one shot. Stateless.

        ``T`` must be a positive multiple of ``buffer_size``. Returns
        ``(..., n_bands, T // buffer_size * 2^(n_spectrograms-1))`` in dB.
        """
        if not x.is_floating_point():
            raise ValueError(f"input must be a floating dtype, got {x.dtype}")
        T = x.shape[-1]
        if T <= 0 or T % self.buffer_size != 0:
            raise ValueError(
                f"input last dim must be a positive multiple of bufferSize={self.buffer_size}, got {T}"
            )

        spectrogram_db = self.spectrogram.forward_fullclip(x)
        spectrogram_db = spectrogram_db + self.spectral_tilt_vector
        log_scaled = self.log_scale(spectrogram_db.transpose(-1, -2)).transpose(-1, -2)
        return self.moving_max_min(log_scaled)


class PerceptualAdaptiveSpectrogram(nn.Module):
    """Stateless wrapper around ``PerceptualAdaptiveSpectrogramStreaming``.

    Forward expects ``(..., T)`` with ``T`` a positive multiple of ``bufferSize``.
    Splits the time axis into ``T // bufferSize`` chunks, ``reset()``s the streaming
    module once before the loop, then runs it per chunk with ``detach_state=False``
    (so gradients flow across chunks at training time), concatenating outputs
    along the frame axis.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.streaming = PerceptualAdaptiveSpectrogramStreaming(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.streaming.forward_fullclip(x)
