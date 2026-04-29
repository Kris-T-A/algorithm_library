"""Benchmark torch_modules.PerceptualAdaptiveSpectrogram on CPU and CUDA.

Run via pytest::

    pytest libs/torch_modules/tests/benchmark_perceptual_adaptive_spectrogram.py --benchmark-only

Or as a standalone script (prints a small table to stdout)::

    python libs/torch_modules/tests/benchmark_perceptual_adaptive_spectrogram.py
"""
from __future__ import annotations

import time

import pytest
import torch

from torch_modules.perceptual_spectral_analysis.perceptual_adaptive_spectrogram import (
    PerceptualAdaptiveSpectrogram,
)


CONFIGS = [
    dict(buffer_size=4096, n_bands=100, n_spectrograms=3, n_folds=1, nonlinearity=1,
         sample_rate=48000.0, frequency_min=20.0, frequency_max=20000.0, spectral_tilt=False),
    dict(buffer_size=4096, n_bands=500, n_spectrograms=3, n_folds=1, nonlinearity=1,
         sample_rate=48000.0, frequency_min=20.0, frequency_max=20000.0, spectral_tilt=False),
]
BATCHES = [1, 32, 256]
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"nBands={c['n_bands']}")
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("device", DEVICES)
def test_benchmark_stateless(benchmark, cfg, batch, device):
    module = PerceptualAdaptiveSpectrogram(**cfg).to(device)
    x = torch.randn(batch, 4 * cfg["buffer_size"], device=device)
    # Warmup once.
    module(x)
    if device == "cuda":
        torch.cuda.synchronize()

    def run():
        out = module(x)
        if device == "cuda":
            torch.cuda.synchronize()
        return out

    benchmark(run)


LONG_CLIP_MULTIPLIERS = [4, 64]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"nBands={c['n_bands']}")
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mult", LONG_CLIP_MULTIPLIERS, ids=lambda m: f"T={m}xBuf")
def test_benchmark_stateless_clip_length(benchmark, cfg, batch, device, mult):
    module = PerceptualAdaptiveSpectrogram(**cfg).to(device)
    x = torch.randn(batch, mult * cfg["buffer_size"], device=device)
    module(x)  # warmup
    if device == "cuda":
        torch.cuda.synchronize()

    def run():
        out = module(x)
        if device == "cuda":
            torch.cuda.synchronize()
        return out

    benchmark(run)


if __name__ == "__main__":
    print(f"{'config':<40} {'batch':>6} {'device':>6} {'ms/call':>10}")
    for cfg in CONFIGS:
        for batch in BATCHES:
            for device in DEVICES:
                module = PerceptualAdaptiveSpectrogram(**cfg).to(device)
                x = torch.randn(batch, 4 * cfg["buffer_size"], device=device)
                module(x)  # warmup
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(5):
                    module(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / 5 * 1000
                cfg_str = f"nBands={cfg['n_bands']}"
                print(f"{cfg_str:<40} {batch:>6} {device:>6} {ms:>10.2f}")
