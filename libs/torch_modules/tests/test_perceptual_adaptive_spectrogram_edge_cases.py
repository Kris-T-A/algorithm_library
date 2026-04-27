import numpy as np
import pytest
import torch

from torch_modules.perceptual_spectral_analysis.perceptual_adaptive_spectrogram import (
    PerceptualAdaptiveSpectrogram,
    PerceptualAdaptiveSpectrogramStreaming,
)


@pytest.fixture
def cfg_small():
    return dict(
        buffer_size=64, n_bands=8, n_spectrograms=3, n_folds=1, nonlinearity=0,
        sample_rate=48000.0, frequency_min=20.0, frequency_max=20000.0, spectral_tilt=False,
    )


def test_silence_produces_finite_output(cfg_small):
    module = PerceptualAdaptiveSpectrogramStreaming(**cfg_small)
    out = module(torch.zeros(1, cfg_small["buffer_size"]))
    assert torch.isfinite(out).all()


def test_constant_dc_produces_finite_output(cfg_small):
    module = PerceptualAdaptiveSpectrogramStreaming(**cfg_small)
    out = module(torch.full((1, cfg_small["buffer_size"]), 0.5))
    assert torch.isfinite(out).all()


def test_multichannel_flatten_roundtrip(cfg_small):
    """module(x_batch)[i, c] == module(x_batch[i, c]) for every (i, c)."""
    module = PerceptualAdaptiveSpectrogramStreaming(**cfg_small)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((2, 3, cfg_small["buffer_size"])).astype(np.float32))

    out_batched = module(x)
    assert out_batched.shape[:2] == (2, 3)

    for i in range(2):
        for c in range(3):
            module_per = PerceptualAdaptiveSpectrogramStreaming(**cfg_small)
            out_per = module_per(x[i, c])
            torch.testing.assert_close(out_batched[i, c], out_per, atol=1e-6, rtol=1e-6)


def test_reset_clears_state(cfg_small):
    """Same input twice with reset() between == identical output."""
    module = PerceptualAdaptiveSpectrogramStreaming(**cfg_small)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((1, cfg_small["buffer_size"])).astype(np.float32))

    out1 = module(x)
    module.reset()
    out2 = module(x)
    torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-6)


def test_batch_shape_mismatch_raises(cfg_small):
    module = PerceptualAdaptiveSpectrogramStreaming(**cfg_small)
    module(torch.zeros(2, cfg_small["buffer_size"]))
    with pytest.raises(ValueError, match="batch shape"):
        module(torch.zeros(3, cfg_small["buffer_size"]))


def test_gradcheck_on_stateless_float64():
    """End-to-end autograd check (run on the stateless wrapper for hermetic perturbations)."""
    cfg = dict(
        buffer_size=64, n_bands=8, n_spectrograms=3, n_folds=1, nonlinearity=0,
        sample_rate=48000.0, frequency_min=20.0, frequency_max=20000.0, spectral_tilt=False,
    )
    module = PerceptualAdaptiveSpectrogram(**cfg).to(torch.float64)

    x = torch.randn(1, cfg["buffer_size"], dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4, nondet_tol=1e-5)
