import numpy as np
import pytest
import torch

from torch_modules.perceptual_spectral_analysis.perceptual_adaptive_spectrogram import (
    PerceptualAdaptiveSpectrogramStreaming,
)


@pytest.mark.parametrize("nonlinearity", [0, 1])
@pytest.mark.parametrize("n_bands", [100, 800])
def test_streaming_against_cpp(pal, nonlinearity, n_bands):
    cfg = dict(
        bufferSize=512,
        nBands=n_bands,
        sampleRate=48000.0,
        frequencyMin=20.0,
        frequencyMax=20000.0,
        spectralTilt=False,  # tested separately to keep the alignment matrix small
        nSpectrograms=3,
        nFolds=1,
        nonlinearity=nonlinearity,
        method="Adaptive",
    )

    rng = np.random.default_rng(0)
    n_chunks = 12
    signal = rng.standard_normal(n_chunks * cfg["bufferSize"]).astype(np.float32)

    oracle = pal.PerceptualSpectralAnalysis()
    oracle.setCoefficients(cfg)

    module = PerceptualAdaptiveSpectrogramStreaming(
        buffer_size=cfg["bufferSize"], n_bands=cfg["nBands"],
        n_spectrograms=cfg["nSpectrograms"], n_folds=cfg["nFolds"],
        nonlinearity=cfg["nonlinearity"], sample_rate=cfg["sampleRate"],
        frequency_min=cfg["frequencyMin"], frequency_max=cfg["frequencyMax"],
        spectral_tilt=cfg["spectralTilt"],
    )

    delay_ref = (2 * cfg["bufferSize"] + 1) - 1  # n_bands_internal - 1
    n_warmup = -(-delay_ref // cfg["bufferSize"]) + 1

    for i in range(n_chunks):
        chunk = signal[i * cfg["bufferSize"]:(i + 1) * cfg["bufferSize"]]
        cpp_out = oracle.process(chunk)
        torch_out = module(torch.from_numpy(chunk)).numpy()
        if i >= n_warmup:
            np.testing.assert_allclose(torch_out, cpp_out, atol=1e-2)


def test_streaming_with_spectral_tilt(pal):
    """Verify spectralTilt=True is correctly applied (alignment vs C++).

    The C++ tilt vector includes -inf at DC (log10(0)). The torch port substitutes 0.0 there
    (documented spec improvement). We mask out non-finite rows of the C++ output before comparing.
    """
    cfg = dict(
        bufferSize=512, nBands=128, sampleRate=48000.0,
        frequencyMin=20.0, frequencyMax=20000.0, spectralTilt=True,
        nSpectrograms=3, nFolds=1, nonlinearity=0, method="Adaptive",
    )
    rng = np.random.default_rng(0)
    n_chunks = 8
    signal = rng.standard_normal(n_chunks * cfg["bufferSize"]).astype(np.float32)

    oracle = pal.PerceptualSpectralAnalysis()
    oracle.setCoefficients(cfg)

    module = PerceptualAdaptiveSpectrogramStreaming(
        buffer_size=cfg["bufferSize"], n_bands=cfg["nBands"],
        n_spectrograms=cfg["nSpectrograms"], n_folds=cfg["nFolds"],
        nonlinearity=cfg["nonlinearity"], sample_rate=cfg["sampleRate"],
        frequency_min=cfg["frequencyMin"], frequency_max=cfg["frequencyMax"],
        spectral_tilt=cfg["spectralTilt"],
    )

    delay_ref = (2 * cfg["bufferSize"] + 1) - 1
    n_warmup = -(-delay_ref // cfg["bufferSize"]) + 1

    for i in range(n_chunks):
        chunk = signal[i * cfg["bufferSize"]:(i + 1) * cfg["bufferSize"]]
        cpp_out = oracle.process(chunk)
        torch_out = module(torch.from_numpy(chunk)).numpy()
        if i >= n_warmup:
            mask = np.isfinite(cpp_out)  # True where C++ is finite; excludes -inf DC bin rows
            np.testing.assert_allclose(torch_out[mask], cpp_out[mask], atol=1e-2)


def test_constructor_rejects_unsupported_methods():
    with pytest.raises(ValueError, match="ADAPTIVE"):
        PerceptualAdaptiveSpectrogramStreaming(
            buffer_size=512, n_bands=100, n_spectrograms=3, n_folds=1, nonlinearity=0,
            sample_rate=48000.0, frequency_min=20.0, frequency_max=20000.0,
            spectral_tilt=False, method="NONLINEAR",
        )

    with pytest.raises(ValueError, match="nFolds"):
        PerceptualAdaptiveSpectrogramStreaming(
            buffer_size=512, n_bands=100, n_spectrograms=3, n_folds=2, nonlinearity=0,
            sample_rate=48000.0, frequency_min=20.0, frequency_max=20000.0,
            spectral_tilt=False,
        )
