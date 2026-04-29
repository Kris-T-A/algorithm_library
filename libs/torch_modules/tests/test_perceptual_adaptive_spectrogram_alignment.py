import numpy as np
import pytest
import torch

from torch_modules.perceptual_spectral_analysis.perceptual_adaptive_spectrogram import (
    PerceptualAdaptiveSpectrogram,
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


def test_stateless_equals_streaming_chunked():
    cfg = dict(
        buffer_size=256, n_bands=64, n_spectrograms=3, n_folds=1, nonlinearity=0,
        sample_rate=48000.0, frequency_min=20.0, frequency_max=20000.0, spectral_tilt=False,
    )
    n_chunks = 5
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((1, n_chunks * cfg["buffer_size"])).astype(np.float32))

    stateless = PerceptualAdaptiveSpectrogram(**cfg)
    out_stateless = stateless(x)

    streaming = PerceptualAdaptiveSpectrogramStreaming(**cfg)
    streaming.reset()
    chunks = []
    for i in range(n_chunks):
        chunk = x[..., i * cfg["buffer_size"]:(i + 1) * cfg["buffer_size"]]
        chunks.append(streaming(chunk, detach_state=False))
    out_streaming = torch.cat(chunks, dim=-1)

    torch.testing.assert_close(out_stateless, out_streaming, atol=1e-6, rtol=1e-6)


def test_streaming_forward_fullclip_matches_per_chunk_loop():
    """PerceptualAdaptiveSpectrogramStreaming.forward_fullclip equals chunk-by-chunk forward."""
    cfg = dict(
        buffer_size=256, n_bands=64, n_spectrograms=3, n_folds=1, nonlinearity=1,
        sample_rate=48000.0, frequency_min=20.0, frequency_max=20000.0,
        spectral_tilt=True,
    )

    for n_chunks in (1, 2, 5):
        module = PerceptualAdaptiveSpectrogramStreaming(**cfg)

        rng = np.random.default_rng(n_chunks)
        x_full = torch.from_numpy(
            rng.standard_normal((2, n_chunks * cfg["buffer_size"])).astype(np.float32)
        )

        module.reset()
        streaming_chunks = []
        for k in range(n_chunks):
            c = x_full[..., k * cfg["buffer_size"] : (k + 1) * cfg["buffer_size"]]
            streaming_chunks.append(module(c, detach_state=False))
        streaming_full = torch.cat(streaming_chunks, dim=-1)

        module.reset()
        fullclip = module.forward_fullclip(x_full)

        torch.testing.assert_close(
            fullclip, streaming_full, atol=1e-5, rtol=1e-5,
            msg=f"n_chunks={n_chunks}",
        )


def test_stateless_rejects_partial_chunks():
    cfg = dict(
        buffer_size=256, n_bands=64, n_spectrograms=3, n_folds=1, nonlinearity=0,
        sample_rate=48000.0, frequency_min=20.0, frequency_max=20000.0, spectral_tilt=False,
    )
    module = PerceptualAdaptiveSpectrogram(**cfg)
    with pytest.raises(ValueError, match="multiple of bufferSize"):
        module(torch.zeros(1, 300))
