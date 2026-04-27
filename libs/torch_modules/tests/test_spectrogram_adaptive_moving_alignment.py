"""Alignment and self-consistency tests for SpectrogramAdaptiveMoving.

Alignment tests compare the torch port against the C++ oracle
(``PythonAlgorithmLibrary.SpectrogramAdaptive`` with ``method="Moving"``).
Self-consistency tests verify that reset + re-run reproduces identical output.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from torch_modules.perceptual_spectral_analysis.spectrogram_adaptive_moving import (
    SpectrogramAdaptiveMoving,
)


@pytest.mark.parametrize("nonlinearity", [0, 1])
def test_per_chunk_alignment(pal, nonlinearity):
    cfg = dict(
        bufferSize=512,
        nBands=2 * 512 + 1,  # internal nBands used directly
        nSpectrograms=3,
        nFolds=1,
        nonlinearity=nonlinearity,
        sampleRate=48000.0,
    )
    n_chunks = 8
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(n_chunks * cfg["bufferSize"]).astype(np.float32)

    oracle = pal.SpectrogramAdaptive()
    oracle.setCoefficients({**cfg, "method": "Moving"})

    module = SpectrogramAdaptiveMoving(
        buffer_size=cfg["bufferSize"],
        n_bands=cfg["nBands"],
        n_spectrograms=cfg["nSpectrograms"],
        n_folds=cfg["nFolds"],
        nonlinearity=cfg["nonlinearity"],
    )

    # delay_ref = frame_size // 2 = (2 * (n_bands - 1)) // 2 = n_bands - 1
    delay_ref = cfg["nBands"] - 1
    # warmup: enough chunks to flush the initial sentinel values
    n_warmup = -(-delay_ref // cfg["bufferSize"]) + 1  # ceiling division + 1

    max_diffs = []
    for i in range(n_chunks):
        chunk = signal[i * cfg["bufferSize"] : (i + 1) * cfg["bufferSize"]]
        cpp_out = oracle.process(chunk)  # shape (n_bands, 2^(n_spectrograms-1))
        torch_out = module(torch.from_numpy(chunk)).numpy()
        if i >= n_warmup:
            diff = np.abs(torch_out - cpp_out).max()
            max_diffs.append(diff)
            np.testing.assert_allclose(
                torch_out,
                cpp_out,
                atol=1e-2,
                err_msg=f"Mismatch at chunk {i}, nonlinearity={nonlinearity}, max diff={diff:.4f}",
            )

    if max_diffs:
        print(
            f"\nnonlinearity={nonlinearity}: max abs diff over post-warmup chunks = {max(max_diffs):.4e}"
        )


def test_full_clip_self_consistency():
    """Same module run on chunks vs. reset+re-run agrees with itself."""
    cfg = dict(
        buffer_size=256,
        n_bands=513,
        n_spectrograms=3,
        n_folds=1,
        nonlinearity=0,
    )
    n_chunks = 4
    rng = np.random.default_rng(7)
    signal = rng.standard_normal(n_chunks * cfg["buffer_size"]).astype(np.float32)

    module = SpectrogramAdaptiveMoving(**cfg)
    chunks_out = []
    for i in range(n_chunks):
        chunk = signal[i * cfg["buffer_size"] : (i + 1) * cfg["buffer_size"]]
        chunks_out.append(module(torch.from_numpy(chunk)))
    out_streaming = torch.cat(chunks_out, dim=-1)

    module.reset()
    out_full_chunks = []
    for i in range(n_chunks):
        chunk = signal[i * cfg["buffer_size"] : (i + 1) * cfg["buffer_size"]]
        out_full_chunks.append(module(torch.from_numpy(chunk)))
    out_full = torch.cat(out_full_chunks, dim=-1)

    torch.testing.assert_close(out_streaming, out_full)


def test_forward_rejects_non_floating_dtype():
    module = SpectrogramAdaptiveMoving(buffer_size=64, n_bands=129, n_spectrograms=3, n_folds=1, nonlinearity=0)
    with pytest.raises(ValueError, match="floating dtype"):
        module(torch.zeros(64, dtype=torch.long))
