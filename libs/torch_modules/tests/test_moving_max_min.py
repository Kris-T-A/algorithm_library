import numpy as np
import pytest
import torch

from torch_modules.perceptual_spectral_analysis.moving_max_min import (
    MovingMaxMinHorizontal,
    MovingMaxMinVertical,
)


def _numpy_oracle_horizontal(x: np.ndarray, L: int) -> np.ndarray:
    """Re-implement `src/moving_max_min/moving_max_min_horizontal.h` in NumPy.

    x has shape (n_channels, n_samples). Returns shape (n_channels, n_samples).
    """
    n_channels, n_samples = x.shape
    max_buf = np.full((n_channels, L), -np.inf, dtype=np.float32)
    min_buf = np.full((n_channels, L), np.inf, dtype=np.float32)
    out = np.empty_like(x)
    counter = 0
    for s in range(n_samples):
        max_buf[:, counter] = x[:, s]
        min_buf[:, counter] = max_buf.max(axis=1)
        out[:, s] = min_buf.min(axis=1)
        counter = (counter + 1) % L
    return out


def _numpy_oracle_vertical(x: np.ndarray, L: int) -> np.ndarray:
    """Edge-replicate pad by (L-1, L-1) on axis 0, then cascaded max-pool(L) -> min-pool(L).

    At L=1 the C++ has a peculiarity: output[i] = input[i+1] for i < N-1,
    output[N-1] = input[N-1] (a one-sample shift with last-row replication).
    See src/moving_max_min/moving_max_min_vertical.h:48-83.
    """
    if L == 1:
        # C++ peculiarity: one-sample shift with last-row replication.
        out = np.empty_like(x)
        out[:-1] = x[1:]
        out[-1] = x[-1]
        return out
    padded = np.pad(x, ((L - 1, L - 1), (0, 0)), mode="edge")
    max_out = np.lib.stride_tricks.sliding_window_view(padded, L, axis=0).max(axis=-1)
    min_out = np.lib.stride_tricks.sliding_window_view(max_out, L, axis=0).min(axis=-1)
    return min_out


@pytest.mark.parametrize("L", [1, 2, 3, 4, 8])
def test_horizontal_full_clip_matches_numpy_oracle(L):
    n_channels = 7
    n_samples = 50
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_channels, n_samples)).astype(np.float32)

    expected = _numpy_oracle_horizontal(x, L)

    module = MovingMaxMinHorizontal(filter_length=L, n_channels=n_channels)
    out = module(torch.from_numpy(x)).numpy()

    np.testing.assert_allclose(out, expected, atol=0)


@pytest.mark.parametrize("L", [1, 2, 4, 8])
def test_horizontal_chunked_equals_full(L):
    """Streaming property: chunked + state carryover == full pass."""
    n_channels = 5
    n_samples = 32
    chunk = 8
    rng = np.random.default_rng(1)
    x = rng.standard_normal((n_channels, n_samples)).astype(np.float32)

    module = MovingMaxMinHorizontal(filter_length=L, n_channels=n_channels)
    chunks = []
    for i in range(0, n_samples, chunk):
        chunks.append(module(torch.from_numpy(x[:, i:i + chunk])))
    chunked = torch.cat(chunks, dim=-1).numpy()

    module.reset()
    full = module(torch.from_numpy(x)).numpy()

    np.testing.assert_allclose(chunked, full, atol=0)


@pytest.mark.parametrize("L", [1, 2, 3, 4, 8])
def test_vertical_matches_pad_replicate_oracle(L):
    n_channels = 30
    n_cols = 4
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_channels, n_cols)).astype(np.float32)

    expected = _numpy_oracle_vertical(x, L)

    module = MovingMaxMinVertical(filter_length=L)
    out = module(torch.from_numpy(x)).numpy()

    np.testing.assert_allclose(out, expected, atol=0)
