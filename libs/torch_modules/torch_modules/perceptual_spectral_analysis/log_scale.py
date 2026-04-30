"""Logarithmic scale transform — port of `src/scale_transform/log_scale.h`."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from torch_modules.interpolation.interpolation_cubic import InterpolationCubic


def _energy_to_db(x: np.ndarray) -> np.ndarray:
    """10 * log10(x) for x ∈ (0, 1] — used to convert linear triangular-window weights to dB.

    Mirrors `10 * std::log10` used in `src/scale_transform/log_scale.h`'s constructor.
    Both implementations precompute these weights at construction time, so exact log10 is fine.
    """
    return (10.0 * np.log10(x)).astype(np.float32)


class LogScale(nn.Module):
    """Logarithmic scale transform from linear-frequency to log-frequency bins.

    Stateless. Forward expects ``(..., n_inputs)`` and returns ``(..., n_outputs)``.
    """

    def __init__(self, n_inputs: int, n_outputs: int, output_start: float, output_end: float, input_end: float):
        super().__init__()
        assert output_end <= input_end

        # Match the C++ constructor (use float64 for the log/pow conversions).
        scale = 1.0  # LOGARITHMIC mode
        min_log = math.log10(1.0 + scale * output_start)
        max_log = math.log10(1.0 + scale * output_end)
        lin_logs = np.linspace(min_log, max_log, n_outputs, dtype=np.float64)
        freq_per_bin = scale * float(input_end) / (n_inputs - 1)
        center_bins = ((np.power(10.0, lin_logs) - 1.0) / freq_per_bin).astype(np.float32)

        # nLinearBins (mirrors C++ while-loop at log_scale.h:30-33).
        n_linear_bins = 0
        while (n_linear_bins < n_outputs - 1) and (
            (center_bins[n_linear_bins + 1] - center_bins[n_linear_bins] <= 1.0)
            or (center_bins[n_linear_bins] < 1.0)
        ):
            n_linear_bins += 1

        output_start_idx = center_bins[:n_linear_bins].astype(np.int64)
        fraction_linear = center_bins[:n_linear_bins] - output_start_idx.astype(np.float32)
        linear_pair_idx = np.concatenate([output_start_idx, output_start_idx + 1]) if n_linear_bins > 0 else np.zeros(0, dtype=np.int64)

        # nCubicBins (mirrors C++ while-loop at log_scale.h:40-43).
        n_sum = n_linear_bins
        while (n_sum < n_outputs - 2) and (
            (center_bins[n_sum + 1] - center_bins[n_sum] <= 2.0)
            or (center_bins[n_sum] < 2.0)
        ):
            n_sum += 1
        n_cubic_bins = n_sum - n_linear_bins
        fraction_cubic = center_bins[n_linear_bins:n_linear_bins + n_cubic_bins].copy()

        n_triangular_bins = n_outputs - n_sum
        fraction_triangular = center_bins[n_sum:n_sum + n_triangular_bins].copy()
        distance_triangular = (
            center_bins[n_sum:n_sum + n_triangular_bins]
            - center_bins[n_sum - 1:n_sum - 1 + n_triangular_bins]
        ).copy() if n_triangular_bins > 0 else np.zeros(0, dtype=np.float32)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_linear_bins = n_linear_bins
        self.n_cubic_bins = n_cubic_bins
        self.n_triangular_bins = n_triangular_bins

        # Always register buffers (even zero-length) so device migration is consistent.
        self.register_buffer("linear_pair_idx", torch.from_numpy(linear_pair_idx), persistent=False)
        self.register_buffer("fraction_linear", torch.from_numpy(fraction_linear), persistent=False)
        self.register_buffer("fraction_cubic", torch.from_numpy(fraction_cubic), persistent=False)
        self.interpolation_cubic = InterpolationCubic()

        # Build the dense triangular weight matrix in dB (precomputed at construction; exact log10).
        # weights shape: (n_triangular_bins, n_inputs); -inf outside the triangular window.
        weights = np.full((max(n_triangular_bins, 1), n_inputs), -np.inf, dtype=np.float32)
        # The "max(.., 1)" keeps the buffer non-empty for register_buffer; we mask below.
        for i in range(n_triangular_bins):
            i_mid = int(math.ceil(fraction_triangular[i]))
            i_start = int(math.ceil(fraction_triangular[i] - distance_triangular[i]))
            if i < n_triangular_bins - 1:
                i_end = int(math.ceil(fraction_triangular[i] + distance_triangular[i + 1]))
            else:
                i_end = i_mid  # last bin has no right half (mirrors C++ at log_scale.h:124-133)

            weights[i, i_mid] = 0.0  # full weight (linear 1.0 → 0 dB)
            dist_left = float(i_mid - i_start)
            # Skip i_bin = i_start: lin_weight = 0 → log10(0) = -inf, identical to the default sentinel fill.
            for i_bin in range(i_start + 1, i_mid):
                lin_weight = 1.0 - (i_mid - i_bin) / dist_left
                weights[i, i_bin] = float(_energy_to_db(np.array([lin_weight]))[0])
            if i_end > i_mid:
                dist_right = float(i_end - i_mid)
                for i_bin in range(i_mid + 1, i_end):
                    lin_weight = np.float32(1.0 - (i_bin - i_mid) / dist_right)
                    weights[i, i_bin] = _energy_to_db(np.array([lin_weight]))[0]

        # Register a real-shape buffer (drop the dummy first row if no triangular bins).
        if n_triangular_bins == 0:
            weights = np.zeros((0, n_inputs), dtype=np.float32)
        self.register_buffer("triangular_weights", torch.from_numpy(weights[:n_triangular_bins]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_inputs)
        out_chunks = []

        # Linear region.
        if self.n_linear_bins > 0:
            pair = x.index_select(-1, self.linear_pair_idx).unflatten(-1, (2, self.n_linear_bins))
            x0, x1 = pair.unbind(dim=-2)
            linear_out = x0 + self.fraction_linear * (x1 - x0)
            out_chunks.append(linear_out)

        # Cubic region: delegate.
        if self.n_cubic_bins > 0:
            cubic_out = self.interpolation_cubic(x, self.fraction_cubic)
            out_chunks.append(cubic_out)

        # Triangular region: max over (input + weights_dB) along input dim.
        # The naive `shifted = x.unsqueeze(-2) + triangular_weights` materializes a
        # (..., n_tri, n_inputs) tensor — for spectrogram-shaped inputs at training
        # batch sizes that's multi-GB. Chunk along the flattened leading dims so the
        # intermediate tensor stays bounded; amax(dim=-1) is per-row so chunking is
        # numerically identical to the unchunked path.
        if self.n_triangular_bins > 0:
            out_chunks.append(self._triangular_amax(x))

        return torch.cat(out_chunks, dim=-1)

    # Target ceiling for the intermediate `shifted` tensor, in bytes. 128 MB is small
    # enough to fit comfortably inside any modern training budget, large enough that
    # loop overhead is negligible relative to the matmul-equivalent broadcast.
    _TRI_CHUNK_BYTES: int = 128 * 1024 * 1024

    def _triangular_amax(self, x: torch.Tensor) -> torch.Tensor:
        """Compute `(x.unsqueeze(-2) + triangular_weights).max(dim=-1)[0]` chunked
        along the flattened leading dims so the intermediate stays bounded.

        Uses `.max(dim=-1)[0]` rather than `.amax(dim=-1)` because PyTorch's amax
        backward saves the full pre-reduction tensor (~6 GB at training shapes),
        while max(dim) backward saves only the argmax indices and scatters via
        `value_selecting_reduction_backward`. Numerical result is identical.
        """
        leading = x.shape[:-1]
        n_inputs = x.shape[-1]
        flat = x.reshape(-1, n_inputs)  # (N, n_inputs)
        N = flat.shape[0]
        n_tri = self.triangular_weights.shape[0]
        bytes_per_row = n_tri * n_inputs * x.element_size()
        chunk = max(1, self._TRI_CHUNK_BYTES // max(1, bytes_per_row))
        if chunk >= N:
            shifted = flat.unsqueeze(-2) + self.triangular_weights
            out_flat = shifted.max(dim=-1)[0]
        else:
            outs = []
            for i in range(0, N, chunk):
                sub = flat[i:i + chunk]
                shifted = sub.unsqueeze(-2) + self.triangular_weights
                outs.append(shifted.max(dim=-1)[0])
            out_flat = torch.cat(outs, dim=0)
        return out_flat.reshape(*leading, n_tri)
