"""Logarithmic scale transform — port of `src/scale_transform/log_scale.h`."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from torch_modules.interpolation.interpolation_cubic import InterpolationCubic


def _energy_to_db(x: np.ndarray) -> np.ndarray:
    """10 * log10(x + 1e-16) for x in [0, 1].

    Mirrors `10 * std::log10(linWeight + 1e-16f)` used in
    `src/scale_transform/log_scale.h`'s constructor.
    """
    return (10.0 * np.log10(x + np.float32(1e-16))).astype(np.float32)


def _cpp_round_positive(x: float) -> int:
    """Match `std::round` for the non-negative center-bin values used here."""
    return int(math.floor(x + 0.5))


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

        # nLinearBins (mirrors C++ while-loop at log_scale.h:29-32).
        n_linear_bins = 0
        while (n_linear_bins < n_outputs - 1) and (center_bins[n_linear_bins] < 1.0):
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

        triangular_starts = []
        triangular_weight_rows = []
        for i in range(n_triangular_bins):
            c_start = float(center_bins[n_sum + i - 1])
            c_mid = float(center_bins[n_sum + i])
            if i < n_triangular_bins - 1:
                c_end = float(center_bins[n_sum + i + 1])
            else:
                c_end = float(_cpp_round_positive(c_mid) + 1)

            i_start = int(math.ceil(c_start))
            i_mid = _cpp_round_positive(c_mid)
            i_end = int(math.ceil(c_end))

            triangular_starts.append(i_start)
            row = np.empty(i_end - i_start, dtype=np.float32)
            for i_bin in range(i_start, i_mid):
                lin_weight = np.float32(1.0 - (c_mid - i_bin) / (c_mid - c_start))
                row[i_bin - i_start] = float(_energy_to_db(np.array([lin_weight]))[0])
            row[i_mid - i_start] = 0.0  # full linear weight = 1.0 -> 0 dB
            for i_bin in range(i_mid + 1, i_end):
                lin_weight = np.float32(1.0 - (i_bin - c_mid) / (c_end - c_mid))
                row[i_bin - i_start] = _energy_to_db(np.array([lin_weight]))[0]
            triangular_weight_rows.append(row)

        # Store the same per-window ranges as C++, padded only to the widest window so
        # PyTorch can gather/reduce them as one tensor.
        if n_triangular_bins == 0:
            triangular_idx = np.zeros((0, 0), dtype=np.int64)
            weights = np.zeros((0, 0), dtype=np.float32)
        else:
            max_width = max(row.shape[0] for row in triangular_weight_rows)
            starts = np.asarray(triangular_starts, dtype=np.int64)
            offsets = np.arange(max_width, dtype=np.int64)[None, :]
            lengths = np.asarray([row.shape[0] for row in triangular_weight_rows], dtype=np.int64)[:, None]
            triangular_idx = starts[:, None] + offsets
            triangular_idx = np.where(offsets < lengths, triangular_idx, 0)
            weights = np.full((n_triangular_bins, max_width), -np.inf, dtype=np.float32)
            for i, row in enumerate(triangular_weight_rows):
                weights[i, :row.shape[0]] = row
        self.register_buffer("triangular_idx", torch.from_numpy(triangular_idx), persistent=False)
        self.register_buffer("triangular_weights", torch.from_numpy(weights), persistent=False)

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

        # Triangular region: max over the same compact per-bin windows as C++.
        # Chunk along the flattened leading dims so the intermediate stays bounded.
        if self.n_triangular_bins > 0:
            out_chunks.append(self._triangular_amax(x))

        return torch.cat(out_chunks, dim=-1)

    # Target ceiling for the intermediate `shifted` tensor, in bytes. 128 MB is small
    # enough to fit comfortably inside any modern training budget, large enough that
    # loop overhead is negligible relative to the matmul-equivalent broadcast.
    _TRI_CHUNK_BYTES: int = 128 * 1024 * 1024

    def _triangular_amax(self, x: torch.Tensor) -> torch.Tensor:
        """Compute each C++ triangular window chunked along the flattened leading dims.

        Uses `.max(dim=-1)[0]` rather than `.amax(dim=-1)` because PyTorch's amax
        backward saves the full pre-reduction tensor (~6 GB at training shapes),
        while max(dim) backward saves only the argmax indices and scatters via
        `value_selecting_reduction_backward`. Numerical result is identical.
        """
        leading = x.shape[:-1]
        n_inputs = x.shape[-1]
        flat = x.reshape(-1, n_inputs)  # (N, n_inputs)
        N = flat.shape[0]
        n_tri, window_width = self.triangular_weights.shape
        bytes_per_row = n_tri * window_width * x.element_size()
        chunk = max(1, self._TRI_CHUNK_BYTES // max(1, bytes_per_row))
        if chunk >= N:
            gathered = flat[:, self.triangular_idx]
            out_flat = (gathered + self.triangular_weights).max(dim=-1)[0]
        else:
            outs = []
            for i in range(0, N, chunk):
                sub = flat[i:i + chunk]
                gathered = sub[:, self.triangular_idx]
                outs.append((gathered + self.triangular_weights).max(dim=-1)[0])
            out_flat = torch.cat(outs, dim=0)
        return out_flat.reshape(*leading, n_tri)
