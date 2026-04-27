"""Logarithmic scale transform — port of `src/scale_transform/log_scale.h`."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from torch_modules.interpolation.interpolation_cubic import InterpolationCubic


def _fasterlog2(x: np.ndarray) -> np.ndarray:
    """Bit-twiddling fast log2 — exact mirror of `fasterlog2` in
    `src/utilities/fastonebigheader.h` lines 199-208.

    The C++ union-punning reads the float bits as uint32, multiplies by
    ``1.1920928955078125e-7`` (= 2^-23) and subtracts ``126.94269504``.
    """
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32).astype(np.float32)
    return bits * np.float32(1.1920928955078125e-7) - np.float32(126.94269504)


def _energy_to_db(x: np.ndarray) -> np.ndarray:
    """Mirror C++ ``energy2dB(x) = 3.010299956639812f * fasterlog2(x)``.

    Uses the same bit-twiddling fast log2 as the C++ side so that
    pre-computed triangular weights match the C++ oracle exactly.
    """
    return (np.float32(3.010299956639812) * _fasterlog2(x)).astype(np.float32)


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

        output_start_idx = center_bins[:n_linear_bins].astype(np.int32)
        fraction_linear = center_bins[:n_linear_bins] - output_start_idx.astype(np.float32)

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
        self.register_buffer("output_start_idx", torch.from_numpy(output_start_idx).long(), persistent=False)
        self.register_buffer("fraction_linear", torch.from_numpy(fraction_linear), persistent=False)
        self.register_buffer("fraction_cubic", torch.from_numpy(fraction_cubic), persistent=False)
        self.interpolation_cubic = InterpolationCubic()

        # Build the dense triangular weight matrix in dB.
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
            for i_bin in range(i_start, i_mid):
                # Linear weight: 1 - |i_bin - i_mid| / dist_left, in (0, 1)
                lin_weight = np.float32(1.0 - (i_mid - i_bin) / dist_left)
                weights[i, i_bin] = _energy_to_db(np.array([lin_weight]))[0]
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
            x0 = x.index_select(-1, self.output_start_idx)
            x1 = x.index_select(-1, self.output_start_idx + 1)
            f = self.fraction_linear
            linear_out = (1.0 - f) * x0 + f * x1
            out_chunks.append(linear_out)

        # Cubic region: delegate.
        if self.n_cubic_bins > 0:
            cubic_out = self.interpolation_cubic(x, self.fraction_cubic)
            out_chunks.append(cubic_out)

        # Triangular region: max over (input + weights_dB) along input dim.
        if self.n_triangular_bins > 0:
            shifted = x.unsqueeze(-2) + self.triangular_weights  # (..., n_tri, n_inputs)
            tri_out, _ = shifted.max(dim=-1)
            out_chunks.append(tri_out)

        return torch.cat(out_chunks, dim=-1)
