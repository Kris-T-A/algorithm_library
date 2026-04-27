"""Cubic Hermite (Catmull-Rom) interpolation, vectorized across leading dims.

Matches `src/interpolation/interpolation_cubic.h` (4-point, 3rd order Hermite).
"""
from __future__ import annotations

import torch
from torch import nn


class InterpolationCubic(nn.Module):
    """Cubic Hermite interpolation, vectorized across leading dims.

    Forward signature::

        out = module(src, indices)

    where ``src`` has shape ``(..., n_inputs)`` and ``indices`` is a 1D tensor of
    fractional positions ``∈ [1.0, n_inputs - 2.0]`` into ``src``'s last dim.
    Output shape: ``(..., n_outputs)``.

    Mirrors the C++ formula: index = floor(fractional_index); f = fractional - index;
    samples = src[index-1 .. index+2]; output = c3*f^3 + c2*f^2 + c1*f + c0
    where c0..c3 are the Hermite cubic coefficients from
    src/interpolation/interpolation_cubic.h:21-30.
    """

    def forward(self, src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # No bounds clamping — assume caller respects [1, n-2] (matches the C++ asserts).
        i = torch.floor(indices).to(torch.long)  # (n_outputs,)
        f = indices - i.to(indices.dtype)        # (n_outputs,) fractional part

        # Gather 4 samples per output index. src.index_select on the last dim.
        s0 = src.index_select(-1, i - 1)
        s1 = src.index_select(-1, i)
        s2 = src.index_select(-1, i + 1)
        s3 = src.index_select(-1, i + 2)

        c0 = s1
        c1 = 0.5 * (s2 - s0)
        c2 = s0 - 2.5 * s1 + 2.0 * s2 - 0.5 * s3
        c3 = 0.5 * (s3 - s0) + 1.5 * (s1 - s2)
        return ((c3 * f + c2) * f + c1) * f + c0
