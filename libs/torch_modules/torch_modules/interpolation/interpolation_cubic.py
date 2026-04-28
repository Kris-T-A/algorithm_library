"""Cubic Hermite (Catmull-Rom) interpolation, vectorized across leading dims.

Matches `src/interpolation/interpolation_cubic.h` (4-point, 3rd order Hermite).
"""
from __future__ import annotations

import torch
from torch import nn


class InterpolationCubic(nn.Module):
    """Cubic Hermite (Catmull-Rom) interpolation, vectorized across leading dims.

    Forward signature::

        out = module(src, indices)

    where ``src`` has shape ``(..., n_inputs)`` and ``indices`` is a 1D tensor of
    fractional positions ``∈ [1.0, n_inputs - 2.0]`` into ``src``'s last dim.
    Output shape: ``(..., n_outputs)``.
    """

    def forward(self, src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # indices are guaranteed >= 1, so .long() truncates equivalently to floor.
        i = indices.long()
        f = indices - i.to(indices.dtype)

        n = indices.shape[0]
        offsets = torch.stack((i - 1, i, i + 1, i + 2), dim=0).reshape(-1)
        s = src.index_select(-1, offsets).unflatten(-1, (4, n))
        s0, s1, s2, s3 = s.unbind(dim=-2)

        f2 = f * f
        f3 = f2 * f
        w0 = -0.5 * f + f2 - 0.5 * f3
        w1 = 1.0 - 2.5 * f2 + 1.5 * f3
        w2 = 0.5 * f + 2.0 * f2 - 1.5 * f3
        w3 = -0.5 * f2 + 0.5 * f3
        return w0 * s0 + w1 * s1 + w2 * s2 + w3 * s3
