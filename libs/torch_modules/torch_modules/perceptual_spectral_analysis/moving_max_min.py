"""Moving max-min cascades — vertical (stateless) and horizontal (stateful).

Both implement a cascaded max-pool(L) → min-pool(L) over a 2D tensor along one axis.
Vertical: stateless, edge-replicate padding (matches the C++ per-call init pattern).
Horizontal: causal/streaming with circular-buffer-equivalent prev-state tensors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MovingMaxMinVertical(nn.Module):
    """Stateless vertical moving max-min cascade with edge-replicate boundary.

    Forward expects ``(..., n_channels, n_cols)`` and returns the same shape.
    The cascade is along axis -2 (the n_channels axis).

    Equivalent to: F.pad(x, (L-1, L-1), mode='replicate') along axis -2,
    then cascaded max-pool(L) → min-pool(L), kernel L stride 1. The output
    length equals the input length.
    """

    def __init__(self, filter_length: int):
        super().__init__()
        assert filter_length > 0
        self.filter_length = filter_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.filter_length
        if L == 1:
            return x

        # F.max_pool1d operates on the last dim, so transpose, flatten leading dims,
        # pool, then unflatten and transpose back.
        x_t = x.transpose(-1, -2)  # (..., n_cols, n_channels)
        leading = x_t.shape[:-1]
        x_flat = x_t.reshape(-1, x_t.shape[-1]).unsqueeze(1)  # (B*, 1, n_channels)

        padded = F.pad(x_flat, (L - 1, L - 1), mode="replicate")
        max_out = F.max_pool1d(padded, kernel_size=L, stride=1)
        min_out = -F.max_pool1d(-max_out, kernel_size=L, stride=1)

        out = min_out.squeeze(1).reshape(*leading, -1).transpose(-1, -2)
        return out


class MovingMaxMinHorizontal(nn.Module):
    """Causal horizontal moving max-min cascade with streaming state.

    Forward expects ``(..., n_channels, T)``. Per-call ``T >= L - 1`` is required
    (private invariant — not validated; the calling pattern in this codebase
    always uses ``T = L``). State buffers (``prev_input``, ``prev_max_out``)
    are lazily allocated on the first forward to match the input's flattened
    batch shape ``B*``.

    Cascade: max-pool over [prev_input, x] (kernel L) → min-pool over
    [prev_max_out, max_out] (kernel L). Both pools stride 1.
    """

    def __init__(self, filter_length: int, n_channels: int):
        super().__init__()
        assert filter_length > 0
        self.filter_length = filter_length
        self.n_channels = n_channels
        # Empty placeholders so .to(device) works before first forward; reallocated lazily.
        self.register_buffer("prev_input", torch.empty(0), persistent=False)
        self.register_buffer("prev_max_out", torch.empty(0), persistent=False)
        self._allocated_batch_shape: tuple[int, ...] | None = None

    def reset(self) -> None:
        """Drop state allocation; the next forward re-allocates from the input."""
        self.prev_input = torch.empty(0, device=self.prev_input.device, dtype=self.prev_input.dtype)
        self.prev_max_out = torch.empty(0, device=self.prev_max_out.device, dtype=self.prev_max_out.dtype)
        self._allocated_batch_shape = None

    def _allocate(self, batch_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> None:
        L = self.filter_length
        shape = (*batch_shape, self.n_channels, L - 1)
        self.prev_input = torch.full(shape, float("-inf"), device=device, dtype=dtype)
        self.prev_max_out = torch.full(shape, float("inf"), device=device, dtype=dtype)
        self._allocated_batch_shape = tuple(batch_shape)

    def forward(self, x: torch.Tensor, *, detach_state: bool = True) -> torch.Tensor:
        L = self.filter_length
        if L == 1:
            return x  # cascade collapses to identity

        assert x.shape[-2] == self.n_channels, (
            f"expected n_channels={self.n_channels}, got {x.shape[-2]}"
        )
        batch_shape = tuple(x.shape[:-2])

        if self._allocated_batch_shape is None:
            self._allocate(batch_shape, x.device, x.dtype)
        elif self._allocated_batch_shape != batch_shape:
            raise ValueError(
                f"MovingMaxMinHorizontal: batch shape mismatch. "
                f"Allocated for {self._allocated_batch_shape}, got {batch_shape}. "
                "Call reset() first to rebind."
            )

        # Stage 1: max over [prev_input, x] with kernel L, stride 1.
        max_input = torch.cat([self.prev_input, x], dim=-1)  # (..., n_channels, L-1+T)
        flat = max_input.reshape(-1, self.n_channels, max_input.shape[-1])
        max_out = F.max_pool1d(flat, kernel_size=L, stride=1)  # (B*, n_channels, T)
        max_out = max_out.reshape(*batch_shape, self.n_channels, -1)

        # Stage 2: min over [prev_max_out, max_out] with kernel L, stride 1.
        min_input = torch.cat([self.prev_max_out, max_out], dim=-1)
        flat2 = min_input.reshape(-1, self.n_channels, min_input.shape[-1])
        out = -F.max_pool1d(-flat2, kernel_size=L, stride=1)
        out = out.reshape(*batch_shape, self.n_channels, -1)

        # Update state: trailing L-1 columns of each concatenated stream.
        new_prev_input = max_input[..., -(L - 1):]
        new_prev_max_out = min_input[..., -(L - 1):]
        if detach_state:
            new_prev_input = new_prev_input.detach()
            new_prev_max_out = new_prev_max_out.detach()
        self.prev_input = new_prev_input
        self.prev_max_out = new_prev_max_out

        return out
