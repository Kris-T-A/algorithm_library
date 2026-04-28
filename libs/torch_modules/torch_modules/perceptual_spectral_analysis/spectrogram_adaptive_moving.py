"""Port of ``SpectrogramAdaptiveMoving`` from C++ to PyTorch.

Implements the FFT-cascade spectrogram with progressive 2x upscaling and
moving-min combination. Mirrors the state layout and algorithm of the C++
``SpectrogramAdaptiveMoving`` class.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from torch_modules.perceptual_spectral_analysis.moving_max_min import (
    MovingMaxMinHorizontal,
)


def _build_hann_periodic(n: int) -> np.ndarray:
    """Periodic Hann window of length ``n`` (matches C++ ``hann()`` in functions.h).

    Equivalent to ``np.hanning(n + 1)[:-1]``.  Uses the formula
    ``0.5 * (1 - cos(2*pi*k/n))`` for k in [0, n).
    """
    k = np.arange(n, dtype=np.float64)
    return (0.5 * (1.0 - np.cos(2.0 * np.pi * k / n))).astype(np.float32)


def _build_windows(
    n_bands: int, n_spectrograms: int, nonlinearity: int
) -> tuple[dict[tuple[int, int], np.ndarray], float, int]:
    """Build per-(level, filterbank) analysis windows.

    Returns ``(windows, win_scale, n_filterbanks)`` where:
    - ``windows[(level, fb)]`` is a float32 ndarray of length ``frame_size``
      sum-normalised to ``win_scale``.
    - ``win_scale`` is the sum of the level-0 FB-0 base window.
    - ``n_filterbanks`` is 1 (nonlinearity == 0) or 3.

    Mirrors the window construction in:
    - ``FilterbankShared::getAnalysisWindow`` (filterbank.cpp:61-67)
    - ``SpectrogramNonlinear`` constructor (spectrogram_nonlinear.h:25-47)
    - ``SpectrogramSetZeropad`` constructor (spectrogram_set_zeropad.h:79-105)
    """
    frame_size = 2 * (n_bands - 1)  # nFolds=1 → frameSize = 2*(nBands-1)
    n_filterbanks = 1 if nonlinearity == 0 else 3

    # Level-0 FB-0: getAnalysisWindow → hann(frameSize) / (frameSize / 4)
    base = _build_hann_periodic(frame_size) / (frame_size / 4.0)
    win_scale = float(base.sum())  # SpectrogramSetZeropad uses sum, not energy

    windows: dict[tuple[int, int], np.ndarray] = {}

    if n_filterbanks == 1:
        # Only FB 0 at every level.
        windows[(0, 0)] = base.copy()

        for iSG in range(1, n_spectrograms):
            stride = 1 << iSG
            win_small_size = frame_size // stride
            # Map(window.data, winSmallSize, InnerStride(stride)) = window[::stride][:winSmallSize]
            small = base[::stride][:win_small_size]
            window = np.zeros(frame_size, dtype=np.float32)
            window[-win_small_size:] = small
            window_sum = window.sum()
            window *= win_scale / window_sum
            windows[(iSG, 0)] = window

    else:
        # Asymmetric windows built by SpectrogramNonlinear for level 0.
        stride_nl = 1 << nonlinearity
        frame_size_small = frame_size // stride_nl
        # windowSmall = base[::stride_nl][:frameSizeSmall]
        window_small_nl = base[::stride_nl][:frame_size_small]

        # FB 1: asymmetric left — zero the left quarter, fill with left half of small window.
        # C++: window.head((frameSize-frameSizeSmall)/2).setZero()
        #       window.segment((frameSize-frameSizeSmall)/2, frameSizeSmall/2) = windowSmall.head(frameSizeSmall/2)
        zero_part = (frame_size - frame_size_small) // 2
        win1 = base.copy()
        win1[:zero_part] = 0.0
        win1[zero_part : zero_part + frame_size_small // 2] = window_small_nl[: frame_size_small // 2]
        # (Energy normalization from SpectrogramNonlinear is skipped — it gets
        #  overridden by SpectrogramSetZeropad's sum-normalization below.)

        # FB 2: asymmetric right — zero the right quarter, fill with right half of small window.
        # C++: window.tail((frameSize-frameSizeSmall)/2).setZero()
        #       window.segment(frameSize/2, frameSizeSmall/2) = windowSmall.tail(frameSizeSmall/2)
        win2 = base.copy()
        win2[frame_size - zero_part :] = 0.0
        win2[frame_size // 2 : frame_size // 2 + frame_size_small // 2] = window_small_nl[
            -(frame_size_small // 2) :
        ]

        # Sum-normalize all level-0 filterbanks to win_scale.
        windows[(0, 0)] = (base * (win_scale / base.sum())).astype(np.float32)
        windows[(0, 1)] = (win1 * (win_scale / win1.sum())).astype(np.float32)
        windows[(0, 2)] = (win2 * (win_scale / win2.sum())).astype(np.float32)

        # Levels iSG > 0: stride the level-0 PRE-sum-norm source window for each FB,
        # place at tail of zeros, then sum-normalize.
        # The source is the window AS BUILT BY SpectrogramNonlinear (before SpectrogramSetZeropad
        # sum-normalises it), which is the same at every level since bufferSize doesn't affect
        # the window for nFolds=1.
        src_windows = [base, win1, win2]
        for iSG in range(1, n_spectrograms):
            stride = 1 << iSG
            win_small_size = frame_size // stride
            for fb_idx, src in enumerate(src_windows):
                small = src[::stride][:win_small_size]
                window = np.zeros(frame_size, dtype=np.float32)
                window[-win_small_size:] = small
                window *= win_scale / window.sum()
                windows[(iSG, fb_idx)] = window.astype(np.float32)

    return windows, win_scale, n_filterbanks


class SpectrogramAdaptiveMoving(nn.Module):
    """Streaming adaptive spectrogram via FFT cascade + progressive upscaling.

    Mirrors ``SpectrogramAdaptiveMoving`` in
    ``src/spectrogram_adaptive/spectrogram_adaptive_moving.h``.

    Parameters
    ----------
    buffer_size:
        Input frame length (samples).  Must be a power of two.
    n_bands:
        Number of FFT output bands = ``frameSize/2 + 1 = n_bands``.
        For the public API this is ``2*bufferSize + 1``; for internal use
        it can be set independently.
    n_spectrograms:
        Number of cascade levels (default 3 → output has 4 columns).
    n_folds:
        Number of FFT folds (only 1 is supported).
    nonlinearity:
        Asymmetric-window nonlinearity order (0 = off, 1 = 2× time-res).

    Forward
    -------
    Input:  ``(..., buffer_size)``
    Output: ``(..., n_bands, 2^(n_spectrograms-1))`` in dB.
    """

    def __init__(
        self,
        *,
        buffer_size: int,
        n_bands: int,
        n_spectrograms: int = 3,
        n_folds: int = 1,
        nonlinearity: int = 0,
    ) -> None:
        super().__init__()

        if n_folds != 1:
            raise ValueError(f"only n_folds=1 is supported in v1, got {n_folds}")

        self.buffer_size = buffer_size
        self.n_bands = n_bands
        self.n_spectrograms = n_spectrograms
        self.n_folds = n_folds
        self.nonlinearity = nonlinearity

        # frame_size = nFolds * 2 * (nBands - 1) = 2 * (n_bands - 1)
        self.frame_size = 2 * (n_bands - 1)

        # Build analysis windows and register as buffers.
        win_dict, self._win_scale, self.n_filterbanks = _build_windows(
            n_bands, n_spectrograms, nonlinearity
        )
        for (level, fb), w in win_dict.items():
            self.register_buffer(
                f"window_{level}_{fb}", torch.from_numpy(w)
            )

        # Per-level MovingMaxMinHorizontal instances (levels 1..n_spectrograms-1).
        # Level iFB (0-indexed) has filter_length = 2^(iFB+1).
        self.moving_max_min = nn.ModuleList(
            [
                MovingMaxMinHorizontal(
                    filter_length=1 << (iFB + 1),
                    n_channels=n_bands,
                )
                for iFB in range(n_spectrograms - 1)
            ]
        )

        # Pre-compute spectrogram_buffer column counts for each level 1..n_spectrograms-1.
        # C++: nCols = 1 + (delayRef - delay) / bufferSize_i
        # (the +positivePow2(i)-1 and -positivePow2(i)+1 cancel out)
        delay_ref = self.frame_size // 2  # = n_bands - 1
        self.spectrogram_buffer_cols: list[int] = []
        for iSG in range(1, n_spectrograms):
            buf_size_iSG = buffer_size >> iSG
            delay_iSG = delay_ref >> iSG
            n_cols = 1 + (delay_ref - delay_iSG) // buf_size_iSG
            self.spectrogram_buffer_cols.append(n_cols)

        # Register placeholder buffers for spectrogram_buffer (indices 0..n_spectrograms-2).
        # They will be properly allocated on first forward.
        for i in range(n_spectrograms - 1):
            self.register_buffer(f"spectrogram_buffer_{i}", torch.empty(0), persistent=False)

        # left_boundaries: (B*, n_bands, n_spectrograms-1) — placeholder.
        self.register_buffer("left_boundaries", torch.empty(0), persistent=False)

        # Time buffers: per (level, fb) — placeholder empty buffers; lazy-allocated.
        for iSG in range(n_spectrograms):
            for fb in range(self.n_filterbanks):
                self.register_buffer(f"time_buffer_{iSG}_{fb}", torch.empty(0), persistent=False)

        self._allocated_batch_shape: tuple[int, ...] | None = None

        # Validate shift_cols > 0 for all cascade levels at construction time.
        for iFB in range(n_spectrograms - 1):
            n_cols = self.spectrogram_buffer_cols[iFB]
            new_cols = 1 << (iFB + 1)
            shift_cols = n_cols - new_cols
            if shift_cols <= 0:
                raise ValueError(
                    f"shift_cols={shift_cols} <= 0 at iFB={iFB}; "
                    f"spectrogram_buffer_cols={n_cols}, new_cols={new_cols}. "
                    "Check buffer_size, n_bands, and n_spectrograms."
                )

    # ------------------------------------------------------------------
    # Helpers for buffer registration
    # ------------------------------------------------------------------

    def _get_spectrogram_buffer(self, i: int) -> torch.Tensor:
        return getattr(self, f"spectrogram_buffer_{i}")

    def _set_spectrogram_buffer(self, i: int, value: torch.Tensor) -> None:
        setattr(self, f"spectrogram_buffer_{i}", value)

    def _get_time_buffer(self, iSG: int, fb: int) -> torch.Tensor:
        return getattr(self, f"time_buffer_{iSG}_{fb}")

    def _set_time_buffer(self, iSG: int, fb: int, value: torch.Tensor) -> None:
        setattr(self, f"time_buffer_{iSG}_{fb}", value)

    # ------------------------------------------------------------------
    # Window accessors
    # ------------------------------------------------------------------

    def _get_window(self, level: int, fb: int) -> torch.Tensor:
        return getattr(self, f"window_{level}_{fb}")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop all state; the next forward re-allocates from the input."""
        self._allocated_batch_shape = None
        # Reset time buffers to empty placeholder.
        for iSG in range(self.n_spectrograms):
            for fb in range(self.n_filterbanks):
                buf = self._get_time_buffer(iSG, fb)
                self._set_time_buffer(iSG, fb, torch.empty(0, device=buf.device, dtype=buf.dtype))
        # Reset spectrogram buffers to empty placeholder.
        for i in range(self.n_spectrograms - 1):
            buf = self._get_spectrogram_buffer(i)
            self._set_spectrogram_buffer(
                i, torch.empty(0, device=buf.device, dtype=buf.dtype)
            )
        # Reset left_boundaries.
        self.left_boundaries = torch.empty(
            0,
            device=self.left_boundaries.device,
            dtype=self.left_boundaries.dtype,
        )
        # Reset all MovingMaxMinHorizontal instances.
        for mmm in self.moving_max_min:
            mmm.reset()

    def _allocate(
        self,
        leading: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Allocate all state tensors for a given flattened batch shape."""
        B_flat = int(np.prod(leading)) if leading else 1

        # Time buffers: shape (B_flat, frame_size) per (level, fb).
        for iSG in range(self.n_spectrograms):
            for fb in range(self.n_filterbanks):
                self._set_time_buffer(
                    iSG, fb,
                    torch.zeros(B_flat, self.frame_size, device=device, dtype=dtype),
                )

        # spectrogram_buffer[i]: shape (B_flat, n_bands, n_cols_i), filled with 1e6.
        for i in range(self.n_spectrograms - 1):
            n_cols = self.spectrogram_buffer_cols[i]
            self._set_spectrogram_buffer(
                i,
                torch.full(
                    (B_flat, self.n_bands, n_cols), 1e6, device=device, dtype=dtype
                ),
            )

        # left_boundaries: shape (B_flat, n_bands, n_spectrograms-1), filled with 1e6.
        self.left_boundaries = torch.full(
            (B_flat, self.n_bands, self.n_spectrograms - 1),
            1e6,
            device=device,
            dtype=dtype,
        )

        self._allocated_batch_shape = tuple(leading)

    # ------------------------------------------------------------------
    # FFT (overlap-save, all sub-frames of one level batched)
    # ------------------------------------------------------------------

    def _fft_level(
        self,
        x_flat: torch.Tensor,  # (B, buffer_size)
        level: int,
        fb: int,
        detach_state: bool,
    ) -> torch.Tensor:
        """All sub-frames of one (level, fb) in a single batched rfft.

        Each sub-frame's overlap-save buffer is a sliding window of size
        ``frame_size``, stride ``chunk_size``, starting at offset ``chunk_size``
        into ``concat([prev_time_buffer, x_flat])`` — so the per-chunk
        sequential dependency collapses into one ``unfold`` view.

        Returns ``|X|²`` of shape ``(B, n_bands, n_subframes)``.
        """
        chunk_size = self.buffer_size >> level
        time_buf = self._get_time_buffer(level, fb)  # (B, frame_size)

        concat = torch.cat([time_buf, x_flat], dim=-1)  # (B, frame_size + buffer_size)
        windows = concat[:, chunk_size:].unfold(-1, self.frame_size, chunk_size)
        # (B, n_subframes, frame_size) — view, no copy

        fft_in = windows * self._get_window(level, fb)
        spectrum = torch.fft.rfft(fft_in, n=self.frame_size, dim=-1)
        power = spectrum.real ** 2 + spectrum.imag ** 2  # (B, n_subframes, n_bands)

        # contiguous() so the buffer doesn't keep `concat` (size frame_size + buffer_size) alive.
        new_state = concat[:, -self.frame_size:].contiguous()
        self._set_time_buffer(level, fb, new_state.detach() if detach_state else new_state)

        return power.transpose(-2, -1)  # (B, n_bands, n_subframes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        *,
        detach_state: bool = True,
    ) -> torch.Tensor:
        """Process one buffer of audio.

        Parameters
        ----------
        x:
            Input tensor of shape ``(..., buffer_size)``.
        detach_state:
            If True (default), detach all state tensors from the autograd graph
            after each forward, matching the C++ behaviour of non-differentiable
            state.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_bands, 2^(n_spectrograms-1))`` in dB.
        """
        if not x.is_floating_point():
            raise ValueError(f"input must be a floating dtype, got {x.dtype}")
        if x.shape[-1] != self.buffer_size:
            raise ValueError(
                f"Expected last dim {self.buffer_size}, got {x.shape[-1]}"
            )

        # Flatten leading dims.
        leading = x.shape[:-1]
        x_flat = x.reshape(-1, self.buffer_size)  # (B, buffer_size)
        B = x_flat.shape[0]

        # Lazy allocation / batch-shape check.
        if self._allocated_batch_shape is None:
            self._allocate(leading, x_flat.device, x_flat.dtype)
        elif self._allocated_batch_shape != leading:
            raise ValueError(
                f"SpectrogramAdaptiveMoving: batch shape mismatch. "
                f"Allocated for {self._allocated_batch_shape}, got {leading}. "
                "Call reset() first to rebind."
            )

        # ----------------------------------------------------------------
        # FFT cascade
        # ----------------------------------------------------------------
        # spectrograms[iSG]: (B, n_bands, 2^iSG) linear power spectrogram
        spectrograms: list[torch.Tensor] = []
        for iSG in range(self.n_spectrograms):
            if self.n_filterbanks == 1:
                power = self._fft_level(x_flat, iSG, 0, detach_state)
            else:
                p0 = self._fft_level(x_flat, iSG, 0, detach_state)
                p1 = self._fft_level(x_flat, iSG, 1, detach_state)
                p2 = self._fft_level(x_flat, iSG, 2, detach_state)
                power = torch.minimum(torch.minimum(p0, p1), p2)
            spectrograms.append(power)  # (B, n_bands, 2^iSG)

        # ----------------------------------------------------------------
        # Build output (mirrors processAlgorithm lines 57-81)
        # ----------------------------------------------------------------
        # Use a purely functional approach (no in-place writes on tensors that
        # participate in the autograd graph) so that gradcheck / backward pass works.
        # current_block tracks the "live" dB output for the current cascade level.
        # It grows from 1 column → 2 → 4 … as we process each level.

        # Level 0: single column, convert to dB.
        current_block = 10.0 * torch.log10(
            spectrograms[0].clamp(min=1e-20)
        )  # (B, n_bands, 1)

        # Collect new left boundaries; rebuild self.left_boundaries once at the end.
        # Reads of self.left_boundaries[..., iFB:iFB+1] inside the loop only ever
        # consume the previous-forward stash, never a freshly-written column.
        new_lbs: list[torch.Tensor] = []

        for iFB in range(self.n_spectrograms - 1):
            prev_cols = 1 << iFB          # 2^iFB
            new_cols = 1 << (iFB + 1)     # 2^(iFB+1)
            current_cols = self.spectrogram_buffer_cols[iFB]
            shift_cols = current_cols - new_cols

            # Build input to upscale: [leftBoundary | current_block[0..prev_cols-1]]
            # left_boundaries has shape (B, n_bands, n_spectrograms-1)
            left_boundary = self.left_boundaries[..., iFB : iFB + 1]  # (B, n_bands, 1)
            # current_block at this point has exactly prev_cols columns
            upscale_input = torch.cat([left_boundary, current_block], dim=-1)
            # shape: (B, n_bands, prev_cols + 1)

            # Save next left boundary (last column of current_block) before upscaling.
            # C++: leftBoundaries.col(iFB) = output.col(prevCols - 1)
            new_lbs.append(current_block[..., prev_cols - 1 : prev_cols])

            # 2x horizontal upscale with leftBoundaryExcluded=true.
            # upscale_input has prev_cols+1 columns (col 0 = boundary, cols 1..prev_cols = data).
            # Output has 2*prev_cols = new_cols columns.
            # Even output cols (0, 2, 4, ...) = midpoints of adjacent input cols.
            # Odd output cols (1, 3, 5, ...) = input data cols (1..prev_cols).
            even_cols = 0.5 * (upscale_input[..., :-1] + upscale_input[..., 1:])
            # shape: (B, n_bands, prev_cols)
            odd_cols = upscale_input[..., 1:]
            # shape: (B, n_bands, prev_cols)
            # Interleave: [even0, odd0, even1, odd1, ...] → shape (B, n_bands, new_cols)
            current_block = torch.stack([even_cols, odd_cols], dim=-1).reshape(
                B, self.n_bands, new_cols
            )

            # Per-level moving max-min on spectrograms[iFB+1] (linear power).
            # C++: movingMaxMin[iFB].process(spectrograms[iFB+1], spectrograms[iFB+1])
            moving_input = spectrograms[iFB + 1]  # (B, n_bands, new_cols)
            moving_output = self.moving_max_min[iFB](
                moving_input, detach_state=detach_state
            )  # (B, n_bands, new_cols)
            new_db = 10.0 * torch.log10(moving_output.clamp(min=1e-20))
            # shape: (B, n_bands, new_cols)

            # Shift-register update on spectrogram_buffer[iFB]:
            # leftCols(shiftCols) = rightCols(shiftCols), then rightCols(newCols) = new_db
            sb = self._get_spectrogram_buffer(iFB)  # (B, n_bands, current_cols)
            new_sb = torch.cat([sb[..., -shift_cols:], new_db], dim=-1)
            # shape: (B, n_bands, current_cols)  [shift_cols + new_cols = current_cols]
            if detach_state:
                new_sb = new_sb.detach()
            self._set_spectrogram_buffer(iFB, new_sb)

            # Combine: current_block = min(upscaled, buffer[0..new_cols-1])
            current_block = torch.minimum(current_block, new_sb[..., :new_cols])

        if new_lbs:
            new_lb_tensor = torch.cat(new_lbs, dim=-1)
            self.left_boundaries = (
                new_lb_tensor.detach() if detach_state else new_lb_tensor
            )

        # Reshape back to leading dims.
        return current_block.reshape(*leading, self.n_bands, 1 << (self.n_spectrograms - 1))
