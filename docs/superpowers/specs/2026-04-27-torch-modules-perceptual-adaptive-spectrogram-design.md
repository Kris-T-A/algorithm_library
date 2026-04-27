# Torch Modules — PerceptualAdaptiveSpectrogram Port

**Date:** 2026-04-27
**Status:** Approved (pending implementation plan)

## Goal

Stand up a new sibling library at `libs/torch_modules/` that hosts PyTorch reimplementations of algorithms from `src/`. The first port is `PerceptualAdaptiveSpectrogram` from `src/perceptual_spectral_analysis/perceptual_adaptive_spectrogram.h`. The library is designed so subsequent ports drop in naturally under the same structure.

## Use cases

The port must serve both:

1. **Training-time feature extractor** — used inside a PyTorch model graph; runs on GPU; gradients flow through it.
2. **Inference / preprocessing** — produces tensors for downstream models, with output that matches the C++ implementation closely enough to swap one for the other.

## Approach

Reimplement the algorithm in **pure Python using PyTorch ops** (the "torchaudio convention" — Python `nn.Module` classes calling `torch.fft`, `F.max_pool1d`, matmul, etc.). No custom C++ kernels in v1. If profiling later shows a kernel is needed, it can be added via `torch.utils.cpp_extension` without restructuring the package.

This approach was chosen over wrapping the existing C++/Eigen algorithm because (1) GPU execution and autograd are required for training, (2) most of the algorithm decomposes into ops PyTorch already provides efficiently, and (3) the only non-differentiable steps (`ceil`, bin index selection in `LogScale`) are precomputed at construction time and don't enter the forward graph.

## Two public modules

Both subclass `torch.nn.Module`. Both follow the leading-dim convention from `torch.stft`: any number of leading dims are preserved, the last dim is time.

| Module | State | Input shape | Output shape | Use case |
|---|---|---|---|---|
| `PerceptualAdaptiveSpectrogram` | stateless | `(..., T)` | `(..., nBands, T // bufferSize * 2^(nSpectrograms-1))` | training, full-clip processing |
| `PerceptualAdaptiveSpectrogramStreaming` | stateful | `(..., bufferSize)` | `(..., nBands, 2^(nSpectrograms-1))` | inference parity with C++ |

The stateless module is a thin loop around the streaming module: it chunks the input into `bufferSize`-sized blocks, calls `reset()` on the streaming module, runs the streaming module per chunk, and concatenates outputs along the frame axis. This guarantees the two modules produce equivalent output by construction (a property the test suite verifies).

The streaming module holds state (`spectrogramBuffer`, `leftBoundaries`) as `register_buffer`s, exposes a `reset()` method that restores the C++ sentinel value (`1e6f`), and detaches state between calls so autograd doesn't stretch across the full stream.

## Project layout

```
libs/torch_modules/
├── pyproject.toml                       # PEP 621; deps: torch, numpy, pytest, pytest-benchmark
├── README.md
├── torch_modules/
│   ├── __init__.py
│   ├── interpolation/
│   │   ├── __init__.py
│   │   └── interpolation_cubic.py
│   └── perceptual_spectral_analysis/
│       ├── __init__.py
│       ├── perceptual_adaptive_spectrogram.py   # both public modules
│       ├── spectrogram_adaptive_moving.py
│       ├── log_scale.py
│       └── moving_max_min.py
└── tests/
    ├── conftest.py                                              # imports python_algorithm_library oracle
    ├── test_interpolation_cubic_alignment.py
    ├── test_log_scale_alignment.py
    ├── test_moving_max_min_alignment.py
    ├── test_spectrogram_adaptive_moving_alignment.py
    ├── test_perceptual_adaptive_spectrogram_alignment.py
    ├── test_perceptual_adaptive_spectrogram_edge_cases.py
    └── benchmark_perceptual_adaptive_spectrogram.py
```

The directory hierarchy mirrors `src/`. Future algorithm ports follow the same pattern: `torch_modules/<algorithm_family>/<algorithm>.py`. Shared building blocks (like `InterpolationCubic`) are top-level under `torch_modules/`.

No CMake in v1. The project is consumed via `pip install -e libs/torch_modules` (or by adding the path to `sys.path`). When/if a C++ kernel is added later, it bolts on with `torch.utils.cpp_extension` and a small `setup.py` extension build — no restructuring needed.

## Components

The internal sub-modules mirror the C++ class boundaries. Each is independently testable.

### `MovingMaxMin`
Covers both `MovingMaxMinVertical` and `MovingMaxMinHorizontal` via an `axis` constructor argument. Forward implementation: `F.max_pool1d` for max; `-F.max_pool1d(-x)` for min. Both use `kernel_size=L, stride=1, padding=(L-1)//2`. Edge-cell semantics need to match C++ (van Herk style); this is verified by the alignment test.

### `LogScale`
Constructor precomputes the same lookup tables as the C++ version (`outputStart`, `fractionLinear`, `fractionCubic`, `fractionTriangular`, `distanceTriangular`) and additionally precomputes dense weight matrices for the linear and triangular sections so the forward pass is matmul + max-reduce instead of Python loops. All precomputed tensors are `register_buffer`s.

Forward decomposes into three concatenated chunks:
- **Linear bins:** weighted sum, expressed as a single matmul against an `(nLinearBins, nInputs)` weight matrix.
- **Cubic bins:** delegate to `InterpolationCubic`, vectorized across the batch and frame dims.
- **Triangular bins:** the C++ does `max(input(iBin) + weight_dB)` over a triangular window in dB space. Vectorized as `(input.unsqueeze(...) + weights).max(dim=...)` with masking for variable-width windows.

### `InterpolationCubic`
Top-level shared module under `torch_modules/interpolation/`. Standard cubic interpolation with a vector of fractional indices. Vectorized across leading dims.

### `SpectrogramAdaptiveMoving`
Composes the FFT cascade (`SpectrogramSetZeropad`-equivalent), 2x horizontal upscale with left boundary (`Upscale2DLinear`-equivalent), and per-level `MovingMaxMinHorizontal`. The FFT cascade uses `torch.fft.rfft` directly. Holds state as `register_buffer`s in the streaming module; held as plain locals in the stateless module's loop.

The forward is a direct port of the C++ `processAlgorithm` loop, vectorized where the C++ couldn't be (across batch).

### `PerceptualAdaptiveSpectrogramStreaming`
Composes `SpectrogramAdaptiveMoving`, `LogScale`, `MovingMaxMinVertical`. Holds the `spectralTiltVector` constant as a `register_buffer`. Forward mirrors the C++ `processAlgorithm`.

### `PerceptualAdaptiveSpectrogram`
Wraps a `PerceptualAdaptiveSpectrogramStreaming`. Forward chunks input along time into `T // bufferSize` blocks, resets streaming state, runs the streaming module per chunk, concatenates outputs along the frame axis.

## Data flow

```
x: (..., bufferSize)  float32
  → flatten leading dims to (B*, bufferSize)
  → SpectrogramAdaptiveMoving         → (B*, 2*bufferSize+1, frames)  dB
  → optional spectralTilt (add buffer) → same shape
  → LogScale                           → (B*, nBands, frames)         dB
  → MovingMaxMinVertical               → (B*, nBands, frames)         dB
  → unflatten leading dims             → (..., nBands, frames)
y
```

For the stateless module, this runs once per chunk in a Python loop, with outputs concatenated along the frame axis.

## Device, dtype, gradients

- **Device:** all non-parameter constants are `register_buffer`s, so `.to('cuda')` works out of the box. No CUDA-specific code paths in v1.
- **Dtype:** float32 only. Asserted in `forward`. Users `.to(torch.float32)` upstream if needed.
- **Gradients:** `requires_grad=True` input gets gradients. Buffers (lookup tables, sentinel state) are fixed and don't participate. State buffers in the streaming module are detached between calls.
- **Multi-channel:** handled implicitly by the leading-dim flattening — `(B, C, T)` is processed as `(B*C, T)` then reshaped back. The C++ algorithm is mono-only, so the alignment test runs with no extra leading dims.

## Error handling

- `forward` validates: input dtype is `float32`; last dim equals `bufferSize` (streaming) or is a positive multiple of `bufferSize` (stateless). Mismatch raises `ValueError` with the actual shape.
- Constructor validates coefficients: `nBands > 0`, `0 < frequencyMin < frequencyMax <= sampleRate / 2`, `nSpectrograms >= 1`, `nFolds >= 1`, `bufferSize > 0`. Same constraints the C++ implies. Invalid values raise `ValueError`.
- No silent fallbacks, no clamping of weird inputs, no try/except wrappers.

## Testing

### Reference oracle
The existing `python_algorithm_library` already exposes `PerceptualAdaptiveSpectrogram` to Python via pybind. Tests import it directly. `conftest.py` handles the `sys.path` insertion for the .pyd at `libs/python_algorithm_library/python_script/`.

For per-component tests, each torch sub-module is compared against its corresponding public C++ algorithm (which the C++ side dispatches to the same internal implementation):

| Torch sub-module | C++ public interface | Already in pybind? |
|---|---|---|
| `InterpolationCubic` | `Interpolation` (cubic mode) | yes |
| `MovingMaxMin` | `FilterMinMax` | yes |
| `LogScale` | `ScaleTransform` (LOGARITHMIC mode) | **no — needs adding** |
| `SpectrogramAdaptiveMoving` | `SpectrogramAdaptive` | yes |

Adding the `ScaleTransform` binding to `libs/python_algorithm_library/src/main.cpp` is a one-line include plus the standard binding template — counted as part of this work.

### Per-component alignment tests

| Test | Inputs | Tolerance |
|---|---|---|
| `InterpolationCubic` | random input + fractional indices | `atol=1e-5, rtol=1e-5` |
| `MovingMaxMin` (both axes) | random 2D, sweep `filterLength ∈ {1, 2, 4, 8}` | `atol=1e-5, rtol=1e-5` |
| `LogScale` (linear + cubic regions) | random spectrogram | `atol=1e-5, rtol=1e-5` |
| `LogScale` (triangular region) | random spectrogram | `atol=1e-3` (dB) |
| `SpectrogramAdaptiveMoving` | N consecutive `bufferSize` chunks; per-chunk parity | `atol=1e-2` (dB) |
| `PerceptualAdaptiveSpectrogramStreaming` end-to-end | N consecutive chunks | `atol=1e-2` (dB) |
| `PerceptualAdaptiveSpectrogram` (stateless) | full clip; equals `concat(streaming(chunk_i))` | `atol=1e-6` (no FFT diff — same code) |

Tolerances are starting estimates. The largest source of drift between PyTorch and Eigen+pffft is the FFT backend (~0.001 dB per FFT, accumulating through the cascade). If parity is tighter in practice, tighten the thresholds.

### Edge-case tests (no oracle needed)
- Silence (zeros) → finite output, no NaN/Inf.
- Constant DC → finite output.
- Multi-channel `(B, C, T)`: verify `(..., T)` flattening + reshape round-trips correctly — output of `module(x_batch)[i, c]` matches `module(x_batch[i, c])` for every `(i, c)`.
- `reset()` actually clears state — same input run twice with `reset()` between produces identical output.
- `gradcheck` on a tiny config to confirm autograd end-to-end.

### Benchmarks
`tests/benchmark_perceptual_adaptive_spectrogram.py`, run via `pytest-benchmark` or as a standalone script. Configs:
- Default coeffs; `nBands ∈ {100, 500}`; batch ∈ `{1, 32, 256}`.
- CPU and CUDA (skip CUDA if not available).
- Both torch modules, plus a row for the C++ oracle (single-batch, CPU-only, for reference).

Output: a small table printed to stdout. No regression-tracking infrastructure in v1.

## Out of scope (v1)

- Custom C++ / CUDA kernels.
- CMake / installable C++ artifact.
- TorchScript serialization (the modules will likely be scriptable but it's not a v1 acceptance criterion).
- Other algorithms from `src/`. The structure supports them; this spec covers only `PerceptualAdaptiveSpectrogram`.
- Float64 / float16 / bfloat16 support.
- Backward-pass numerical validation against a hand-derived gradient. `gradcheck` is the only autograd test.
