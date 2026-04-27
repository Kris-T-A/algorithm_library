import numpy as np
import torch

from torch_modules.perceptual_spectral_analysis.log_scale import LogScale


def _make_oracle(pal, *, n_inputs, n_outputs, output_start, output_end, input_end):
    cpp = pal.ScaleTransform()
    cpp.setCoefficients({
        "nInputs": n_inputs,
        "nOutputs": n_outputs,
        "outputStart": float(output_start),
        "outputEnd": float(output_end),
        "inputEnd": float(input_end),
        "transformType": "Logarithmic Scale",
    })
    return cpp


def test_full_alignment_matches_cpp(pal):
    """Spectrogram-shaped input through the full LogScale (all three regions)."""
    n_inputs = 8193  # 2 * bufferSize + 1 for bufferSize=4096
    n_outputs = 100
    sample_rate = 48000.0
    rng = np.random.default_rng(0)
    spectrogram_db = rng.uniform(-80, 0, size=(n_inputs, 4)).astype(np.float32)

    oracle = _make_oracle(
        pal,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        output_start=20.0,
        output_end=20000.0,
        input_end=sample_rate / 2,
    )
    cpp_out = oracle.process(spectrogram_db)

    module = LogScale(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        output_start=20.0,
        output_end=20000.0,
        input_end=sample_rate / 2,
    )
    torch_in = torch.from_numpy(spectrogram_db.T)  # (frames, n_inputs)
    torch_out = module(torch_in).T.numpy()  # back to (n_inputs, frames)

    n_lin = module.n_linear_bins
    n_cub = module.n_cubic_bins
    # Linear + cubic regions are tight.
    np.testing.assert_allclose(
        torch_out[:n_lin + n_cub, :],
        cpp_out[:n_lin + n_cub, :],
        atol=1e-5, rtol=1e-5,
    )
    # Triangular region uses C++ fasterlog2 vs torch exact log10 — looser tolerance.
    np.testing.assert_allclose(
        torch_out[n_lin + n_cub:, :],
        cpp_out[n_lin + n_cub:, :],
        atol=1e-1,  # ~0.15 dB drift expected from fasterlog2 approximation in C++
    )


def test_split_mostly_linear():
    """Narrow frequency range so almost all bins are linear (nCubic = 0, nTriangular = 1).

    The C++ while-loop bound is ``nLinearBins < nOutputs - 1`` so n_linear_bins is at
    most ``n_outputs - 1``; the remaining 1 bin falls into the triangular region.
    ``output_end = 100 Hz`` with ``n_outputs = 100`` keeps all bin spacings <= 1,
    giving the maximum possible linear split.
    """
    module = LogScale(n_inputs=8193, n_outputs=100, output_start=20.0, output_end=100.0, input_end=24000.0)
    assert module.n_linear_bins == 99
    assert module.n_cubic_bins == 0
    assert module.n_triangular_bins == 1
