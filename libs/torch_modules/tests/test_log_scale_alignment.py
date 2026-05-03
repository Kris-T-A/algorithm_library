import numpy as np
import torch

from torch_modules.perceptual_spectral_analysis.log_scale import LogScale


def _cpp_round_positive(x: float) -> int:
    return int(np.floor(x + 0.5))


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
    # Triangular region: both sides now use exact log10, tight tolerance.
    np.testing.assert_allclose(
        torch_out[n_lin + n_cub:, :],
        cpp_out[n_lin + n_cub:, :],
        atol=1e-5, rtol=1e-5,  # was atol=1e-1 to absorb fasterlog2 drift; both sides now use exact log10
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


def test_triangular_region_uses_cpp_window_bounds():
    """Regression for the C++ fix: only bins in each triangular window participate."""
    n_inputs = 64
    n_outputs = 12
    output_start = 1.0
    output_end = 60.0
    input_end = 64.0
    module = LogScale(n_inputs, n_outputs, output_start, output_end, input_end)

    rng = np.random.default_rng(1)
    x_np = rng.uniform(-80.0, 0.0, size=n_inputs).astype(np.float32)
    torch_out = module(torch.from_numpy(x_np[None, :]))[0].numpy()

    scale = 1.0
    min_log = np.log10(1.0 + scale * output_start)
    max_log = np.log10(1.0 + scale * output_end)
    lin_logs = np.linspace(min_log, max_log, n_outputs, dtype=np.float64)
    freq_per_bin = scale * input_end / (n_inputs - 1)
    center_bins = ((np.power(10.0, lin_logs) - 1.0) / freq_per_bin).astype(np.float32)

    n_sum = module.n_linear_bins + module.n_cubic_bins
    expected = []
    for i in range(module.n_triangular_bins):
        c_start = center_bins[n_sum + i - 1]
        c_mid = center_bins[n_sum + i]
        c_end = center_bins[n_sum + i + 1] if i < module.n_triangular_bins - 1 else _cpp_round_positive(float(c_mid)) + 1

        i_start = int(np.ceil(c_start))
        i_mid = _cpp_round_positive(float(c_mid))
        i_end = int(np.ceil(c_end))

        values = []
        for i_bin in range(i_start, i_mid):
            lin_weight = 1.0 - (c_mid - i_bin) / (c_mid - c_start)
            values.append(x_np[i_bin] + 10.0 * np.log10(lin_weight + 1e-16))
        values.append(x_np[i_mid])
        for i_bin in range(i_mid + 1, i_end):
            lin_weight = 1.0 - (i_bin - c_mid) / (c_end - c_mid)
            values.append(x_np[i_bin] + 10.0 * np.log10(lin_weight + 1e-16))
        expected.append(max(values))

    actual = torch_out[n_sum:]
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float32), atol=1e-5, rtol=1e-5)


def test_triangular_region_ignores_out_of_window_values():
    """C++ reduces only each triangular window, so outside values must not participate."""
    module = LogScale(n_inputs=64, n_outputs=12, output_start=1.0, output_end=60.0, input_end=64.0)

    x = torch.full((1, module.n_inputs), -80.0)
    x[0, -1] = float("inf")

    out = module(x)

    assert torch.isfinite(out[..., module.n_linear_bins + module.n_cubic_bins:]).all()
