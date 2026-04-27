import numpy as np
import torch

from torch_modules.interpolation.interpolation_cubic import InterpolationCubic


def test_alignment_against_cpp_oracle(pal):
    n_inputs = 64
    n_outputs = 32
    rng = np.random.default_rng(0)
    src = rng.standard_normal(n_inputs).astype(np.float32)
    # Indices must be in [1, n_inputs - 2].
    fractional_indices = rng.uniform(1.0, n_inputs - 2.0, size=n_outputs).astype(np.float32)

    cpp = pal.Interpolation()
    cpp_out = cpp.process(src, fractional_indices)  # 1D out

    module = InterpolationCubic()
    torch_out = module(torch.from_numpy(src), torch.from_numpy(fractional_indices)).numpy()

    np.testing.assert_allclose(torch_out, cpp_out, atol=1e-5, rtol=1e-5)


def test_vectorizes_across_leading_dims():
    """Same fractional indices across a (B, T, n_inputs) input must equal per-row computation."""
    rng = np.random.default_rng(1)
    src = torch.from_numpy(rng.standard_normal((2, 3, 64)).astype(np.float32))
    indices = torch.linspace(1.0, 61.0, 32)  # in [1, n-3]; floor(idx)+2 <= 63 == n-1

    module = InterpolationCubic()
    out_batched = module(src, indices)
    assert out_batched.shape == (2, 3, 32)

    for i in range(2):
        for j in range(3):
            torch.testing.assert_close(out_batched[i, j], module(src[i, j], indices))
