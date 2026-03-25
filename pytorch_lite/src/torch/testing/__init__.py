"""Minimal torch.testing shim."""

import numpy as np


def assert_close(actual, expected, rtol=None, atol=None, **kwargs):
    """Compare two tensors element-wise, like torch.testing.assert_close."""
    from torch._tensor import Tensor

    a = actual._data if isinstance(actual, Tensor) else np.asarray(actual)
    b = expected._data if isinstance(expected, Tensor) else np.asarray(expected)

    # Handle shape mismatch
    if a.shape != b.shape:
        raise AssertionError(
            f"Tensor-likes are not close!\n"
            f"Shape mismatch: {a.shape} vs {b.shape}"
        )

    # Promote both to the higher-precision type for comparison
    common = np.result_type(a.dtype, b.dtype)
    a = a.astype(common)
    b = b.astype(common)

    if rtol is None or atol is None:
        # Default tolerances matching PyTorch's assert_close for each dtype
        if common == np.float16:
            rtol = rtol or 1e-3
            atol = atol or 1e-5
        elif common == np.float64:
            rtol = rtol or 1.3e-6
            atol = atol or 1e-5
        else:  # float32 and others
            rtol = rtol or 1.3e-6
            atol = atol or 1e-5

    try:
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"Tensor-likes are not close!\n{e}") from None
