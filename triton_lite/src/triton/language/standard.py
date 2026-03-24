"""triton.language.standard — Standard library functions."""

import torch
import math


def _log2(x):
    """Compute log base 2 — returns int for exact powers of 2 (constexpr context)."""
    if isinstance(x, torch.Tensor):
        return torch.log2(x)
    result = math.log2(x)
    # In Triton, _log2 is used for constexpr sizes (always powers of 2)
    if result == int(result):
        return int(result)
    return result


def zeros_like(x):
    """Create a tensor of zeros with the same shape and dtype as x."""
    if isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    return 0
