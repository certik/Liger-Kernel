"""
Triton-lite: a pure-Python interpreter shim for Triton kernels.

Only implements enough to run kernels as regular Python (no GPU required).
The @triton.jit decorator wraps kernel functions so they can be launched
with the standard kernel[(grid,)](...) syntax, executing them sequentially
on CPU using torch tensors.
"""

__version__ = "3.6.0"

import math

import torch

from triton._interpreter import JITFunction

# Make submodules accessible as attributes (e.g., triton.language, triton.runtime)
import triton.language  # noqa: F401
import triton.runtime  # noqa: F401

# Triton kernels use .cast(dtype) on tensors — add it to torch.Tensor if missing
if not hasattr(torch.Tensor, "cast"):
    torch.Tensor.cast = lambda self, dtype: self.to(dtype)


def jit(fn=None, **kwargs):
    """Decorator that wraps a Triton kernel for CPU interpretation."""
    if fn is not None:
        return JITFunction(fn)
    # Support @triton.jit(noinline=True, ...) style
    def wrapper(f):
        return JITFunction(f)
    return wrapper


def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (int(n) - 1).bit_length()


def cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


class Config:
    """Stub for triton.Config used in @triton.autotune."""
    def __init__(self, kwargs=None, num_warps=4, num_stages=2, **extra):
        self.kwargs = kwargs or {}
        self.num_warps = num_warps
        self.num_stages = num_stages


def autotune(configs, key, **kwargs):
    """Stub for @triton.autotune — just uses the first config."""
    def decorator(fn):
        return jit(fn)
    return decorator
