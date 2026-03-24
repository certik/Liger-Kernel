"""
Torch-lite: a pure-Python, numpy-backed torch shim for macOS/CPU development.

Only implements enough of the PyTorch API surface to run Liger-Kernel tests
without requiring the real PyTorch C++ runtime.
"""

__version__ = "2.11.0"

import numpy as np

# ── Core types & dtypes ─────────────────────────────────────────────────────
from torch._tensor import (  # noqa: F401
    Tensor,
    dtype,
    float16,
    float32,
    float64,
    bfloat16,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    bool,
    _wrap,
    _to_np,
    _from_np,
)

# ── Submodules (must be importable as torch.nn, torch.cuda, etc.) ───────────
import torch.autograd  # noqa: F401
import torch.nn  # noqa: F401
import torch.amp  # noqa: F401
import torch.cuda  # noqa: F401
import torch.xpu  # noqa: F401
import torch.distributed  # noqa: F401
import torch._library  # noqa: F401

# Also expose version info as a submodule-like namespace
class version:
    __name__ = "torch.version"
    hip = None
    cuda = None
import sys
sys.modules["torch.version"] = version


# ── Convenience aliases ─────────────────────────────────────────────────────
FloatTensor = Tensor
LongTensor = Tensor
HalfTensor = Tensor
BoolTensor = Tensor


# ── no_grad context manager / decorator ─────────────────────────────────────
class _DecoratorContext:
    """Base for no_grad / enable_grad / inference_mode.

    Supports three calling conventions:
      @torch.no_grad        – class used directly as decorator (no parens)
      @torch.no_grad()      – instance used as decorator
      with torch.no_grad(): – context manager
    """

    def __init__(self, fn=None, **kwargs):
        self._fn = fn

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        if self._fn is not None:
            # Used as @no_grad (without parens) — self wraps fn
            return self._fn(*args, **kwargs)
        # Used as @no_grad() — being called as decorator factory
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        raise TypeError


class no_grad(_DecoratorContext):
    pass


class enable_grad(_DecoratorContext):
    pass


class inference_mode(_DecoratorContext):
    def __init__(self, mode=True, fn=None):
        super().__init__(fn=fn)


# ── Factory functions ───────────────────────────────────────────────────────

def tensor(data, dtype=None, device=None, requires_grad=False):
    logical = dtype if dtype is bfloat16 else None
    t = Tensor(data, dtype_=dtype, requires_grad=requires_grad, _logical_dtype=logical)
    return t


def as_tensor(data, dtype=None, device=None):
    return tensor(data, dtype=dtype, device=device)


def randn(*shape, dtype=None, device=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    nd = _to_np(dtype) or np.float32
    data = np.random.randn(*shape).astype(nd)
    logical = dtype if dtype is bfloat16 else None
    t = _wrap(data, logical)
    t._requires_grad = requires_grad
    return t


def rand(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    nd = _to_np(dtype) or np.float32
    return _wrap(np.random.rand(*shape).astype(nd))


def zeros(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    nd = _to_np(dtype) or np.float32
    return _wrap(np.zeros(shape, dtype=nd))


def ones(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    nd = _to_np(dtype) or np.float32
    return _wrap(np.ones(shape, dtype=nd))


def full(shape, fill_value, dtype=None, device=None):
    if isinstance(shape, int):
        shape = (shape,)
    nd = _to_np(dtype) or np.float32
    return _wrap(np.full(shape, fill_value, dtype=nd))


def empty(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    nd = _to_np(dtype) or np.float32
    return _wrap(np.empty(shape, dtype=nd))


def empty_like(input, dtype=None, device=None):
    if dtype is not None:
        nd = _to_np(dtype)
        return _wrap(np.empty_like(input._data, dtype=nd), dtype if dtype is bfloat16 else None)
    return _wrap(np.empty_like(input._data), input._logical_dtype)


def zeros_like(input, dtype=None, device=None):
    if dtype is not None:
        nd = _to_np(dtype)
        return _wrap(np.zeros_like(input._data, dtype=nd))
    return _wrap(np.zeros_like(input._data), input._logical_dtype)


def ones_like(input, dtype=None, device=None):
    if dtype is not None:
        nd = _to_np(dtype)
        return _wrap(np.ones_like(input._data, dtype=nd))
    return _wrap(np.ones_like(input._data), input._logical_dtype)


def randn_like(input, dtype=None, device=None):
    if dtype is not None:
        nd = _to_np(dtype)
        return _wrap(np.random.randn(*input.shape).astype(nd))
    return _wrap(np.random.randn(*input.shape).astype(input._data.dtype), input._logical_dtype)


def arange(start, end=None, step=1, dtype=None, device=None):
    if end is None:
        start, end = 0, start
    nd = _to_np(dtype) or np.int64
    return _wrap(np.arange(start, end, step, dtype=nd))


def linspace(start, end, steps, dtype=None, device=None):
    nd = _to_np(dtype) or np.float32
    return _wrap(np.linspace(start, end, steps, dtype=nd))


def stack(tensors, dim=0):
    arrays = [t._data for t in tensors]
    return _wrap(np.stack(arrays, axis=dim))


def cat(tensors, dim=0):
    arrays = [t._data for t in tensors]
    return _wrap(np.concatenate(arrays, axis=dim))


# ── Math ops ────────────────────────────────────────────────────────────────

def matmul(a, b):
    return a.__matmul__(b) if isinstance(a, Tensor) else _wrap(np.asarray(a) @ b._data)


def abs(input):
    return _wrap(np.abs(input._data), input._logical_dtype)


def exp(input):
    return _wrap(np.exp(input._data), input._logical_dtype)


def log(input):
    return _wrap(np.log(input._data), input._logical_dtype)


def sqrt(input):
    return _wrap(np.sqrt(input._data), input._logical_dtype)


def rsqrt(input):
    return _wrap(1.0 / np.sqrt(input._data.astype(np.float64)).astype(input._data.dtype), input._logical_dtype)


def tanh(input):
    if isinstance(input, Tensor):
        return _wrap(np.tanh(input._data), input._logical_dtype)
    return _wrap(np.tanh(np.asarray(input)))


def sigmoid(input):
    x = input._data.astype(np.float32)
    return _wrap((1.0 / (1.0 + np.exp(-x))).astype(input._data.dtype), input._logical_dtype)


def clamp(input, min=None, max=None):
    return _wrap(np.clip(input._data, min, max), input._logical_dtype)

clip = clamp


def sum(input, dim=None, keepdim=False):
    return _wrap(np.sum(input._data, axis=dim, keepdims=keepdim), input._logical_dtype)


def cumsum(input, dim):
    return _wrap(np.cumsum(input._data, axis=dim), input._logical_dtype)


def max(input, dim=None, keepdim=False):
    if isinstance(input, Tensor):
        return input.max(dim=dim, keepdim=keepdim)
    return _wrap(np.asarray(np.max(input)))


def min(input, dim=None, keepdim=False):
    if isinstance(input, Tensor):
        return input.min(dim=dim, keepdim=keepdim)
    return _wrap(np.asarray(np.min(input)))


def maximum(a, b):
    ad = a._data if isinstance(a, Tensor) else np.asarray(a)
    bd = b._data if isinstance(b, Tensor) else np.asarray(b)
    return _wrap(np.maximum(ad, bd))


def minimum(a, b):
    ad = a._data if isinstance(a, Tensor) else np.asarray(a)
    bd = b._data if isinstance(b, Tensor) else np.asarray(b)
    return _wrap(np.minimum(ad, bd))


def where(condition, x, y):
    cd = condition._data if isinstance(condition, Tensor) else condition
    xd = x._data if isinstance(x, Tensor) else x
    yd = y._data if isinstance(y, Tensor) else y
    return _wrap(np.where(cd, xd, yd))


def erf(input):
    from scipy.special import erf as _erf
    return _wrap(_erf(input._data), input._logical_dtype)


# ── Comparison / utility ────────────────────────────────────────────────────

def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    ad = a._data if isinstance(a, Tensor) else np.asarray(a)
    bd = b._data if isinstance(b, Tensor) else np.asarray(b)
    return builtins.bool(np.allclose(ad, bd, rtol=rtol, atol=atol, equal_nan=equal_nan))


def equal(a, b):
    return builtins.bool(np.array_equal(a._data, b._data))


import builtins  # noqa: E402


# ── RNG ─────────────────────────────────────────────────────────────────────

def manual_seed(seed):
    np.random.seed(seed)


def seed():
    np.random.seed()


# ── Device helpers ──────────────────────────────────────────────────────────

def device(d):
    return d

Size = tuple
