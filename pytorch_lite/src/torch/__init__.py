"""
Torch-lite: a pure-Python, numpy-backed torch shim for macOS/CPU development.

Only implements enough of the PyTorch API surface to run Liger-Kernel tests
without requiring the real PyTorch C++ runtime.
"""

__version__ = "2.11.0"

import numpy as np
import math as _math

def _np_erf(x):
    """Vectorized erf using Python math.erf."""
    vfunc = np.vectorize(_math.erf, otypes=[np.float64])
    return vfunc(x.astype(np.float64)).astype(x.dtype)

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
    complex32,
    complex64,
    complex128,
    _wrap,
    _to_np,
    _from_np,
    _MaxMinResult,
)

# ── Submodules (must be importable as torch.nn, torch.cuda, etc.) ───────────
import torch.autograd  # noqa: F401
import torch.nn  # noqa: F401
import torch.amp  # noqa: F401
import torch.cuda  # noqa: F401
import torch.xpu  # noqa: F401
import torch.distributed  # noqa: F401
import torch._library  # noqa: F401
import torch.testing  # noqa: F401

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


def empty(*shape, dtype=None, device=None, pin_memory=False, layout=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    nd = _to_np(dtype) or np.float32
    return _wrap(np.empty(shape, dtype=nd), dtype if dtype is bfloat16 else None)


def empty_like(input, dtype=None, device=None, memory_format=None, layout=None, requires_grad=False):
    nd = _to_np(dtype) if dtype is not None else input._data.dtype
    logical = dtype if dtype is bfloat16 else (input._logical_dtype if dtype is None else None)
    # When memory_format is contiguous or input is non-contiguous, force C-contiguous
    if memory_format is not None or not input._data.flags['C_CONTIGUOUS']:
        return _wrap(np.empty(input._data.shape, dtype=nd), logical)
    return _wrap(np.empty_like(input._data, dtype=nd), logical)


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
    # Filter out empty 1D tensors when concatenating with higher-dim tensors
    max_ndim = builtins.max((a.ndim for a in arrays), default=1)
    if max_ndim > 1:
        filtered = []
        for a in arrays:
            if a.ndim == 1 and a.size == 0:
                continue  # Skip empty 1D tensors
            filtered.append(a)
        if not filtered:
            # All empty - create appropriate empty result
            return _wrap(np.empty((0,)))
        arrays = filtered
    if not arrays:
        return _wrap(np.empty((0,)))
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


def sum(input, dim=None, keepdim=False, *, dtype=None):
    # Empty list/tuple → full reduction (PyTorch semantics)
    if isinstance(dim, (list, tuple)) and len(dim) == 0:
        dim = None
    data = input._data
    out_dtype = _to_np(dtype) if dtype else input._data.dtype
    is_fp16 = data.dtype == np.float16

    # For multi-dim reduction on fp16, reduce iteratively with fp16 rounding
    # between each dimension to match triton kernel behavior
    if isinstance(dim, (list, tuple)) and len(dim) > 1 and is_fp16:
        result = data
        for d in sorted(dim, reverse=True):
            result = np.sum(result.astype(np.float32), axis=d, keepdims=keepdim).astype(np.float16)
        return _wrap(np.asarray(result).astype(out_dtype), dtype if dtype is bfloat16 else input._logical_dtype)

    # Convert list/tuple of dims to tuple for numpy
    if isinstance(dim, (list, tuple)):
        dim = tuple(dim)
    # Promote fp16 to fp32 for accumulation (matches PyTorch/GPU behavior)
    if is_fp16:
        data = data.astype(np.float32)
    result = np.sum(data, axis=dim, keepdims=keepdim)
    return _wrap(np.asarray(result).astype(out_dtype), dtype if dtype is bfloat16 else input._logical_dtype)


def cumsum(input, dim, *, dtype=None):
    data = input._data
    # When dtype is specified, compute in that dtype
    if dtype is not None:
        out_np = _to_np(dtype)
        result = np.cumsum(data.astype(out_np), axis=dim)
        logical = dtype if dtype is bfloat16 else None
        return _wrap(result, logical)
    # Promote fp16 to fp32 for accumulation, then cast back
    if data.dtype == np.float16:
        result = np.cumsum(data.astype(np.float32), axis=dim).astype(np.float16)
    else:
        result = np.cumsum(data, axis=dim)
    return _wrap(result, input._logical_dtype)


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


def addcdiv(input, tensor1, tensor2, *, value=1):
    inp = input._data.astype(np.float64)
    t1 = tensor1._data.astype(np.float64)
    t2 = tensor2._data.astype(np.float64)
    result = inp + value * (t1 / t2)
    return _wrap(result.astype(input._data.dtype), input._logical_dtype)


def erf(input):
    return _wrap(_np_erf(input._data), input._logical_dtype)


# ── Comparison / utility ────────────────────────────────────────────────────

def softmax(input, dim=-1):
    import torch.nn.functional as F
    return F.softmax(input, dim=dim)


def softplus(input, beta=1, threshold=20):
    x = input._data.astype(np.float32)
    bx = beta * x
    result = np.where(bx > threshold, x, np.log1p(np.exp(bx)) / beta)
    return _wrap(result.astype(input._data.dtype), input._logical_dtype)


def isnan(input):
    return _wrap(np.isnan(input._data))


def isinf(input):
    return _wrap(np.isinf(input._data))


def isposinf(input):
    return _wrap(np.isposinf(input._data))


def isneginf(input):
    return _wrap(np.isneginf(input._data))


def logical_xor(a, b):
    ad = a._data if isinstance(a, Tensor) else np.asarray(a)
    bd = b._data if isinstance(b, Tensor) else np.asarray(b)
    return _wrap(np.logical_xor(ad, bd))


def logical_or(a, b):
    ad = a._data if isinstance(a, Tensor) else np.asarray(a)
    bd = b._data if isinstance(b, Tensor) else np.asarray(b)
    return _wrap(np.logical_or(ad, bd))


def logical_and(a, b):
    ad = a._data if isinstance(a, Tensor) else np.asarray(a)
    bd = b._data if isinstance(b, Tensor) else np.asarray(b)
    return _wrap(np.logical_and(ad, bd))


def logical_not(a):
    ad = a._data if isinstance(a, Tensor) else np.asarray(a)
    return _wrap(np.logical_not(ad))

def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    ad = a._data if isinstance(a, Tensor) else np.asarray(a)
    bd = b._data if isinstance(b, Tensor) else np.asarray(b)
    return builtins.bool(np.allclose(ad, bd, rtol=rtol, atol=atol, equal_nan=equal_nan))


def nonzero(input):
    indices = np.argwhere(input._data)
    return _wrap(indices.astype(np.int64))


def equal(a, b):
    return builtins.bool(np.array_equal(a._data, b._data))


import builtins  # noqa: E402


# ── Additional math / element-wise ops ──────────────────────────────────────

def sub(input, other, *, alpha=1, out=None):
    id_ = input._data if isinstance(input, Tensor) else np.asarray(input, dtype=np.float32)
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    # Determine output dtype from the tensor operand
    has_fp16 = (isinstance(id_, np.ndarray) and id_.dtype == np.float16) or \
               (isinstance(od, np.ndarray) and od.dtype == np.float16)
    if has_fp16:
        id_ = id_.astype(np.float32) if isinstance(id_, np.ndarray) else id_
        od = od.astype(np.float32) if isinstance(od, np.ndarray) else od
    result = id_ - alpha * od
    if has_fp16 and isinstance(result, np.ndarray):
        result = result.astype(np.float16)
    logical = input._logical_dtype if isinstance(input, Tensor) else (
        other._logical_dtype if isinstance(other, Tensor) else None)
    r = _wrap(result, logical)
    if out is not None:
        out.copy_(r)
        return out
    return r


def add(input, other, *, alpha=1, out=None):
    id_ = input._data if isinstance(input, Tensor) else np.asarray(input, dtype=np.float32)
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    has_fp16 = (isinstance(id_, np.ndarray) and id_.dtype == np.float16) or \
               (isinstance(od, np.ndarray) and od.dtype == np.float16)
    if has_fp16:
        id_ = id_.astype(np.float32) if isinstance(id_, np.ndarray) else id_
        od = od.astype(np.float32) if isinstance(od, np.ndarray) else od
    result = id_ + alpha * od
    if has_fp16 and isinstance(result, np.ndarray):
        result = result.astype(np.float16)
    logical = input._logical_dtype if isinstance(input, Tensor) else (
        other._logical_dtype if isinstance(other, Tensor) else None)
    r = _wrap(result, logical)
    if out is not None:
        out.copy_(r)
        return out
    return r


def mul(input, other, *, out=None):
    id = input._data if isinstance(input, Tensor) else np.asarray(input)
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    r = _wrap(id * od, input._logical_dtype if isinstance(input, Tensor) else None)
    if out is not None:
        out.copy_(r)
        return out
    return r


def div(input, other, *, rounding_mode=None, out=None):
    id = input._data if isinstance(input, Tensor) else np.asarray(input)
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    result = id / od
    if rounding_mode == 'trunc':
        result = np.trunc(result)
    elif rounding_mode == 'floor':
        result = np.floor(result)
    if isinstance(result, np.ndarray) and isinstance(id, np.ndarray):
        result = result.astype(id.dtype)
    r = _wrap(result, input._logical_dtype if isinstance(input, Tensor) else None)
    if out is not None:
        out.copy_(r)
        return out
    return r


def addcmul(input, tensor1, tensor2, *, value=1):
    inp = input._data.astype(np.float64)
    t1 = tensor1._data.astype(np.float64)
    t2 = tensor2._data.astype(np.float64)
    result = inp + value * (t1 * t2)
    return _wrap(result.astype(input._data.dtype), input._logical_dtype)


def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    inp = input._data.astype(np.float64)
    m1 = mat1._data.astype(np.float64)
    m2 = mat2._data.astype(np.float64)
    result = beta * inp + alpha * (m1 @ m2)
    return _wrap(result.astype(mat1._data.dtype), mat1._logical_dtype)


def addmv(input, mat, vec, *, beta=1, alpha=1):
    inp = input._data.astype(np.float64)
    m = mat._data.astype(np.float64)
    v = vec._data.astype(np.float64)
    result = beta * inp + alpha * (m @ v)
    return _wrap(result.astype(mat._data.dtype), mat._logical_dtype)


def addr(input, vec1, vec2, *, beta=1, alpha=1):
    inp = input._data.astype(np.float64)
    v1 = vec1._data.astype(np.float64)
    v2 = vec2._data.astype(np.float64)
    result = beta * inp + alpha * np.outer(v1, v2)
    return _wrap(result.astype(input._data.dtype), input._logical_dtype)


def all(input, dim=None, keepdim=False):
    if isinstance(input, Tensor):
        return _wrap(np.all(input._data, axis=dim, keepdims=keepdim))
    return _wrap(np.asarray(np.all(input)))


def any(input, dim=None, keepdim=False):
    if isinstance(input, Tensor):
        return _wrap(np.any(input._data, axis=dim, keepdims=keepdim))
    return _wrap(np.asarray(np.any(input)))


def amax(input, dim=None, keepdim=False):
    return _wrap(np.max(input._data, axis=dim, keepdims=keepdim), input._logical_dtype)


def angle(input):
    return _wrap(np.angle(input._data).astype(np.float32))


def argmax(input, dim=None, keepdim=False):
    if dim is None:
        idx = int(np.argmax(input._data))
        if keepdim:
            out_shape = [1] * input._data.ndim
            return _wrap(np.full(out_shape, idx, dtype=np.int64))
        return _wrap(np.asarray(idx, dtype=np.int64))
    result = np.argmax(input._data, axis=dim).astype(np.int64)
    if keepdim:
        result = np.expand_dims(result, axis=dim)
    return _wrap(result)


def argmin(input, dim=None, keepdim=False):
    if dim is None:
        idx = int(np.argmin(input._data))
        if keepdim:
            out_shape = [1] * input._data.ndim
            return _wrap(np.full(out_shape, idx, dtype=np.int64))
        return _wrap(np.asarray(idx, dtype=np.int64))
    result = np.argmin(input._data, axis=dim).astype(np.int64)
    if keepdim:
        result = np.expand_dims(result, axis=dim)
    return _wrap(result)


def atan(input):
    return _wrap(np.arctan(input._data), input._logical_dtype)


def atan2(input, other):
    id = input._data if isinstance(input, Tensor) else np.asarray(input)
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(np.arctan2(id, od), input._logical_dtype if isinstance(input, Tensor) else None)


def bitwise_and(input, other):
    id = input._data if isinstance(input, Tensor) else np.asarray(input)
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(id & od)


def bitwise_not(input):
    return _wrap(~input._data)


def bitwise_or(input, other):
    id = input._data if isinstance(input, Tensor) else np.asarray(input)
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(id | od)


def bmm(input, mat2):
    a, b = input._data, mat2._data
    if a.dtype == np.float16:
        return _wrap((a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16), input._logical_dtype)
    if a.dtype == np.float32:
        return _wrap((a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32), input._logical_dtype)
    return _wrap(np.matmul(a, b), input._logical_dtype)


def broadcast_shapes(*shapes):
    return np.broadcast_shapes(*shapes)


def broadcast_to(input, shape):
    return _wrap(np.broadcast_to(input._data, shape), input._logical_dtype)


def chunk(input, chunks, dim=0):
    arrays = np.array_split(input._data, chunks, axis=dim)
    return [_wrap(a, input._logical_dtype) for a in arrays]


def clone(input, *, memory_format=None):
    return _wrap(input._data.copy(), input._logical_dtype)


def complex(real, imag):
    rd = real._data if isinstance(real, Tensor) else np.asarray(real)
    id = imag._data if isinstance(imag, Tensor) else np.asarray(imag)
    return _wrap((rd + 1j * id).astype(np.complex64))


def cos(input):
    return _wrap(np.cos(input._data), input._logical_dtype)


def ceil(input):
    return _wrap(np.ceil(input._data), input._logical_dtype)


def floor(input):
    return _wrap(np.floor(input._data), input._logical_dtype)


def log2(input):
    return _wrap(np.log2(input._data), input._logical_dtype)


def count_nonzero(input, dim=None):
    return _wrap(np.count_nonzero(input._data, axis=dim).astype(np.int64))


def cummax(input, dim):
    data = input._data
    values = np.maximum.accumulate(data, axis=dim)
    # compute indices: for each position, index of the max so far
    shape = data.shape
    ndim = data.ndim
    indices = np.zeros_like(data, dtype=np.int64)
    slices_before = [slice(None)] * ndim
    for i in range(shape[dim]):
        slices_before[dim] = i
        current_val = data[tuple(slices_before)]
        if i == 0:
            indices[tuple(slices_before)] = 0
        else:
            prev_slice = list(slices_before)
            prev_slice[dim] = i - 1
            prev_max_idx = indices[tuple(prev_slice)]
            # Get previous max value
            prev_max_val = values[tuple(prev_slice)]
            indices[tuple(slices_before)] = np.where(
                current_val > prev_max_val, i, prev_max_idx
            )
    return _MaxMinResult(_wrap(values, input._logical_dtype), _wrap(indices))


def cummin(input, dim):
    data = input._data
    values = np.minimum.accumulate(data, axis=dim)
    shape = data.shape
    ndim = data.ndim
    indices = np.zeros_like(data, dtype=np.int64)
    slices_before = [slice(None)] * ndim
    for i in range(shape[dim]):
        slices_before[dim] = i
        current_val = data[tuple(slices_before)]
        if i == 0:
            indices[tuple(slices_before)] = 0
        else:
            prev_slice = list(slices_before)
            prev_slice[dim] = i - 1
            prev_min_idx = indices[tuple(prev_slice)]
            prev_min_val = values[tuple(prev_slice)]
            indices[tuple(slices_before)] = np.where(
                current_val < prev_min_val, i, prev_min_idx
            )
    return _MaxMinResult(_wrap(values, input._logical_dtype), _wrap(indices))


def deg2rad(input):
    return _wrap(np.deg2rad(input._data), input._logical_dtype)


def diag(input, diagonal=0):
    return _wrap(np.diag(input._data, k=diagonal), input._logical_dtype)


def dot(input, other):
    return _wrap(np.dot(input._data.ravel(), other._data.ravel()), input._logical_dtype)


def empty_strided(size, stride, *, dtype=None, device=None):
    nd = _to_np(dtype) or np.float32
    return _wrap(np.empty(size, dtype=nd), dtype if dtype is bfloat16 else None)


def eq(input, other):
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(input._data == od)


def exp2(input):
    return _wrap(np.exp2(input._data), input._logical_dtype)


def expm1(input):
    return _wrap(np.expm1(input._data), input._logical_dtype)


def eye(n, m=None, *, dtype=None, device=None):
    if m is None:
        m = n
    nd = _to_np(dtype) or np.float32
    return _wrap(np.eye(n, m, dtype=nd), dtype if dtype is bfloat16 else None)


def fill(input, value):
    out = np.full_like(input._data, value)
    return _wrap(out, input._logical_dtype)


def flip(input, dims):
    if isinstance(dims, int):
        dims = (dims,)
    return _wrap(np.flip(input._data, axis=dims).copy(), input._logical_dtype)


def fmod(input, other):
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(np.fmod(input._data, od), input._logical_dtype)


def frac(input):
    return _wrap(input._data - np.trunc(input._data), input._logical_dtype)


def full_like(input, fill_value, *, dtype=None, device=None):
    if dtype is not None:
        nd = _to_np(dtype)
        return _wrap(np.full_like(input._data, fill_value, dtype=nd), dtype if dtype is bfloat16 else None)
    return _wrap(np.full_like(input._data, fill_value), input._logical_dtype)


def gather(input, dim, index):
    idx = index._data if isinstance(index, Tensor) else np.asarray(index, dtype=np.int64)
    return _wrap(np.take_along_axis(input._data, idx, axis=dim), input._logical_dtype)


def ge(input, other):
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(input._data >= od)


def get_default_dtype():
    return float32


def gt(input, other):
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(input._data > od)


def hstack(tensors):
    arrays = [t._data for t in tensors]
    return _wrap(np.hstack(arrays))


def hypot(input, other):
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(np.hypot(input._data, od), input._logical_dtype)


def is_floating_point(input):
    if isinstance(input, Tensor):
        return input._data.dtype.kind == 'f'
    return False


def isclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(np.isclose(input._data, od, rtol=rtol, atol=atol, equal_nan=equal_nan))


def isfinite(input):
    return _wrap(np.isfinite(input._data))


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    import torch.nn.functional as F
    return F.layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)


def le(input, other):
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(input._data <= od)


def lt(input, other):
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(input._data < od)


def lerp(input, end, weight):
    id = input._data.astype(np.float64)
    ed = end._data.astype(np.float64)
    wd = weight._data.astype(np.float64) if isinstance(weight, Tensor) else float(weight)
    result = id + wd * (ed - id)
    return _wrap(result.astype(input._data.dtype), input._logical_dtype)


def log10(input):
    return _wrap(np.log10(input._data), input._logical_dtype)


def log1p(input):
    return _wrap(np.log1p(input._data), input._logical_dtype)


def log_softmax(input, dim=-1):
    x = input._data.astype(np.float64)
    e = np.exp(x - np.max(x, axis=dim, keepdims=True))
    s = e / np.sum(e, axis=dim, keepdims=True)
    return _wrap(np.log(s).astype(input._data.dtype), input._logical_dtype)


def logspace(start, end, steps, base=10.0, *, dtype=None, device=None):
    nd = _to_np(dtype) or np.float32
    return _wrap(np.logspace(start, end, steps, base=base, dtype=nd))


long = int64


def mean(input, dim=None, keepdim=False, *, dtype=None):
    if isinstance(input, Tensor):
        if isinstance(dim, (list, tuple)):
            result = input._data.astype(np.float64)
            for d in sorted(dim, reverse=True):
                result = np.mean(result, axis=d, keepdims=keepdim)
            out_dtype = _to_np(dtype) if dtype else input._data.dtype
            return _wrap(result.astype(out_dtype), dtype if dtype is bfloat16 else input._logical_dtype)
        out_dtype = _to_np(dtype) if dtype else input._data.dtype
        return _wrap(np.mean(input._data, axis=dim, keepdims=keepdim).astype(out_dtype),
                     dtype if dtype is bfloat16 else input._logical_dtype)
    return _wrap(np.mean(np.asarray(input)))


def mm(input, mat2):
    a, b = input._data, mat2._data
    if a.dtype == np.float16:
        return _wrap((a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16), input._logical_dtype)
    if a.dtype == np.float32:
        return _wrap((a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32), input._logical_dtype)
    return _wrap(a @ b, input._logical_dtype)


def mv(input, vec):
    a, b = input._data, vec._data
    if a.dtype == np.float16:
        return _wrap((a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16), input._logical_dtype)
    return _wrap(a @ b, input._logical_dtype)


def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    return _wrap(np.nan_to_num(input._data, nan=nan, posinf=posinf, neginf=neginf), input._logical_dtype)


def native_layer_norm(input, normalized_shape, weight, bias, eps):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    ndim = len(normalized_shape)
    axes = tuple(range(input._data.ndim - ndim, input._data.ndim))
    x = input._data.astype(np.float32)
    mean_val = np.mean(x, axis=axes, keepdims=True)
    var_val = np.var(x, axis=axes, keepdims=True)
    inv_std = 1.0 / np.sqrt(var_val + eps)
    normed = (x - mean_val) * inv_std
    if weight is not None:
        normed = normed * weight._data.astype(np.float32)
    if bias is not None:
        normed = normed + bias._data.astype(np.float32)
    mean_out = _wrap(mean_val.astype(input._data.dtype), input._logical_dtype)
    inv_std_out = _wrap(inv_std.astype(input._data.dtype), input._logical_dtype)
    return _wrap(normed.astype(input._data.dtype), input._logical_dtype), mean_out, inv_std_out


def ne(input, other):
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    return _wrap(input._data != od)


def numel(input):
    return input._data.size


def polar(abs, angle):
    a = abs._data if isinstance(abs, Tensor) else np.asarray(abs)
    ang = angle._data if isinstance(angle, Tensor) else np.asarray(angle)
    return _wrap((a * np.cos(ang) + 1j * a * np.sin(ang)).astype(np.complex64))


def pow(input, exponent):
    id_ = input._data if isinstance(input, Tensor) else np.asarray(input)
    ed = exponent._data if isinstance(exponent, Tensor) else np.asarray(exponent)
    # Promote fp16 to fp32 for precision
    orig_dtype = id_.dtype if isinstance(id_, np.ndarray) else None
    if orig_dtype == np.float16:
        id_ = id_.astype(np.float32)
        result = np.power(id_, ed).astype(np.float16)
    else:
        result = np.power(id_, ed)
    return _wrap(result, input._logical_dtype if isinstance(input, Tensor) else None)


def prod(input, dim=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        nd = _to_np(dtype)
        return _wrap(np.prod(input._data, axis=dim, keepdims=keepdim).astype(nd))
    return _wrap(np.prod(input._data, axis=dim, keepdims=keepdim), input._logical_dtype)


def promote_types(type1, type2):
    # Special handling for bfloat16
    if type1 is bfloat16 and type2 is bfloat16:
        return bfloat16
    if type1 is bfloat16:
        type1 = float32
    if type2 is bfloat16:
        type2 = float32
    nd1 = _to_np(type1)
    nd2 = _to_np(type2)
    result_np = np.result_type(nd1, nd2)
    return _from_np(result_np)


def rad2deg(input):
    return _wrap(np.rad2deg(input._data), input._logical_dtype)


def randint(low, high=None, size=None, *, dtype=None, device=None):
    if high is None:
        high = low
        low = 0
    if size is None:
        size = ()
    nd = _to_np(dtype) or np.int64
    return _wrap(np.random.randint(low, high, size=size).astype(nd))


def reciprocal(input):
    return _wrap(1.0 / input._data, input._logical_dtype)


def relu(input):
    return _wrap(np.maximum(input._data, 0), input._logical_dtype)


def remainder(input, other):
    id_ = input._data if isinstance(input, Tensor) else np.asarray(input)
    od = other._data if isinstance(other, Tensor) else np.asarray(other)
    logical = input._logical_dtype if isinstance(input, Tensor) else None
    return _wrap(np.remainder(id_, od), logical)


def select_scatter(input, src, dim, index):
    out = input._data.copy()
    idx = [slice(None)] * out.ndim
    idx[dim] = index
    out[tuple(idx)] = src._data if isinstance(src, Tensor) else src
    return _wrap(out, input._logical_dtype)


def sign(input):
    return _wrap(np.sign(input._data), input._logical_dtype)


def sin(input):
    return _wrap(np.sin(input._data), input._logical_dtype)


def slice_scatter(input, src, dim=0, start=None, end=None, step=1):
    out = input._data.copy()
    idx = [slice(None)] * out.ndim
    idx[dim] = slice(start, end, step)
    out[tuple(idx)] = src._data if isinstance(src, Tensor) else src
    return _wrap(out, input._logical_dtype)


def threshold(input, threshold, value):
    data = input._data
    return _wrap(np.where(data > threshold, data, value).astype(data.dtype), input._logical_dtype)


def topk(input, k, dim=-1, largest=True, sorted=True):
    data = input._data
    if largest:
        idx = np.argsort(-data, axis=dim)
    else:
        idx = np.argsort(data, axis=dim)
    slices = [slice(None)] * data.ndim
    slices[dim] = slice(0, k)
    top_idx = idx[tuple(slices)]
    top_val = np.take_along_axis(data, top_idx, axis=dim)
    return _MaxMinResult(_wrap(top_val, input._logical_dtype), _wrap(top_idx.astype(np.int64)))


def tril(input, diagonal=0):
    return _wrap(np.tril(input._data, k=diagonal), input._logical_dtype)


def triu(input, diagonal=0):
    return _wrap(np.triu(input._data, k=diagonal), input._logical_dtype)


def trunc(input):
    return _wrap(np.trunc(input._data), input._logical_dtype)


def var(input, dim=None, unbiased=True, keepdim=False, *, correction=None):
    if correction is not None:
        ddof = correction
    else:
        ddof = 1 if unbiased else 0
    data = input._data
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    return _wrap(np.var(data, axis=dim, ddof=ddof, keepdims=keepdim).astype(input._data.dtype), input._logical_dtype)


def var_mean(input, dim=None, unbiased=True, keepdim=False, *, correction=None):
    if correction is not None:
        ddof = correction
    else:
        ddof = 1 if unbiased else 0
    data = input._data
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    v = np.var(data, axis=dim, ddof=ddof, keepdims=keepdim).astype(input._data.dtype)
    m = np.mean(data, axis=dim, keepdims=keepdim).astype(input._data.dtype)
    return _wrap(v, input._logical_dtype), _wrap(m, input._logical_dtype)


def vdot(input, other):
    return _wrap(np.vdot(input._data.ravel(), other._data.ravel()), input._logical_dtype)


def view_as_complex(input):
    data = input._data
    if data.shape[-1] != 2:
        raise RuntimeError("view_as_complex requires last dim == 2")
    real = data[..., 0]
    imag = data[..., 1]
    return _wrap((real + 1j * imag).astype(np.complex64))


def vstack(tensors):
    arrays = [t._data for t in tensors]
    return _wrap(np.vstack(arrays))


# ── RNG ─────────────────────────────────────────────────────────────────────

def manual_seed(seed):
    np.random.seed(seed)


def seed():
    np.random.seed()


# ── Device helpers ──────────────────────────────────────────────────────────

from torch._tensor import _Device  # noqa: F401

def device(d):
    if isinstance(d, _Device):
        return d
    return _Device(d) if isinstance(d, str) else d

class Size(tuple):
    def numel(self):
        result = 1
        for s in self:
            result *= s
        return result


# ── Memory format sentinels ─────────────────────────────────────────────────

class _MemoryFormat:
    def __init__(self, name):
        self._name = name
    def __repr__(self):
        return f"torch.{self._name}"

preserve_format = _MemoryFormat("preserve_format")
contiguous_format = _MemoryFormat("contiguous_format")
channels_last = _MemoryFormat("channels_last")
channels_last_3d = _MemoryFormat("channels_last_3d")

# ── Layout sentinels ───────────────────────────────────────────────────────

class _Layout:
    def __init__(self, name):
        self._name = name
    def __repr__(self):
        return f"torch.{self._name}"

strided = _Layout("strided")


# ── finfo / iinfo ──────────────────────────────────────────────────────────

class finfo:
    """Minimal torch.finfo shim backed by numpy."""
    def __init__(self, dtype_):
        nd = _to_np(dtype_) or np.float32
        info = np.finfo(nd)
        self.bits = info.bits
        self.eps = float(info.eps)
        self.max = float(info.max)
        self.min = float(info.min)
        self.tiny = float(info.tiny)
        self.dtype = dtype_

class iinfo:
    """Minimal torch.iinfo shim backed by numpy."""
    def __init__(self, dtype_):
        nd = _to_np(dtype_) or np.int32
        info = np.iinfo(nd)
        self.bits = info.bits
        self.max = int(info.max)
        self.min = int(info.min)
        self.dtype = dtype_


# ── ops namespace ───────────────────────────────────────────────────────────

class _Aten:
    @staticmethod
    def _softmax(input, dim, half_to_float):
        import torch.nn.functional as F
        return F.softmax(input, dim=dim)

class _Ops:
    aten = _Aten()

ops = _Ops()


# ── linalg namespace ───────────────────────────────────────────────────────

class _Linalg:
    @staticmethod
    def vector_norm(input, ord=2, dim=None, keepdim=False, *, dtype=None):
        data = input._data.astype(np.float64)
        if ord == float('inf'):
            result = np.max(np.abs(data), axis=dim, keepdims=keepdim)
        elif ord == float('-inf'):
            result = np.min(np.abs(data), axis=dim, keepdims=keepdim)
        elif ord == 0:
            result = np.sum(data != 0, axis=dim, keepdims=keepdim).astype(np.float64)
        elif ord == 1:
            result = np.sum(np.abs(data), axis=dim, keepdims=keepdim)
        elif ord == 2:
            result = np.sqrt(np.sum(data * data, axis=dim, keepdims=keepdim))
        else:
            result = np.power(np.sum(np.power(np.abs(data), ord), axis=dim, keepdims=keepdim), 1.0 / ord)
        out_dtype = _to_np(dtype) if dtype else input._data.dtype
        return _wrap(result.astype(out_dtype), input._logical_dtype)

linalg = _Linalg()

import sys
sys.modules["torch.linalg"] = linalg
