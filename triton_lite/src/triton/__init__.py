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


class _CallableInt(int):
    """An int that can also be called (returns itself).

    In Triton, tensor.numel is a property returning an int.
    In PyTorch, tensor.numel is a method. This class bridges both usages
    so that both ``x.numel`` and ``x.numel()`` work.
    """
    def __call__(self):
        return int(self)


# Monkey-patch torch.Tensor.numel so it can be used as a property-like access
# (x.numel works like an int, x.numel() also works)
_original_numel = torch.Tensor.numel

class _NumelProperty:
    """Descriptor that returns _CallableInt from tensor.numel."""
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return _CallableInt(_original_numel(obj))

torch.Tensor.numel = _NumelProperty()


# Add primitive_bitwidth to torch dtypes (Triton exposes this on dtype objects)
_DTYPE_BITWIDTHS = {
    torch.float16: 16, torch.bfloat16: 16, torch.float32: 32, torch.float64: 64,
    torch.int8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64,
    torch.uint8: 8, torch.bool: 1,
}

class _DtypeWithBitwidth:
    """Wraps a torch.dtype to add .primitive_bitwidth.

    Designed to be transparent — can be used anywhere a torch.dtype is expected.
    """
    __slots__ = ('_dtype', 'primitive_bitwidth')

    def __init__(self, dtype):
        if isinstance(dtype, _DtypeWithBitwidth):
            dtype = dtype._dtype
        object.__setattr__(self, '_dtype', dtype)
        object.__setattr__(self, 'primitive_bitwidth', _DTYPE_BITWIDTHS.get(dtype, 32))

    def __getattr__(self, name):
        # Proxy all unknown attributes to the underlying torch.dtype
        return getattr(self._dtype, name)

    def __eq__(self, other):
        if isinstance(other, _DtypeWithBitwidth):
            return self._dtype == other._dtype
        return self._dtype == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._dtype)

    def __repr__(self):
        return repr(self._dtype)

    def __reduce__(self):
        return (self._dtype.__reduce__())


# Override torch.Tensor.dtype to return _DtypeWithBitwidth
_original_dtype_descriptor = torch.Tensor.dtype

class _DtypeProperty:
    """Descriptor that returns _DtypeWithBitwidth from tensor.dtype."""
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        raw_dtype = _original_dtype_descriptor.__get__(obj, objtype)
        return _DtypeWithBitwidth(raw_dtype)

torch.Tensor.dtype = _DtypeProperty()


def _unwrap_dtype(dtype):
    """Unwrap _DtypeWithBitwidth to plain torch.dtype."""
    if isinstance(dtype, _DtypeWithBitwidth):
        return dtype._dtype
    return dtype


# Monkey-patch torch.Tensor.to to support bitcast=True and _DtypeWithBitwidth
_original_to = torch.Tensor.to

def _to_with_bitcast(self, *args, **kwargs):
    bitcast = kwargs.pop("bitcast", False)
    # Unwrap _DtypeWithBitwidth in args
    args = tuple(_unwrap_dtype(a) if isinstance(a, _DtypeWithBitwidth) else a for a in args)
    if bitcast and args:
        target_dtype = args[0]
        return self.view(target_dtype)
    return _original_to(self, *args, **kwargs)

torch.Tensor.to = _to_with_bitcast


# Patch torch functions to accept _DtypeWithBitwidth in dtype= kwarg
for _fn_name in ('full', 'zeros', 'ones', 'empty', 'zeros_like', 'ones_like',
                  'empty_like', 'full_like', 'rand', 'randn', 'tensor', 'as_tensor',
                  'empty_strided', 'arange', 'linspace', 'logspace'):
    _orig = getattr(torch, _fn_name, None)
    if _orig is not None:
        def _make_wrapper(fn):
            def _wrapper(*args, **kwargs):
                if 'dtype' in kwargs and isinstance(kwargs['dtype'], _DtypeWithBitwidth):
                    kwargs['dtype'] = kwargs['dtype']._dtype
                return fn(*args, **kwargs)
            _wrapper.__name__ = fn.__name__
            return _wrapper
        setattr(torch, _fn_name, _make_wrapper(_orig))

# Patch torch functions that take dtype as positional args
for _fn_name in ('promote_types', 'result_type', 'can_cast', 'finfo', 'iinfo'):
    _orig = getattr(torch, _fn_name, None)
    if _orig is not None:
        def _make_pos_wrapper(fn):
            def _wrapper(*args, **kwargs):
                args = tuple(a._dtype if isinstance(a, _DtypeWithBitwidth) else a for a in args)
                return fn(*args, **kwargs)
            _wrapper.__name__ = fn.__name__
            return _wrapper
        setattr(torch, _fn_name, _make_pos_wrapper(_orig))


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
