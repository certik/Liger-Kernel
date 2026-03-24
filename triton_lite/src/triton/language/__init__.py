"""
triton.language — Simulates Triton's tl.* operations using torch tensors.
"""

import torch

from triton._interpreter import PointerBlock, TensorPointer, get_program_id


# ─── dtype base class (torch._dynamo checks for triton.language.dtype) ───
class dtype:
    """Base type for Triton dtype descriptors. Maps to torch dtypes."""
    pass


# ─── dtype aliases (map to torch dtypes) ───
float16 = torch.float16
float32 = torch.float32
bfloat16 = torch.bfloat16
int8 = torch.int8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64
uint8 = torch.uint8
uint16 = torch.uint16
uint32 = torch.uint32
uint64 = torch.int64  # torch has no uint64, use int64
bool_ = torch.bool


class constexpr:
    """Marker for compile-time constant parameters in Triton kernels.

    In real Triton, constexpr parameters are baked into the compiled kernel.
    Here we just store and forward the value.
    """

    def __init__(self, value=None):
        self.value = value

    def __repr__(self):
        return f"constexpr({self.value})"


# ─── Core operations ───

def program_id(axis):
    """Return the program id for the given axis."""
    pid = get_program_id(axis)
    # Return a 0-dim torch tensor so .to() works on it
    return torch.tensor(pid, dtype=torch.int32)


def arange(start, end):
    """Return a 1-D tensor [start, start+1, ..., end-1]."""
    return torch.arange(start, end, dtype=torch.int32)


def load(pointer, mask=None, other=0, eviction_policy="", cache_modifier="", volatile=False):
    """Load values from memory via pointer(s)."""
    if isinstance(pointer, PointerBlock):
        offsets = pointer.offsets
        if isinstance(offsets, torch.Tensor):
            orig_shape = offsets.shape
            flat_offsets = offsets.reshape(-1).long()
        else:
            orig_shape = None
            flat_offsets = offsets

        if mask is not None and isinstance(mask, torch.Tensor):
            flat_mask = mask.broadcast_to(orig_shape).reshape(-1).bool() if orig_shape else mask.bool()
            result = torch.full(flat_offsets.shape, other,
                                dtype=pointer.data.dtype, device=pointer.data.device)
            if flat_mask.any():
                result[flat_mask] = pointer.data[flat_offsets[flat_mask]]
            if orig_shape is not None:
                result = result.reshape(orig_shape)
            return result
        else:
            result = pointer.data[flat_offsets]
            if orig_shape is not None:
                result = result.reshape(orig_shape)
            return result
    elif isinstance(pointer, TensorPointer):
        # Scalar load
        return pointer.data[pointer.offset]
    else:
        raise TypeError(f"load: unsupported pointer type {type(pointer)}")


def store(pointer, value, mask=None, eviction_policy="", cache_modifier=""):
    """Store values to memory via pointer(s)."""
    if isinstance(pointer, PointerBlock):
        offsets = pointer.offsets
        if isinstance(offsets, torch.Tensor):
            orig_shape = offsets.shape
            flat_offsets = offsets.reshape(-1).long()
        else:
            orig_shape = None
            flat_offsets = offsets

        # Cast value to match destination dtype (Triton does this implicitly)
        if isinstance(value, torch.Tensor) and value.dtype != pointer.data.dtype:
            value = value.to(pointer.data.dtype)

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                flat_mask = mask.broadcast_to(orig_shape).reshape(-1).bool() if orig_shape else mask.bool()
                if flat_mask.any():
                    flat_value = value.reshape(-1) if isinstance(value, torch.Tensor) and value.ndim > 1 else value
                    if isinstance(flat_value, torch.Tensor):
                        pointer.data[flat_offsets[flat_mask]] = flat_value[flat_mask]
                    else:
                        pointer.data[flat_offsets[flat_mask]] = flat_value
            else:
                if mask:
                    pointer.data[flat_offsets] = value.reshape(-1) if isinstance(value, torch.Tensor) and value.ndim > 1 else value
        else:
            flat_value = value.reshape(-1) if isinstance(value, torch.Tensor) and value.ndim > 1 else value
            pointer.data[flat_offsets] = flat_value if isinstance(flat_value, torch.Tensor) else torch.tensor(flat_value)
    elif isinstance(pointer, TensorPointer):
        if isinstance(value, torch.Tensor) and value.dtype != pointer.data.dtype:
            value = value.to(pointer.data.dtype)
        pointer.data[pointer.offset] = value
    else:
        raise TypeError(f"store: unsupported pointer type {type(pointer)}")


def zeros(shape, dtype=float32):
    """Create a tensor of zeros."""
    if isinstance(shape, (list, tuple)):
        if len(shape) == 0:
            return torch.zeros(1, dtype=dtype).squeeze()
        return torch.zeros(*shape, dtype=dtype)
    return torch.zeros(shape, dtype=dtype)


def full(shape, value, dtype=float32):
    """Create a tensor filled with a value."""
    if isinstance(shape, (list, tuple)):
        return torch.full(shape, value, dtype=dtype)
    return torch.full((shape,), value, dtype=dtype)


def trans(input):
    """Transpose a 2D tensor."""
    if input.ndim == 2:
        return input.T
    return input.transpose(-2, -1)


def where(condition, x, y):
    """Element-wise conditional."""
    return torch.where(condition, x, y)


def sum(input, axis=None):
    """Sum reduction."""
    if axis is not None:
        return torch.sum(input, dim=axis)
    return torch.sum(input)


def cumsum(input, axis=0):
    """Cumulative sum along an axis."""
    return torch.cumsum(input, dim=axis)


def cast(input, dtype):
    """Cast a tensor to a different dtype."""
    from triton._interpreter import _PointerDtype
    if isinstance(dtype, _PointerDtype):
        dtype = dtype.element_ty
    if isinstance(input, torch.Tensor):
        return input.to(dtype)
    return torch.tensor(input, dtype=dtype)


def max(input, axis=None):
    """Max reduction."""
    if axis is not None:
        return torch.max(input, dim=axis).values
    return torch.max(input)


def min(input, axis=None):
    """Min reduction."""
    if axis is not None:
        return torch.min(input, dim=axis).values
    return torch.min(input)


def abs(input):
    return torch.abs(input)


def exp(input):
    return torch.exp(input)


def log(input):
    return torch.log(input)


def sqrt(input):
    return torch.sqrt(input)


def sigmoid(input):
    return torch.sigmoid(input)


def dot(a, b, allow_tf32=True):
    """Matrix dot product."""
    return torch.matmul(a, b)


def debug_barrier():
    """No-op barrier for CPU interpretation."""
    pass


def multiple_of(input, value):
    """Hint that input is a multiple of value. No-op in interpreter."""
    return input


def maximum(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    return torch.maximum(a, b)


def minimum(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    return torch.minimum(a, b)


# Atomic operations (simplified — just do the op directly)

def atomic_add(pointer, value, mask=None):
    if isinstance(pointer, PointerBlock):
        offsets = pointer.offsets.long()
        if mask is not None:
            valid = mask.bool()
            pointer.data[offsets[valid]] += value[valid] if isinstance(value, torch.Tensor) else value
        else:
            pointer.data[offsets] += value if isinstance(value, torch.Tensor) else value
    elif isinstance(pointer, TensorPointer):
        pointer.data[pointer.offset] += value


def atomic_max(pointer, value, mask=None):
    if isinstance(pointer, PointerBlock):
        offsets = pointer.offsets.long()
        if mask is not None:
            valid = mask.bool()
            for idx in offsets[valid]:
                pointer.data[idx] = torch.max(pointer.data[idx], value)
        else:
            for idx in offsets:
                pointer.data[idx] = torch.max(pointer.data[idx], value)


# cdiv at the language level too (some kernels use tl.cdiv)
def cdiv(a, b):
    return (a + b - 1) // b


def static_range(start, end=None, step=1):
    """Equivalent to range() — in real Triton this unrolls at compile time."""
    if end is None:
        return builtins.range(start)
    return builtins.range(start, end, step)


# tl.range is the same as static_range in newer Triton versions
import builtins
range = static_range


def split(tensor):
    """Split a 2D tensor along the last dimension, returning individual columns.

    In Triton, tl.split on a (N, K) block returns K tensors of shape (N,).
    """
    if tensor.ndim == 2:
        return tuple(tensor[:, i] for i in range(tensor.shape[1]))
    raise ValueError(f"tl.split expects a 2D tensor, got shape {tensor.shape}")


def join(a, b):
    """Join two 1D tensors into a 2D tensor (inverse of split)."""
    return torch.stack([a, b], dim=-1)


# ─── tl.math submodule ───
class _MathModule:
    """Provides tl.math.* functions."""

    @staticmethod
    def fma(a, b, c):
        """Fused multiply-add: a * b + c."""
        return a * b + c

    @staticmethod
    def tanh(x):
        return torch.tanh(x) if isinstance(x, torch.Tensor) else torch.tanh(torch.tensor(x)).item()

    @staticmethod
    def rsqrt(x):
        return torch.rsqrt(x) if isinstance(x, torch.Tensor) else torch.rsqrt(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def exp(x):
        return torch.exp(x) if isinstance(x, torch.Tensor) else torch.exp(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def log(x):
        return torch.log(x) if isinstance(x, torch.Tensor) else torch.log(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def sqrt(x):
        return torch.sqrt(x) if isinstance(x, torch.Tensor) else torch.sqrt(torch.tensor(x, dtype=torch.float32)).item()


math = _MathModule()
