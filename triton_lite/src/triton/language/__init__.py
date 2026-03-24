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
            offsets = offsets.long()

        if mask is not None and isinstance(mask, torch.Tensor):
            # Only load from valid (masked) indices to avoid out-of-bounds
            result = torch.full((offsets.shape[0],), other,
                                dtype=pointer.data.dtype, device=pointer.data.device)
            valid = mask.bool()
            if valid.any():
                result[valid] = pointer.data[offsets[valid]]
            return result
        else:
            return pointer.data[offsets]
    elif isinstance(pointer, TensorPointer):
        # Scalar load
        return pointer.data[pointer.offset]
    else:
        raise TypeError(f"load: unsupported pointer type {type(pointer)}")


def store(pointer, value, mask=None, eviction_policy=""):
    """Store values to memory via pointer(s)."""
    if isinstance(pointer, PointerBlock):
        offsets = pointer.offsets
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.long()

        # Cast value to match destination dtype (Triton does this implicitly)
        if isinstance(value, torch.Tensor) and value.dtype != pointer.data.dtype:
            value = value.to(pointer.data.dtype)

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                valid = mask.bool()
                if valid.any():
                    if isinstance(value, torch.Tensor):
                        pointer.data[offsets[valid]] = value[valid]
                    else:
                        pointer.data[offsets[valid]] = value
            else:
                if mask:
                    pointer.data[offsets] = value
        else:
            pointer.data[offsets] = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    elif isinstance(pointer, TensorPointer):
        if isinstance(value, torch.Tensor) and value.dtype != pointer.data.dtype:
            value = value.to(pointer.data.dtype)
        pointer.data[pointer.offset] = value
    else:
        raise TypeError(f"store: unsupported pointer type {type(pointer)}")


def zeros(shape, dtype=float32):
    """Create a tensor of zeros."""
    if isinstance(shape, (list, tuple)):
        return torch.zeros(*shape, dtype=dtype)
    return torch.zeros(shape, dtype=dtype)


def full(shape, value, dtype=float32):
    """Create a tensor filled with a value."""
    if isinstance(shape, (list, tuple)):
        return torch.full(shape, value, dtype=dtype)
    return torch.full((shape,), value, dtype=dtype)


def where(condition, x, y):
    """Element-wise conditional."""
    return torch.where(condition, x, y)


def sum(input, axis=None):
    """Sum reduction."""
    if axis is not None:
        return torch.sum(input, dim=axis)
    return torch.sum(input)


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
    return torch.maximum(a, b)


def minimum(a, b):
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
