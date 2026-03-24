"""
triton.language — Simulates Triton's tl.* operations using torch tensors.
"""

import builtins

import torch

from triton._interpreter import PointerBlock, TensorPointer, get_program_id, get_num_programs


# ─── dtype base class (torch._dynamo checks for triton.language.dtype) ───
class dtype:
    """Base type for Triton dtype descriptors. Maps to torch dtypes."""
    pass


# ─── dtype aliases (map to torch dtypes) ───
float16 = torch.float16
float32 = torch.float32
float64 = torch.float64
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
# float8e5 placeholder — not natively supported in torch, alias to float16
float8e5 = torch.float16
int1 = torch.bool


class constexpr:
    """Marker for compile-time constant parameters in Triton kernels.

    In real Triton, constexpr parameters are baked into the compiled kernel.
    Here we just store and forward the value, and support arithmetic so
    constexpr values can be used transparently in expressions.
    """

    def __init__(self, value=None):
        self.value = value

    def __repr__(self):
        return f"constexpr({self.value})"

    # Make constexpr transparent for numeric operations
    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __eq__(self, other):
        if isinstance(other, constexpr):
            return self.value == other.value
        return self.value == other

    def __hash__(self):
        return hash(self.value)

    def __add__(self, other):
        other_val = other.value if isinstance(other, constexpr) else other
        return self.value + other_val

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        other_val = other.value if isinstance(other, constexpr) else other
        return self.value - other_val

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        other_val = other.value if isinstance(other, constexpr) else other
        return self.value * other_val

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        other_val = other.value if isinstance(other, constexpr) else other
        return self.value / other_val

    def __rtruediv__(self, other):
        return other / self.value

    def __neg__(self):
        return -self.value

    def __lt__(self, other):
        other_val = other.value if isinstance(other, constexpr) else other
        return self.value < other_val

    def __le__(self, other):
        other_val = other.value if isinstance(other, constexpr) else other
        return self.value <= other_val

    def __gt__(self, other):
        other_val = other.value if isinstance(other, constexpr) else other
        return self.value > other_val

    def __ge__(self, other):
        other_val = other.value if isinstance(other, constexpr) else other
        return self.value >= other_val


def _unwrap(x):
    """Unwrap constexpr values to their raw Python values."""
    if isinstance(x, constexpr):
        return x.value
    return x


# ─── Helpers ───

def _ensure_tensor(x, reference_dtype=None):
    """Convert scalars to tensors if needed."""
    if isinstance(x, torch.Tensor):
        return x
    dtype = reference_dtype if reference_dtype else torch.float32
    return torch.tensor(x, dtype=dtype)


# ─── Core operations ───

def program_id(axis):
    """Return the program id for the given axis."""
    pid = get_program_id(axis)
    # Return a 0-dim torch tensor so .to() works on it
    return torch.tensor(pid, dtype=torch.int32)


def num_programs(axis):
    """Return the number of programs for the given axis."""
    return get_num_programs(axis)


def arange(start, end):
    """Return a 1-D tensor [start, start+1, ..., end-1]."""
    return torch.arange(start, end, dtype=torch.int32)


def load(pointer, mask=None, other=0, eviction_policy="", cache_modifier="", volatile=False):
    """Load values from memory via pointer(s).

    Promotes fp16/bf16 results to fp32 to match Triton's computation
    precision (Triton promotes scalar-block operations to fp32).
    """
    other = _unwrap(other)
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
            if other is None:
                fill_other = 0
            elif isinstance(other, (int, float)):
                fill_other = float(other)
            else:
                fill_other = other
            result = torch.full(flat_offsets.shape, fill_other,
                                dtype=pointer.data.dtype, device=pointer.data.device)
            if flat_mask.any():
                valid_offsets = flat_offsets[flat_mask]
                valid_offsets = valid_offsets.clamp(0, pointer.data.numel() - 1)
                result[flat_mask] = pointer.data[valid_offsets]
            if orig_shape is not None:
                result = result.reshape(orig_shape)
            if result.dtype in (torch.float16, torch.bfloat16):
                result = result.float()
            return result
        else:
            safe_offsets = flat_offsets.clamp(0, pointer.data.numel() - 1)
            result = pointer.data[safe_offsets]
            if orig_shape is not None:
                result = result.reshape(orig_shape)
            if result.dtype in (torch.float16, torch.bfloat16):
                result = result.float()
            return result
    elif isinstance(pointer, TensorPointer):
        result = pointer.data[pointer.offset].clone()
        if result.dtype in (torch.float16, torch.bfloat16):
            result = result.float()
        return result
    else:
        raise TypeError(f"load: unsupported pointer type {type(pointer)}")


def store(pointer, value, mask=None, eviction_policy="", cache_modifier=""):
    """Store values to memory via pointer(s)."""
    value = _unwrap(value)
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

        # Flatten multi-dim values; keep scalars/0-dim as broadcastable
        if isinstance(value, torch.Tensor) and value.ndim > 1:
            flat_value = value.reshape(-1)
        elif isinstance(value, torch.Tensor) and value.ndim == 0:
            flat_value = value.item()
        else:
            flat_value = value

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                flat_mask = mask.broadcast_to(orig_shape).reshape(-1).bool() if orig_shape else mask.bool()
                if flat_mask.any():
                    if isinstance(flat_value, torch.Tensor):
                        pointer.data[flat_offsets[flat_mask]] = flat_value[flat_mask]
                    else:
                        pointer.data[flat_offsets[flat_mask]] = flat_value
            else:
                if mask:
                    pointer.data[flat_offsets] = flat_value
        else:
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
    x = _unwrap(x)
    y = _unwrap(y)
    return torch.where(condition, x, y)


def reshape(tensor, shape):
    """Reshape a tensor."""
    if isinstance(shape, (list, tuple)):
        return tensor.reshape(*shape)
    return tensor.reshape(shape)


def broadcast_to(tensor, shape):
    """Broadcast a tensor to a given shape."""
    if isinstance(shape, (list, tuple)):
        return tensor.broadcast_to(*shape)
    return tensor.broadcast_to(shape)


# ─── Reduction operations ───

def sum(input, axis=None):
    """Sum reduction."""
    if axis is not None:
        return torch.sum(input, dim=axis)
    return torch.sum(input)


def cumsum(input, axis=0):
    """Cumulative sum along an axis."""
    return torch.cumsum(input, dim=axis)


def max(input, axis=None, return_indices=False):
    """Max reduction."""
    if axis is not None:
        result = torch.max(input, dim=axis)
        if return_indices:
            return result.values, result.indices
        return result.values
    if return_indices:
        flat = input.reshape(-1)
        idx = torch.argmax(flat)
        return flat[idx], idx
    return torch.max(input)


def min(input, axis=None, return_indices=False):
    """Min reduction."""
    if axis is not None:
        result = torch.min(input, dim=axis)
        if return_indices:
            return result.values, result.indices
        return result.values
    if return_indices:
        flat = input.reshape(-1)
        idx = torch.argmin(flat)
        return flat[idx], idx
    return torch.min(input)


def argmax(input, axis=None):
    """Argmax reduction."""
    if axis is not None:
        return torch.argmax(input, dim=axis)
    return torch.argmax(input)


def argmin(input, axis=None):
    """Argmin reduction."""
    if axis is not None:
        return torch.argmin(input, dim=axis)
    return torch.argmin(input)


def reduce(input, axis, combine_fn):
    """General reduction with a user-specified combine function.

    The combine_fn takes two tensors and returns one. We apply it
    iteratively along the specified axis (like a fold/reduce).
    """
    if isinstance(input, tuple):
        # Multi-operand reduce: input is a tuple of tensors
        n = input[0].shape[axis]
        # Start with slices at index 0
        accum = tuple(t.select(axis, 0) for t in input)
        for i in builtins.range(1, n):
            slices = tuple(t.select(axis, i) for t in input)
            accum = combine_fn(*accum, *slices)
            if not isinstance(accum, tuple):
                accum = (accum,)
        return accum if len(accum) > 1 else accum[0]
    else:
        # Single-operand reduce
        n = input.shape[axis]
        accum = input.select(axis, 0)
        for i in builtins.range(1, n):
            accum = combine_fn(accum, input.select(axis, i))
        return accum


def associative_scan(input, axis, combine_fn):
    """Parallel prefix scan (inclusive) with a user-specified combine function.

    Supports both single-tensor and multi-tensor (tuple) inputs.
    """
    if isinstance(input, tuple):
        # Multi-operand scan
        n = input[0].shape[axis]
        # Build output lists
        results = [[] for _ in input]
        accum = tuple(t.select(axis, 0) for t in input)
        for lst, a in zip(results, accum):
            lst.append(a)
        for i in builtins.range(1, n):
            slices = tuple(t.select(axis, i) for t in input)
            combined = combine_fn(*accum, *slices)
            if not isinstance(combined, tuple):
                combined = (combined,)
            accum = combined
            for lst, a in zip(results, accum):
                lst.append(a)
        return tuple(torch.stack(lst, dim=axis) for lst in results)
    else:
        # Single-operand scan
        n = input.shape[axis]
        output = [input.select(axis, 0)]
        for i in builtins.range(1, n):
            combined = combine_fn(output[-1], input.select(axis, i))
            output.append(combined)
        return torch.stack(output, dim=axis)


# ─── Type casting ───

def cast(input, dtype):
    """Cast a tensor to a different dtype."""
    from triton._interpreter import _PointerDtype
    if isinstance(dtype, _PointerDtype):
        dtype = dtype.element_ty
    # Unwrap _DtypeWithBitwidth
    if hasattr(dtype, '_dtype'):
        dtype = dtype._dtype
    if isinstance(input, torch.Tensor):
        return input.to(dtype)
    return torch.tensor(input, dtype=dtype)


# ─── Math operations ───

def abs(input):
    return torch.abs(input)


def exp(input):
    if not isinstance(input, torch.Tensor):
        return torch.exp(torch.tensor(input, dtype=torch.float32)).item()
    return torch.exp(input)


def exp2(input):
    if not isinstance(input, torch.Tensor):
        return torch.exp2(torch.tensor(input, dtype=torch.float32)).item()
    return torch.exp2(input)


def log(input):
    if not isinstance(input, torch.Tensor):
        return torch.log(torch.tensor(input, dtype=torch.float32)).item()
    return torch.log(input)


def log2(input):
    if not isinstance(input, torch.Tensor):
        return torch.log2(torch.tensor(input, dtype=torch.float32)).item()
    return torch.log2(input)


def sqrt(input):
    if not isinstance(input, torch.Tensor):
        return torch.sqrt(torch.tensor(input, dtype=torch.float32)).item()
    return torch.sqrt(input)


def rsqrt(input):
    if not isinstance(input, torch.Tensor):
        return torch.rsqrt(torch.tensor(input, dtype=torch.float32)).item()
    return torch.rsqrt(input)


def sin(input):
    if not isinstance(input, torch.Tensor):
        return torch.sin(torch.tensor(input, dtype=torch.float32)).item()
    return torch.sin(input)


def cos(input):
    if not isinstance(input, torch.Tensor):
        return torch.cos(torch.tensor(input, dtype=torch.float32)).item()
    return torch.cos(input)


def sigmoid(input):
    return torch.sigmoid(input)


def floor(input):
    if not isinstance(input, torch.Tensor):
        return torch.floor(torch.tensor(input, dtype=torch.float32)).item()
    return torch.floor(input)


def dot(a, b, acc=None, allow_tf32=True, out_dtype=None):
    """Matrix dot product — accumulate in float32 to match real Triton."""
    result = torch.matmul(a.float(), b.float())
    if acc is not None:
        if isinstance(acc, torch.Tensor):
            result = result + acc.float()
        else:
            result = result + acc
    if out_dtype is not None:
        result = result.to(out_dtype)
    return result


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


# ─── Assertions ───

def static_assert(condition, msg=""):
    """Compile-time assertion (just a runtime assert in interpreter mode)."""
    assert condition, msg


# ─── Atomic operations (simplified — just do the op directly) ───

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


# ─── Utility ───

def cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


def static_range(start, end=None, step=1):
    """Equivalent to range() — in real Triton this unrolls at compile time."""
    # Ensure args are plain ints (may be 0-dim tensors or floats from expressions)
    if isinstance(start, torch.Tensor):
        start = int(start.item())
    elif isinstance(start, float):
        start = int(start)
    if end is not None:
        if isinstance(end, torch.Tensor):
            end = int(end.item())
        elif isinstance(end, float):
            end = int(end)
    if isinstance(step, torch.Tensor):
        step = int(step.item())
    elif isinstance(step, float):
        step = int(step)
    if end is None:
        return builtins.range(start)
    return builtins.range(start, end, step)


# tl.range is the same as static_range in newer Triton versions
range = static_range


def split(tensor):
    """Split a 2D tensor along the last dimension, returning individual columns."""
    if tensor.ndim == 2:
        return tuple(tensor[:, i] for i in builtins.range(tensor.shape[1]))
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
    def exp2(x):
        return torch.exp2(x) if isinstance(x, torch.Tensor) else torch.exp2(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def log(x):
        return torch.log(x) if isinstance(x, torch.Tensor) else torch.log(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def log2(x):
        return torch.log2(x) if isinstance(x, torch.Tensor) else torch.log2(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def sqrt(x):
        return torch.sqrt(x) if isinstance(x, torch.Tensor) else torch.sqrt(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def floor(x):
        return torch.floor(x) if isinstance(x, torch.Tensor) else torch.floor(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def ceil(x):
        return torch.ceil(x) if isinstance(x, torch.Tensor) else torch.ceil(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def erf(x):
        return torch.erf(x) if isinstance(x, torch.Tensor) else torch.erf(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def sin(x):
        return torch.sin(x) if isinstance(x, torch.Tensor) else torch.sin(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def cos(x):
        return torch.cos(x) if isinstance(x, torch.Tensor) else torch.cos(torch.tensor(x, dtype=torch.float32)).item()

    @staticmethod
    def abs(x):
        return torch.abs(x) if isinstance(x, torch.Tensor) else builtins.abs(x)


math = _MathModule()
