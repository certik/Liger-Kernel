"""
Dtype definitions and Tensor class for torch-lite.

Provides a numpy-backed Tensor with autograd support — just enough to run
Liger-Kernel's GEGLU tests on CPU without the real PyTorch C++ runtime.
"""

import numpy as np


# ─── Dtype system ────────────────────────────────────────────────────────────

class dtype:
    """Lightweight torch dtype descriptor."""

    def __init__(self, name, numpy_dtype):
        self._name = name
        self._numpy_dtype = numpy_dtype

    def __repr__(self):
        return self._name

    def __eq__(self, other):
        if isinstance(other, dtype):
            return self._name == other._name
        return NotImplemented

    def __hash__(self):
        return hash(self._name)


float16 = dtype("torch.float16", np.float16)
float32 = dtype("torch.float32", np.float32)
float64 = dtype("torch.float64", np.float64)
bfloat16 = dtype("torch.bfloat16", np.float32)  # stored as float32
int8 = dtype("torch.int8", np.int8)
int16 = dtype("torch.int16", np.int16)
int32 = dtype("torch.int32", np.int32)
int64 = dtype("torch.int64", np.int64)
uint8 = dtype("torch.uint8", np.uint8)
uint16 = dtype("torch.uint16", np.uint16)
uint32 = dtype("torch.uint32", np.uint32)
bool = dtype("torch.bool", np.bool_)
complex32 = dtype("torch.complex32", np.complex64)
complex64 = dtype("torch.complex64", np.complex64)
complex128 = dtype("torch.complex128", np.complex128)

_NP_TO_TORCH = {
    np.float16: float16,
    np.float32: float32,
    np.float64: float64,
    np.int8: int8,
    np.int16: int16,
    np.int32: int32,
    np.int64: int64,
    np.uint8: uint8,
    np.uint16: uint16,
    np.uint32: uint32,
    np.bool_: bool,
    np.complex64: complex64,
    np.complex128: complex128,
}


def _to_np(td):
    if td is None:
        return None
    if isinstance(td, dtype):
        return td._numpy_dtype
    return td


def _from_np(nd):
    return _NP_TO_TORCH.get(np.dtype(nd).type, float32)


# ─── Autograd helpers ────────────────────────────────────────────────────────

class _GradFn:
    """A node in the autograd computation graph."""

    __slots__ = ("name", "backward_fn", "inputs")

    def __init__(self, name, backward_fn, inputs):
        self.name = name
        self.backward_fn = backward_fn
        self.inputs = inputs  # list[Tensor | None]

    def __repr__(self):
        return f"<{self.name}>"


def _reduce_grad(grad_data, target_shape):
    """Sum-reduce *grad_data* (ndarray) to *target_shape* (undo broadcasting)."""
    while grad_data.ndim > len(target_shape):
        grad_data = grad_data.sum(axis=0)
    for i, (gs, ts) in enumerate(zip(grad_data.shape, target_shape)):
        if ts == 1 and gs != 1:
            grad_data = grad_data.sum(axis=i, keepdims=True)
    return grad_data


def _run_backward(output, grad_output):
    """Reverse-mode autodiff from *output* with seed *grad_output*."""
    order = []
    visited = set()

    def _visit(t):
        if t is None or t._grad_fn is None:
            return
        fid = id(t._grad_fn)
        if fid in visited:
            return
        visited.add(fid)
        for inp in t._grad_fn.inputs:
            if inp is not None:
                _visit(inp)
        order.append(t)

    _visit(output)
    order.reverse()

    grads = {id(output): grad_output}

    for t in order:
        g = grads.get(id(t))
        if g is None or t._grad_fn is None:
            continue
        input_grads = t._grad_fn.backward_fn(g)
        for inp, ig in zip(t._grad_fn.inputs, input_grads):
            if inp is None or ig is None or not inp._requires_grad:
                continue
            if not isinstance(ig, Tensor):
                ig = _wrap(np.asarray(ig))
            if id(inp) in grads:
                grads[id(inp)] = _wrap(grads[id(inp)]._data + ig._data, inp._logical_dtype)
            else:
                grads[id(inp)] = ig
            # Leaf tensor – accumulate
            if inp._grad_fn is None:
                if inp._grad is None:
                    inp._grad = _wrap(ig._data.copy(), inp._logical_dtype)
                else:
                    inp._grad = _wrap(inp._grad._data + ig._data, inp._logical_dtype)


# ─── Tensor ──────────────────────────────────────────────────────────────────

class _MaxMinResult:
    """Tuple-like result for max/min with dim."""
    __slots__ = ("values", "indices")
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices
    def __iter__(self):
        yield self.values
        yield self.indices
    def __getitem__(self, idx):
        return (self.values, self.indices)[idx]
    def __len__(self):
        return 2


class _Device:
    """Minimal device descriptor."""
    def __init__(self, type_, index=None):
        self.type = type_
        self.index = index

    def __repr__(self):
        return f"device(type='{self.type}')"

    def __eq__(self, other):
        if isinstance(other, _Device):
            return self.type == other.type
        if isinstance(other, str):
            return self.type == other
        return NotImplemented

    def __hash__(self):
        return hash(self.type)

    def __str__(self):
        return self.type


class Tensor:
    """Numpy-backed tensor with lightweight autograd."""

    __slots__ = ("_data", "_logical_dtype", "_requires_grad", "_grad", "_grad_fn")

    def __init__(self, data, dtype_=None, requires_grad=False, _logical_dtype=None):
        if isinstance(data, Tensor):
            data = data._data
        if isinstance(data, np.ndarray):
            self._data = data.astype(_to_np(dtype_), copy=False) if dtype_ is not None else data
        else:
            nd = _to_np(dtype_)
            self._data = np.array(data, dtype=nd) if nd else np.array(data)
        self._logical_dtype = _logical_dtype
        self._requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None

    # ── properties ──

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._logical_dtype if self._logical_dtype is not None else _from_np(self._data.dtype)

    @property
    def device(self):
        return _Device("cpu")

    @property
    def data(self):
        return self

    @data.setter
    def data(self, value):
        if isinstance(value, Tensor):
            self._data = value._data
            self._logical_dtype = value._logical_dtype
        else:
            self._data = np.asarray(value)

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def grad_fn(self):
        return self._grad_fn

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, v):
        self._requires_grad = v

    @property
    def T(self):
        return _wrap(self._data.T, self._logical_dtype)

    @property
    def itemsize(self):
        return self._data.itemsize

    @property
    def is_cuda(self):
        return False

    @property
    def real(self):
        if np.issubdtype(self._data.dtype, np.complexfloating):
            return _wrap(self._data.real.copy())
        return self

    @property
    def imag(self):
        if np.issubdtype(self._data.dtype, np.complexfloating):
            return _wrap(self._data.imag.copy())
        return _wrap(np.zeros_like(self._data))

    # ── mutators / converters ──

    def requires_grad_(self, flag=True):
        self._requires_grad = flag
        return self

    def detach(self):
        return _wrap(self._data, self._logical_dtype)

    def clone(self):
        t = _wrap(self._data.copy(), self._logical_dtype)
        t._requires_grad = self._requires_grad
        return t

    def copy_(self, src):
        """In-place copy from another tensor."""
        if isinstance(src, Tensor):
            np.copyto(self._data, src._data)
        else:
            np.copyto(self._data, np.asarray(src))
        return self

    def masked_fill(self, mask, value):
        """Return a new tensor with positions where *mask* is True replaced by *value*."""
        md = mask._data if isinstance(mask, Tensor) else np.asarray(mask)
        md = np.broadcast_to(md.astype(np.bool_), self._data.shape)
        out = self._data.copy()
        out[md] = value
        result = _wrap(out, self._logical_dtype)
        if self._requires_grad:
            result._requires_grad = True
            mask_copy = md.copy()

            def bw(g):
                gd = g._data.copy()
                gd[mask_copy] = 0.0
                return (_wrap(gd, self._logical_dtype),)

            result._grad_fn = _GradFn("MaskedFillBackward", bw, [self])
        return result

    def masked_fill_(self, mask, value):
        """In-place masked fill."""
        md = mask._data if isinstance(mask, Tensor) else np.asarray(mask)
        md = np.broadcast_to(md.astype(np.bool_), self._data.shape)
        self._data[md] = value
        return self

    def is_contiguous(self, memory_format=None):
        return self._data.flags["C_CONTIGUOUS"]

    def contiguous(self, memory_format=None):
        if self._data.flags["C_CONTIGUOUS"]:
            if self._requires_grad:
                # Return self to preserve autograd chain
                return self
            return self
        result = _wrap(np.ascontiguousarray(self._data), self._logical_dtype)
        if self._requires_grad:
            result._requires_grad = True

            def bw(g):
                return (g,)

            result._grad_fn = _GradFn("ContiguousBackward", bw, [self])
        return result

    def to(self, *args, **kwargs):
        target = None
        for a in args:
            if isinstance(a, dtype):
                target = a
            # str → device name, ignore
        if "dtype" in kwargs:
            target = kwargs["dtype"]
        if target is not None:
            nd = _to_np(target)
            logical = target if target is bfloat16 else None
            t = _wrap(self._data.astype(nd, copy=False), logical)
            t._requires_grad = self._requires_grad
            return t
        return self

    def cpu(self):
        return self

    def float(self):
        return self.to(float32)

    def long(self):
        return self.to(int64)

    def bool(self):
        return _wrap(self._data.astype(np.bool_))

    def int(self):
        return self.to(int32)

    def half(self):
        return self.to(float16)

    def double(self):
        return self.to(float64)

    def type(self, dtype_str=None):
        if dtype_str is None:
            return f"torch.{self.dtype._name}"
        return self

    def clamp(self, min=None, max=None):
        return _wrap(np.clip(self._data, min, max), self._logical_dtype)

    def clamp_(self, min=None, max=None):
        np.clip(self._data, min, max, out=self._data)
        return self

    def ceil(self):
        return _wrap(np.ceil(self._data), self._logical_dtype)

    def floor(self):
        return _wrap(np.floor(self._data), self._logical_dtype)

    def log2(self):
        return _wrap(np.log2(self._data), self._logical_dtype)

    def select(self, dim, index):
        idx = [slice(None)] * self._data.ndim
        idx[dim] = index
        return _wrap(self._data[tuple(idx)], self._logical_dtype)

    def cos(self):
        return _wrap(np.cos(self._data), self._logical_dtype)

    def sin(self):
        return _wrap(np.sin(self._data), self._logical_dtype)

    def exp(self):
        return _wrap(np.exp(self._data), self._logical_dtype)

    def exp2(self):
        return _wrap(np.exp2(self._data), self._logical_dtype)

    def log(self):
        return _wrap(np.log(self._data), self._logical_dtype)

    def sqrt(self):
        return _wrap(np.sqrt(self._data), self._logical_dtype)

    def rsqrt(self):
        return _wrap(1.0 / np.sqrt(self._data.astype(np.float64)).astype(self._data.dtype), self._logical_dtype)

    def sigmoid(self):
        x = self._data.astype(np.float32)
        return _wrap((1.0 / (1.0 + np.exp(-x))).astype(self._data.dtype), self._logical_dtype)

    def tanh(self):
        return _wrap(np.tanh(self._data), self._logical_dtype)

    def erf(self):
        import math as _math
        vfunc = np.vectorize(_math.erf, otypes=[np.float64])
        return _wrap(vfunc(self._data.astype(np.float64)).astype(self._data.dtype), self._logical_dtype)

    def prod(self, dim=None, keepdim=False):
        return _wrap(np.prod(self._data, axis=dim, keepdims=keepdim), self._logical_dtype)

    def var(self, dim=None, unbiased=True, keepdim=False, *, correction=None):
        if correction is not None:
            ddof = correction
        else:
            ddof = 1 if unbiased else 0
        data = self._data
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        return _wrap(np.var(data, axis=dim, ddof=ddof, keepdims=keepdim).astype(self._data.dtype), self._logical_dtype)

    # ── shape ops ──

    def view(self, *shape):
        # Handle view(dtype) for bitcast reinterpretation
        if len(shape) == 1 and isinstance(shape[0], dtype):
            target = _to_np(shape[0])
            return _wrap(self._data.view(target), shape[0] if shape[0] is bfloat16 else None)
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        result = _wrap(self._data.reshape(shape), self._logical_dtype)
        if self._requires_grad:
            result._requires_grad = True
            orig_shape = self._data.shape

            def bw(g):
                return (_wrap(g._data.reshape(orig_shape), self._logical_dtype),)

            result._grad_fn = _GradFn("ViewBackward", bw, [self])
        return result

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        result = _wrap(self._data.reshape(shape), self._logical_dtype)
        if self._requires_grad:
            result._requires_grad = True
            orig_shape = self._data.shape

            def bw(g):
                return (_wrap(g._data.reshape(orig_shape), self._logical_dtype),)

            result._grad_fn = _GradFn("ReshapeBackward", bw, [self])
        return result

    def stride(self, dim=None):
        s = tuple(st // self._data.itemsize for st in self._data.strides)
        return s[dim] if dim is not None else s

    def broadcast_to(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return _wrap(np.broadcast_to(self._data, shape), self._logical_dtype)

    def expand(self, *shape):
        return self.broadcast_to(*shape)

    def squeeze(self, dim=None):
        if isinstance(dim, (list, tuple)):
            result = self._data
            for d in sorted(dim, reverse=True):
                result = np.squeeze(result, axis=d)
            return _wrap(result, self._logical_dtype)
        return _wrap(np.squeeze(self._data, axis=dim), self._logical_dtype)

    def unsqueeze(self, dim):
        return _wrap(np.expand_dims(self._data, axis=dim), self._logical_dtype)

    def transpose(self, dim0, dim1):
        axes = list(range(self._data.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        result = _wrap(np.transpose(self._data, axes), self._logical_dtype)
        if self._requires_grad:
            result._requires_grad = True
            inv_axes = [0] * len(axes)
            for i, a in enumerate(axes):
                inv_axes[a] = i

            def bw(g):
                return (_wrap(np.transpose(g._data, inv_axes), self._logical_dtype),)

            result._grad_fn = _GradFn("TransposeBackward", bw, [self])
        return result

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return _wrap(np.transpose(self._data, dims), self._logical_dtype)

    def flatten(self, start_dim=0, end_dim=-1):
        ndim = self._data.ndim
        if ndim == 0:
            return _wrap(self._data.reshape(1), self._logical_dtype)
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim
        if start_dim == 0 and end_dim == ndim - 1:
            return _wrap(self._data.reshape(-1), self._logical_dtype)
        new_shape = list(self._data.shape[:start_dim])
        new_shape.append(-1)
        new_shape.extend(self._data.shape[end_dim + 1:])
        return _wrap(self._data.reshape(new_shape), self._logical_dtype)

    # ── scalar / reduction ──

    def item(self):
        return self._data.item()

    def numel(self):
        return self._data.size

    def dim(self):
        return self._data.ndim

    def size(self, dim=None):
        return self._data.shape[dim] if dim is not None else self._data.shape

    def any(self):
        return _wrap(np.asarray(self._data.any()))

    def all(self):
        return _wrap(np.asarray(self._data.all()))

    def sum(self, dim=None, keepdim=False, *, dtype=None):
        data = self._data
        # Promote fp16 to fp32 for accumulation
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        result_data = np.sum(data, axis=dim, keepdims=keepdim)
        out_dtype = _to_np(dtype) if dtype else self._data.dtype
        result = _wrap(result_data.astype(out_dtype), self._logical_dtype)
        if self._requires_grad:
            result._requires_grad = True
            orig_shape = self._data.shape

            def bw(g):
                gd = g._data
                if dim is not None and not keepdim:
                    gd = np.expand_dims(gd, axis=dim)
                return (_wrap(np.broadcast_to(gd, orig_shape).copy(), self._logical_dtype),)

            result._grad_fn = _GradFn("SumBackward", bw, [self])
        return result

    def mean(self, dim=None, keepdim=False, *, dtype=None):
        data = self._data
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        out_dtype = _to_np(dtype) if dtype else self._data.dtype
        return _wrap(np.mean(data, axis=dim, keepdims=keepdim).astype(out_dtype), self._logical_dtype)

    def cumsum(self, dim, *, dtype=None):
        data = self._data
        if data.dtype == np.float16:
            result = np.cumsum(data.astype(np.float32), axis=dim).astype(np.float16)
        else:
            result = np.cumsum(data, axis=dim)
        if dtype is not None:
            result = result.astype(_to_np(dtype))
        return _wrap(result, self._logical_dtype)

    def max(self, dim=None, keepdim=False):
        if dim is not None:
            v = np.max(self._data, axis=dim, keepdims=keepdim)
            i = np.argmax(self._data, axis=dim)
            return _MaxMinResult(_wrap(v, self._logical_dtype), _wrap(i.astype(np.int64)))
        return _wrap(np.asarray(np.max(self._data)), self._logical_dtype)

    def min(self, dim=None, keepdim=False):
        if dim is not None:
            v = np.min(self._data, axis=dim, keepdims=keepdim)
            i = np.argmin(self._data, axis=dim)
            return _MaxMinResult(_wrap(v, self._logical_dtype), _wrap(i.astype(np.int64)))
        return _wrap(np.asarray(np.min(self._data)), self._logical_dtype)

    def abs(self):
        return _wrap(np.abs(self._data), self._logical_dtype)

    def backward(self, gradient=None, retain_graph=False):
        if gradient is None:
            gradient = _wrap(np.ones_like(self._data))
        _run_backward(self, gradient)

    # ── arithmetic (with autograd) ──

    def _needs_grad(self, other):
        return self._requires_grad or (isinstance(other, Tensor) and other._requires_grad)

    def __add__(self, other):
        if not isinstance(other, (Tensor, int, float, np.ndarray, np.integer, np.floating)):
            return NotImplemented
        od = _get_data(other)
        r = _wrap(self._data + od, self._logical_dtype)
        if self._needs_grad(other):
            r._requires_grad = True
            ot = other if isinstance(other, Tensor) else None
            ss, os = self._data.shape, np.shape(od)

            def bw(g):
                gd = g._data
                ga = _wrap(_reduce_grad(gd, ss), self._logical_dtype)
                gb = _wrap(_reduce_grad(gd, os), self._logical_dtype) if ot is not None else None
                return (ga, gb)

            r._grad_fn = _GradFn("AddBackward", bw, [self, ot])
        return r

    def __radd__(self, other):
        if not isinstance(other, (Tensor, int, float, np.ndarray, np.integer, np.floating)):
            return NotImplemented
        return self.__add__(other)

    def __iadd__(self, other):
        if not isinstance(other, (Tensor, int, float, np.ndarray, np.integer, np.floating)):
            return NotImplemented
        self._data = self._data + _get_data(other)
        return self

    def add_(self, other, *, alpha=1):
        od = _get_data(other)
        self._data = self._data + alpha * od
        return self

    def mul_(self, other):
        self._data = self._data * _get_data(other)
        return self

    def div_(self, other):
        self._data = self._data / _get_data(other)
        return self

    def __sub__(self, other):
        if not isinstance(other, (Tensor, int, float, np.ndarray, np.integer, np.floating)):
            return NotImplemented
        od = _get_data(other)
        r = _wrap(self._data - od, self._logical_dtype)
        if self._needs_grad(other):
            r._requires_grad = True
            ot = other if isinstance(other, Tensor) else None
            ss, os = self._data.shape, np.shape(od)

            def bw(g):
                gd = g._data
                ga = _wrap(_reduce_grad(gd, ss), self._logical_dtype)
                gb = _wrap(_reduce_grad(-gd, os), self._logical_dtype) if ot is not None else None
                return (ga, gb)

            r._grad_fn = _GradFn("SubBackward", bw, [self, ot])
        return r

    def __rsub__(self, other):
        if not isinstance(other, (Tensor, int, float, np.ndarray, np.integer, np.floating)):
            return NotImplemented
        od = _get_data(other)
        r = _wrap(od - self._data, self._logical_dtype)
        if self._requires_grad:
            r._requires_grad = True

            def bw(g):
                return (_wrap(-g._data, self._logical_dtype),)

            r._grad_fn = _GradFn("RSubBackward", bw, [self])
        return r

    def __mul__(self, other):
        if not isinstance(other, (Tensor, int, float, np.ndarray, np.integer, np.floating)):
            return NotImplemented
        od = _get_data(other)
        # For fp16, compute in fp32 to avoid overflow (matches PyTorch behavior)
        if self._data.dtype == np.float16 and not isinstance(od, np.ndarray):
            r = _wrap((self._data.astype(np.float32) * od).astype(np.float16), self._logical_dtype)
        elif self._data.dtype == np.float16 and isinstance(od, np.ndarray) and od.dtype != np.float16:
            r = _wrap((self._data.astype(np.float32) * od.astype(np.float32)).astype(np.float16), self._logical_dtype)
        else:
            r = _wrap(self._data * od, self._logical_dtype)
        if self._needs_grad(other):
            r._requires_grad = True
            ot = other if isinstance(other, Tensor) else None
            sd, od_c = self._data.copy(), np.array(od, copy=True) if isinstance(od, np.ndarray) else od
            ss, os = self._data.shape, np.shape(od)

            def bw(g):
                gd = g._data
                ga = _wrap(_reduce_grad(gd * od_c, ss), self._logical_dtype)
                gb = _wrap(_reduce_grad(gd * sd, os), self._logical_dtype) if ot is not None else None
                return (ga, gb)

            r._grad_fn = _GradFn("MulBackward", bw, [self, ot])
        return r

    def __rmul__(self, other):
        if not isinstance(other, (Tensor, int, float, np.ndarray, np.integer, np.floating)):
            return NotImplemented
        return self.__mul__(other)

    def __truediv__(self, other):
        return _wrap(self._data / _get_data(other), self._logical_dtype)

    def __rtruediv__(self, other):
        return _wrap(_get_data(other) / self._data, self._logical_dtype)

    def __floordiv__(self, other):
        return _wrap(self._data // _get_data(other), self._logical_dtype)

    def __rfloordiv__(self, other):
        return _wrap(_get_data(other) // self._data, self._logical_dtype)

    def __mod__(self, other):
        return _wrap(self._data % _get_data(other), self._logical_dtype)

    def __rmod__(self, other):
        return _wrap(_get_data(other) % self._data, self._logical_dtype)

    def __neg__(self):
        return _wrap(-self._data, self._logical_dtype)

    def __pow__(self, other):
        data = self._data
        exp = _get_data(other)
        if data.dtype == np.float16:
            return _wrap(np.power(data.astype(np.float32), exp).astype(np.float16), self._logical_dtype)
        return _wrap(data ** exp, self._logical_dtype)

    def __matmul__(self, other):
        od = _get_data(other)
        sd = self._data
        # Promote to higher precision for matmul accuracy
        if sd.dtype == np.float16:
            r = _wrap((sd.astype(np.float32) @ od.astype(np.float32)).astype(np.float16), self._logical_dtype)
        elif sd.dtype == np.float32:
            r = _wrap((sd.astype(np.float64) @ od.astype(np.float64)).astype(np.float32), self._logical_dtype)
        else:
            r = _wrap(sd @ od, self._logical_dtype)
        if self._needs_grad(other):
            r._requires_grad = True
            ot = other if isinstance(other, Tensor) else None
            sd_c = sd.copy()
            od_c = od.copy() if isinstance(od, np.ndarray) else od

            def bw(g):
                gd = g._data
                ga = _wrap(gd @ np.swapaxes(od_c, -2, -1), self._logical_dtype)
                gb = _wrap(np.swapaxes(sd_c, -2, -1) @ gd, self._logical_dtype) if ot is not None else None
                return (ga, gb)

            r._grad_fn = _GradFn("MatmulBackward", bw, [self, ot])
        return r

    # ── comparisons ──

    def __lt__(self, other):
        return _wrap(self._data < _get_data(other))

    def __le__(self, other):
        return _wrap(self._data <= _get_data(other))

    def __gt__(self, other):
        return _wrap(self._data > _get_data(other))

    def __ge__(self, other):
        return _wrap(self._data >= _get_data(other))

    def __eq__(self, other):
        if isinstance(other, dtype):
            return self.dtype == other
        return _wrap(self._data == _get_data(other))

    def __ne__(self, other):
        if isinstance(other, dtype):
            return self.dtype != other
        return _wrap(self._data != _get_data(other))

    def __and__(self, other):
        return _wrap(self._data & _get_data(other))

    def __rand__(self, other):
        return _wrap(_get_data(other) & self._data)

    def __or__(self, other):
        return _wrap(self._data | _get_data(other))

    def __ror__(self, other):
        return _wrap(_get_data(other) | self._data)

    def __invert__(self):
        return _wrap(~self._data)

    def __xor__(self, other):
        return _wrap(self._data ^ _get_data(other))

    def __rxor__(self, other):
        return _wrap(_get_data(other) ^ self._data)

    def __lshift__(self, other):
        return _wrap(self._data << _get_data(other))

    def __rshift__(self, other):
        return _wrap(self._data >> _get_data(other))

    # ── indexing ──

    def __getitem__(self, key):
        key = _index_key(key)
        result = self._data[key]
        if isinstance(result, np.ndarray):
            return _wrap(result, self._logical_dtype)
        return _wrap(np.asarray(result), self._logical_dtype)

    def __setitem__(self, key, value):
        key = _index_key(key)
        self._data[key] = value._data if isinstance(value, Tensor) else value

    # ── misc ──

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"tensor({self._data})"

    def __bool__(self):
        return builtins.bool(self._data)

    def __int__(self):
        return builtins.int(self._data)

    def __index__(self):
        return builtins.int(self._data)

    def __float__(self):
        return builtins.float(self._data)

    def __hash__(self):
        return id(self)

    def __format__(self, spec):
        return format(self._data, spec)

    def __iter__(self):
        for i in range(len(self._data)):
            yield _wrap(self._data[i], self._logical_dtype)

    # ── additional methods ──

    def t(self):
        """Transpose for 2D tensors (swap dims 0 and 1)."""
        if self._data.ndim < 2:
            return _wrap(self._data, self._logical_dtype)
        return _wrap(self._data.T, self._logical_dtype)

    def pow(self, exponent):
        exp = _get_data(exponent)
        data = self._data
        if data.dtype == np.float16:
            return _wrap(np.power(data.astype(np.float32), exp).astype(np.float16), self._logical_dtype)
        return _wrap(np.power(data, exp), self._logical_dtype)

    def neg(self):
        return _wrap(-self._data, self._logical_dtype)

    def conj(self):
        return _wrap(np.conj(self._data), self._logical_dtype)

    @property
    def is_quantized(self):
        return False

    @property
    def is_sparse(self):
        return False

    def double(self):
        return self.to(float64)

    def log(self):
        return _wrap(np.log(self._data), self._logical_dtype)

    def exp(self):
        return _wrap(np.exp(self._data), self._logical_dtype)

    def sqrt(self):
        return _wrap(np.sqrt(self._data), self._logical_dtype)

    def rsqrt(self):
        return _wrap(1.0 / np.sqrt(self._data.astype(np.float64)).astype(self._data.dtype), self._logical_dtype)

    def cos(self):
        return _wrap(np.cos(self._data), self._logical_dtype)

    def sin(self):
        return _wrap(np.sin(self._data), self._logical_dtype)

    def tanh(self):
        return _wrap(np.tanh(self._data), self._logical_dtype)

    def clamp(self, min=None, max=None):
        return _wrap(np.clip(self._data, min, max), self._logical_dtype)

    def clamp_(self, min=None, max=None):
        np.clip(self._data, min, max, out=self._data)
        return self

    def sign(self):
        return _wrap(np.sign(self._data), self._logical_dtype)

    def reciprocal(self):
        return _wrap(1.0 / self._data, self._logical_dtype)

    def argmax(self, dim=None, keepdim=False):
        return _wrap(np.argmax(self._data, axis=dim).astype(np.int64))

    def argmin(self, dim=None, keepdim=False):
        return _wrap(np.argmin(self._data, axis=dim).astype(np.int64))

    def tolist(self):
        return self._data.tolist()

    def numpy(self):
        return self._data.copy()

    def new_zeros(self, *shape, dtype=None, device=None):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        nd = _to_np(dtype) if dtype else self._data.dtype
        return _wrap(np.zeros(shape, dtype=nd))

    def new_ones(self, *shape, dtype=None, device=None):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        nd = _to_np(dtype) if dtype else self._data.dtype
        return _wrap(np.ones(shape, dtype=nd))

    def new_empty(self, *shape, dtype=None, device=None):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        nd = _to_np(dtype) if dtype else self._data.dtype
        return _wrap(np.empty(shape, dtype=nd))

    def new_full(self, shape, fill_value, dtype=None, device=None):
        if isinstance(shape, int):
            shape = (shape,)
        nd = _to_np(dtype) if dtype else self._data.dtype
        return _wrap(np.full(shape, fill_value, dtype=nd))

    def index_select(self, dim, index):
        idx = index._data if isinstance(index, Tensor) else np.asarray(index)
        return _wrap(np.take(self._data, idx, axis=dim), self._logical_dtype)

    def narrow(self, dim, start, length):
        slices = [slice(None)] * self._data.ndim
        slices[dim] = slice(start, start + length)
        return _wrap(self._data[tuple(slices)].copy(), self._logical_dtype)

    def chunk(self, chunks, dim=0):
        splits = np.array_split(self._data, chunks, axis=dim)
        return tuple(_wrap(s, self._logical_dtype) for s in splits)

    def flip(self, dims):
        if isinstance(dims, int):
            dims = (dims,)
        result = self._data
        for d in dims:
            result = np.flip(result, axis=d)
        return _wrap(result.copy(), self._logical_dtype)

    def topk(self, k, dim=-1, largest=True, sorted=True):
        if dim < 0:
            dim = self._data.ndim + dim
        if largest:
            idx = np.argsort(-self._data, axis=dim)
        else:
            idx = np.argsort(self._data, axis=dim)
        slices = [slice(None)] * self._data.ndim
        slices[dim] = slice(0, k)
        idx_k = idx[tuple(slices)]
        vals = np.take_along_axis(self._data, idx_k, axis=dim)
        return _MaxMinResult(_wrap(vals, self._logical_dtype), _wrap(idx_k.astype(np.int64)))

    def sort(self, dim=-1, descending=False, stable=False):
        if descending:
            idx = np.argsort(-self._data, axis=dim)
        else:
            idx = np.argsort(self._data, axis=dim)
        vals = np.take_along_axis(self._data, idx, axis=dim)
        return _MaxMinResult(_wrap(vals, self._logical_dtype), _wrap(idx.astype(np.int64)))

    def resolve_conj(self):
        return self

    def resolve_neg(self):
        return self

    def fill_(self, value):
        self._data.fill(value)
        return self

    def zero_(self):
        self._data.fill(0)
        return self

    def sub_(self, other, *, alpha=1):
        od = _get_data(other)
        self._data = self._data - alpha * od
        return self

    def tril_(self, diagonal=0):
        self._data[:] = np.tril(self._data, k=diagonal)
        return self

    def triu_(self, diagonal=0):
        self._data[:] = np.triu(self._data, k=diagonal)
        return self

    def scatter_(self, dim, index, src):
        idx = index._data if isinstance(index, Tensor) else np.asarray(index, dtype=np.int64)
        sd = src._data if isinstance(src, Tensor) else np.full_like(self._data, src) if not isinstance(src, np.ndarray) else src
        np.put_along_axis(self._data, idx, sd, axis=dim)
        return self

    def gather(self, dim, index):
        idx = index._data if isinstance(index, Tensor) else np.asarray(index, dtype=np.int64)
        return _wrap(np.take_along_axis(self._data, idx, axis=dim), self._logical_dtype)

    def repeat(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            sizes = tuple(sizes[0])
        return _wrap(np.tile(self._data, sizes), self._logical_dtype)

    def repeat_interleave(self, repeats, dim=None, *, output_size=None):
        if isinstance(repeats, Tensor):
            repeats = repeats._data
        return _wrap(np.repeat(self._data, repeats, axis=dim), self._logical_dtype)

    def storage_offset(self):
        base = self._data.base
        if base is None:
            return 0
        # byte offset from base start to view start, in elements
        return (self._data.ctypes.data - base.ctypes.data) // self._data.itemsize

    def data_ptr(self):
        return self._data.ctypes.data

    def element_size(self):
        return self._data.itemsize

    def is_floating_point(self):
        return self._data.dtype.kind == 'f'

    def is_complex(self):
        return self._data.dtype.kind == 'c'

    def is_conj(self):
        return False

    def is_neg(self):
        return False

    def view_as(self, other):
        return self.view(*other.shape)

    def expand_as(self, other):
        return self.expand(*other.shape)

    def movedim(self, source, destination):
        return _wrap(np.moveaxis(self._data, source, destination), self._logical_dtype)

    def resize_(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        self._data = np.resize(self._data, shape)
        return self

    def outer(self, other):
        return _wrap(np.outer(self._data.ravel(), other._data.ravel()), self._logical_dtype)

    def untyped_storage(self):
        """Return a storage-like object with a size() method."""
        # For broadcast/view arrays, return the base storage size
        base = self._data if self._data.base is None else self._data.base
        nbytes = base.nbytes
        return type("UntypedStorage", (), {"size": lambda self_=None, nb=nbytes: nb})()

    def as_strided(self, size, stride, storage_offset=0):
        # Get the actual base storage
        base = self._data if self._data.base is None else self._data.base
        flat = base.ravel()
        return _wrap(
            np.lib.stride_tricks.as_strided(
                flat[storage_offset:] if storage_offset > 0 else flat,
                shape=size,
                strides=tuple(s * self._data.itemsize for s in stride),
            ),
            self._logical_dtype,
        )


# ─── Fast helpers ─────────────────────────────────────────────────────────────

import builtins  # noqa: E402  (after Tensor so __bool__ can reference it)


def _wrap(data, logical_dtype=None):
    t = Tensor.__new__(Tensor)
    t._data = data if isinstance(data, np.ndarray) else np.asarray(data)
    t._logical_dtype = logical_dtype
    t._requires_grad = False
    t._grad = None
    t._grad_fn = None
    return t


def _get_data(x):
    if isinstance(x, Tensor):
        return x._data
    return x


def _index_key(key):
    if isinstance(key, Tensor):
        return key._data
    if isinstance(key, tuple):
        return tuple(k._data if isinstance(k, Tensor) else k for k in key)
    return key
