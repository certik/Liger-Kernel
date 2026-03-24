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
        return "cpu"

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

    def is_contiguous(self):
        return self._data.flags["C_CONTIGUOUS"]

    def contiguous(self):
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

    def type(self, dtype_str=None):
        if dtype_str is None:
            return f"torch.{self.dtype._name}"
        return self

    # ── shape ops ──

    def view(self, *shape):
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
        return _wrap(np.transpose(self._data, dims), self._logical_dtype)

    def flatten(self, start_dim=0, end_dim=-1):
        return _wrap(self._data.reshape(-1), self._logical_dtype)

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

    def sum(self, dim=None, keepdim=False):
        result = _wrap(np.sum(self._data, axis=dim, keepdims=keepdim), self._logical_dtype)
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

    def mean(self, dim=None, keepdim=False):
        return _wrap(np.mean(self._data, axis=dim, keepdims=keepdim).astype(self._data.dtype), self._logical_dtype)

    def max(self, dim=None, keepdim=False):
        if dim is not None:
            v = np.max(self._data, axis=dim, keepdims=keepdim)
            i = np.argmax(self._data, axis=dim)
            return type("max_result", (), {"values": _wrap(v, self._logical_dtype), "indices": _wrap(i)})()
        return _wrap(np.asarray(np.max(self._data)), self._logical_dtype)

    def min(self, dim=None, keepdim=False):
        if dim is not None:
            v = np.min(self._data, axis=dim, keepdims=keepdim)
            i = np.argmin(self._data, axis=dim)
            return type("min_result", (), {"values": _wrap(v, self._logical_dtype), "indices": _wrap(i)})()
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
        return self.__add__(other)

    def __iadd__(self, other):
        self._data = self._data + _get_data(other)
        return self

    def __sub__(self, other):
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
        od = _get_data(other)
        r = _wrap(od - self._data, self._logical_dtype)
        if self._requires_grad:
            r._requires_grad = True

            def bw(g):
                return (_wrap(-g._data, self._logical_dtype),)

            r._grad_fn = _GradFn("RSubBackward", bw, [self])
        return r

    def __mul__(self, other):
        od = _get_data(other)
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
        return _wrap(self._data ** _get_data(other), self._logical_dtype)

    def __matmul__(self, other):
        od = _get_data(other)
        r = _wrap(self._data @ od, self._logical_dtype)
        if self._needs_grad(other):
            r._requires_grad = True
            ot = other if isinstance(other, Tensor) else None
            sd = self._data.copy()
            od_c = od.copy() if isinstance(od, np.ndarray) else od

            def bw(g):
                gd = g._data
                ga = _wrap(gd @ np.swapaxes(od_c, -2, -1), self._logical_dtype)
                gb = _wrap(np.swapaxes(sd, -2, -1) @ gd, self._logical_dtype) if ot is not None else None
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

    def __float__(self):
        return builtins.float(self._data)

    def __hash__(self):
        return id(self)

    def __format__(self, spec):
        return format(self._data, spec)


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
