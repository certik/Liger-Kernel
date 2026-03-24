"""torch.nn — Module, Parameter, Linear and common layers."""

import numpy as np

from torch._tensor import (
    Tensor,
    _GradFn,
    _get_data,
    _wrap,
    _to_np,
    bfloat16,
)
import torch.nn.functional as F  # noqa: F401 (re-export)


# ─── Parameter ───────────────────────────────────────────────────────────────

class Parameter(Tensor):
    """A Tensor that is automatically marked as requiring gradients."""

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = _wrap(np.empty(0))
        if isinstance(data, Tensor):
            obj = Tensor.__new__(cls)
            obj._data = data._data
            obj._logical_dtype = data._logical_dtype
            obj._requires_grad = requires_grad
            obj._grad = None
            obj._grad_fn = None
            return obj
        return Tensor.__new__(cls)

    def __init__(self, data=None, requires_grad=True):
        # __new__ already initialised the slots
        pass

    def __repr__(self):
        return f"Parameter containing:\ntensor({self._data})"


# ─── Module ──────────────────────────────────────────────────────────────────

class Module:
    """Minimal stand-in for torch.nn.Module."""

    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            object.__setattr__(self, name, value)
            self.__dict__.setdefault("_parameters", {})[name] = value
        elif isinstance(value, Module):
            object.__setattr__(self, name, value)
            self.__dict__.setdefault("_modules", {})[name] = value
        else:
            object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self, recurse=True):
        for p in self._parameters.values():
            yield p
        if recurse:
            for m in self._modules.values():
                yield from m.parameters(recurse=True)

    def named_parameters(self, prefix="", recurse=True):
        for n, p in self._parameters.items():
            yield (f"{prefix}{n}" if prefix else n), p
        if recurse:
            for mn, m in self._modules.items():
                sub = f"{prefix}{mn}." if prefix else f"{mn}."
                yield from m.named_parameters(prefix=sub, recurse=True)

    def named_modules(self, prefix=""):
        yield prefix, self
        for mn, m in self._modules.items():
            sub = f"{prefix}.{mn}" if prefix else mn
            yield from m.named_modules(prefix=sub)

    def children(self):
        return iter(self._modules.values())

    def modules(self):
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def to(self, *args, **kwargs):
        target_dtype = None
        for a in args:
            from torch._tensor import dtype as _DType

            if isinstance(a, _DType):
                target_dtype = a
        if "dtype" in kwargs:
            target_dtype = kwargs["dtype"]
        if target_dtype is not None:
            for p in self._parameters.values():
                new = p.to(target_dtype)
                p._data = new._data
                p._logical_dtype = new._logical_dtype
                if p._grad is not None:
                    ng = p._grad.to(target_dtype)
                    p._grad._data = ng._data
                    p._grad._logical_dtype = ng._logical_dtype
            for m in self._modules.values():
                m.to(*args, **kwargs)
        return self

    def state_dict(self):
        sd = {}
        for n, p in self.named_parameters():
            sd[n] = p
        return sd

    def train(self, mode=True):
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p._grad = None

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for n, m in self._modules.items():
            lines.append(f"  ({n}): {m}")
        lines.append(")")
        return "\n".join(lines)


# ─── Linear ──────────────────────────────────────────────────────────────────

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = 1.0 / in_features
        sq = np.sqrt(k)
        nd = _to_np(dtype) or np.float32
        self.weight = Parameter(
            _wrap(np.random.uniform(-sq, sq, (out_features, in_features)).astype(nd))
        )
        if bias:
            self.bias = Parameter(_wrap(np.random.uniform(-sq, sq, (out_features,)).astype(nd)))
        else:
            self.bias = None

    def forward(self, input):
        w = self.weight._data
        out_data = input._data @ w.T
        if self.bias is not None:
            out_data = out_data + self.bias._data
        result = _wrap(out_data, input._logical_dtype)

        if input._requires_grad or self.weight._requires_grad:
            result._requires_grad = True
            in_copy = input._data.copy()
            w_copy = w.copy()
            of = w.shape[0]   # out_features from actual weight
            inf = w.shape[1]  # in_features from actual weight
            has_bias = self.bias is not None

            def bw(g):
                gd = g._data
                gi = _wrap(gd @ w_copy, input._logical_dtype)
                gf = gd.reshape(-1, of)
                gw = _wrap(gf.T @ in_copy.reshape(-1, inf), input._logical_dtype)
                gb = _wrap(gf.sum(axis=0), input._logical_dtype) if has_bias else None
                return (gi, gw, gb)

            inputs = [input, self.weight, self.bias]
            result._grad_fn = _GradFn("LinearBackward", bw, inputs)

        return result

    def __repr__(self):
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )


# ─── Re-exports expected at torch.nn.* ──────────────────────────────────────

functional = F


# ─── Stub loss / layer classes (only needed for subclassing at import time) ──

class KLDivLoss(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input, target):
        raise NotImplementedError


class CrossEntropyLoss(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input, target):
        raise NotImplementedError


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, **kwargs):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        if elementwise_affine:
            self.weight = Parameter(_wrap(np.ones(normalized_shape, dtype=np.float32)))
            self.bias = Parameter(_wrap(np.zeros(normalized_shape, dtype=np.float32)))

    def forward(self, input):
        x = input._data.astype(np.float64)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mean) / np.sqrt(var + self.eps)
        out = xhat.astype(input._data.dtype)
        if hasattr(self, "weight"):
            out = out * self.weight._data + self.bias._data
        result = _wrap(out, input._logical_dtype)
        if input._requires_grad:
            result._requires_grad = True
            ns = self.normalized_shape
            n = 1
            for s in ns:
                n *= s
            xhat_f = xhat.astype(input._data.dtype)
            std_inv = (1.0 / np.sqrt(var + self.eps)).astype(input._data.dtype)
            w = self.weight._data.copy() if hasattr(self, "weight") else None

            def bw(g):
                gd = g._data.astype(np.float64)
                if w is not None:
                    gd = gd * w.astype(np.float64)
                dxhat = gd
                dmean = -np.sum(dxhat, axis=-1, keepdims=True) * std_inv.astype(np.float64)
                dvar = np.sum(dxhat * (x - mean), axis=-1, keepdims=True) * (-0.5) * (var + self.eps) ** (-1.5)
                dx = dxhat * std_inv.astype(np.float64) + (2.0 * (x - mean) * dvar + dmean) / n
                return (_wrap(dx.astype(input._data.dtype), input._logical_dtype),)

            inputs = [input]
            if hasattr(self, "weight"):
                inputs.extend([self.weight, self.bias])
            result._grad_fn = _GradFn("LayerNormBackward", bw, inputs)
        return result


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(_wrap(np.random.randn(num_embeddings, embedding_dim).astype(np.float32)))

    def forward(self, input):
        return _wrap(self.weight._data[input._data])


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p

    def forward(self, input):
        return input


class DataParallel(Module):
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


# ─── torch.nn.init stubs ────────────────────────────────────────────────────

class _Init:
    @staticmethod
    def kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
        return tensor

    @staticmethod
    def xavier_uniform_(tensor, gain=1.0):
        return tensor

    @staticmethod
    def zeros_(tensor):
        tensor._data.fill(0)
        return tensor

    @staticmethod
    def ones_(tensor):
        tensor._data.fill(1)
        return tensor

    @staticmethod
    def normal_(tensor, mean=0, std=1):
        tensor._data[:] = np.random.normal(mean, std, tensor._data.shape).astype(tensor._data.dtype)
        return tensor

    @staticmethod
    def constant_(tensor, val):
        tensor._data.fill(val)
        return tensor

init = _Init()

