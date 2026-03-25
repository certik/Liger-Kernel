"""torch.nn.functional — activation and linear helpers."""

import numpy as np

from torch._tensor import Tensor, _GradFn, _wrap, _get_data, bfloat16


def gelu(input, approximate="none"):
    x = input._data.astype(np.float32)
    if approximate == "tanh":
        sqrt_2_over_pi = 0.7978845608028654
        cube = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * cube)
        tanh_val = np.tanh(inner)
        out = (0.5 * x * (1 + tanh_val)).astype(input._data.dtype)
        result = _wrap(out, input._logical_dtype)
        if input._requires_grad:
            result._requires_grad = True

            def bw(g):
                gd = g._data.astype(np.float32)
                left = 0.5 * (1 + tanh_val)
                d_inner = sqrt_2_over_pi * (1 + 3 * 0.044715 * x * x)
                right = 0.5 * x * (1 - tanh_val * tanh_val) * d_inner
                return (_wrap((gd * (left + right)).astype(input._data.dtype), input._logical_dtype),)

            result._grad_fn = _GradFn("GELUBackward", bw, [input])
        return result
    else:
        import math as _math
        vfunc = np.vectorize(_math.erf, otypes=[np.float64])
        cdf = 0.5 * (1 + vfunc(x.astype(np.float64) / np.sqrt(2)).astype(x.dtype))
        return _wrap((x * cdf).astype(input._data.dtype), input._logical_dtype)


def silu(input):
    x = input._data.astype(np.float32)
    sig = 1.0 / (1.0 + np.exp(-x))
    out = (x * sig).astype(input._data.dtype)
    result = _wrap(out, input._logical_dtype)
    if input._requires_grad:
        result._requires_grad = True

        def bw(g):
            gd = g._data.astype(np.float32)
            d = sig * (1 + x * (1 - sig))
            return (_wrap((gd * d).astype(input._data.dtype), input._logical_dtype),)

        result._grad_fn = _GradFn("SiLUBackward", bw, [input])
    return result


def linear(input, weight, bias=None):
    out = _wrap(input._data @ weight._data.T, input._logical_dtype)
    if input._requires_grad or weight._requires_grad:
        out._requires_grad = True
        in_copy = input._data.copy()
        w_copy = weight._data.copy()
        in_feat = weight._data.shape[1]
        out_feat = weight._data.shape[0]

        def bw(g):
            gd = g._data
            gi = _wrap(gd @ w_copy, input._logical_dtype)
            gf = gd.reshape(-1, out_feat)
            gw = _wrap(gf.T @ in_copy.reshape(-1, in_feat), input._logical_dtype)
            gb = _wrap(gf.sum(axis=0)) if bias is not None else None
            return (gi, gw, gb)

        out._grad_fn = _GradFn("LinearBackward", bw, [input, weight, bias])
    return out


def relu(input, inplace=False):
    return _wrap(np.maximum(input._data, 0), input._logical_dtype)


def softmax(input, dim=-1, _stacklevel=3, dtype=None):
    x = input._data.astype(np.float64)
    e = np.exp(x - np.max(x, axis=dim, keepdims=True))
    s = e / np.sum(e, axis=dim, keepdims=True)
    out = s.astype(input._data.dtype)
    result = _wrap(out, input._logical_dtype)
    if input._requires_grad:
        result._requires_grad = True
        s_copy = s.copy().astype(input._data.dtype)

        def bw(g):
            gd = g._data
            dot = np.sum(gd * s_copy, axis=dim, keepdims=True)
            grad = (s_copy * (gd - dot)).astype(input._data.dtype)
            return (_wrap(grad, input._logical_dtype),)

        result._grad_fn = _GradFn("SoftmaxBackward", bw, [input])
    return result


def cross_entropy(input, target, **kwargs):
    raise NotImplementedError("cross_entropy not implemented in torch-lite")


def log_sigmoid(input):
    x = input._data.astype(np.float32)
    result = -np.logaddexp(0, -x)
    return _wrap(result.astype(input._data.dtype), input._logical_dtype)

logsigmoid = log_sigmoid


def softplus(input, beta=1, threshold=20):
    x = input._data.astype(np.float32)
    bx = beta * x
    result = np.where(bx > threshold, x, np.log1p(np.exp(bx)) / beta)
    return _wrap(result.astype(input._data.dtype), input._logical_dtype)


def threshold(input, threshold, value, inplace=False):
    data = input._data
    return _wrap(np.where(data > threshold, data, value).astype(data.dtype), input._logical_dtype)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    ndim = len(normalized_shape)
    axes = tuple(range(input._data.ndim - ndim, input._data.ndim))
    x = input._data.astype(np.float32)
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        normed = normed * weight._data.astype(np.float32)
    if bias is not None:
        normed = normed + bias._data.astype(np.float32)
    return _wrap(normed.astype(input._data.dtype), input._logical_dtype)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    idx = input._data if isinstance(input, Tensor) else np.asarray(input, dtype=np.int64)
    return _wrap(weight._data[idx], weight._logical_dtype)


def elu(input, alpha=1.0, inplace=False):
    x = input._data.astype(np.float32)
    result = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return _wrap(result.astype(input._data.dtype), input._logical_dtype)


def celu(input, alpha=1.0, inplace=False):
    x = input._data.astype(np.float32)
    result = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1))
    return _wrap(result.astype(input._data.dtype), input._logical_dtype)


def glu(input, dim=-1):
    a, b = np.split(input._data, 2, axis=dim)
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    sig_b = 1.0 / (1.0 + np.exp(-b_f))
    return _wrap((a_f * sig_b).astype(input._data.dtype), input._logical_dtype)
