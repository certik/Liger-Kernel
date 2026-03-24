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
        from scipy.special import erf  # noqa: F811

        cdf = 0.5 * (1 + erf(x / np.sqrt(2)))
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
    out = (e / np.sum(e, axis=dim, keepdims=True)).astype(input._data.dtype)
    return _wrap(out, input._logical_dtype)


def cross_entropy(input, target, **kwargs):
    raise NotImplementedError("cross_entropy not implemented in torch-lite")
