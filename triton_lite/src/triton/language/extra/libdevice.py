"""libdevice math functions — implemented via torch."""

import torch


def tanh(x):
    if isinstance(x, torch.Tensor):
        return torch.tanh(x)
    return torch.tanh(torch.tensor(x)).item()


def rsqrt(x):
    if isinstance(x, torch.Tensor):
        return torch.rsqrt(x)
    return torch.rsqrt(torch.tensor(x, dtype=torch.float32)).item()


def sqrt(x):
    if isinstance(x, torch.Tensor):
        return torch.sqrt(x)
    return torch.sqrt(torch.tensor(x, dtype=torch.float32)).item()


def exp(x):
    if isinstance(x, torch.Tensor):
        return torch.exp(x)
    return torch.exp(torch.tensor(x, dtype=torch.float32)).item()


def log(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x)
    return torch.log(torch.tensor(x, dtype=torch.float32)).item()


def abs(x):
    if isinstance(x, torch.Tensor):
        return torch.abs(x)
    return torch.abs(torch.tensor(x)).item()


def atan2(y, x):
    if isinstance(y, torch.Tensor) and isinstance(x, torch.Tensor):
        return torch.atan2(y, x)
    y_t = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    result = torch.atan2(y_t, x_t)
    if isinstance(y, torch.Tensor) or isinstance(x, torch.Tensor):
        return result
    return result.item()


def floor(x):
    if isinstance(x, torch.Tensor):
        return torch.floor(x)
    return torch.floor(torch.tensor(x, dtype=torch.float32)).item()


def ceil(x):
    if isinstance(x, torch.Tensor):
        return torch.ceil(x)
    return torch.ceil(torch.tensor(x, dtype=torch.float32)).item()


def erf(x):
    if isinstance(x, torch.Tensor):
        return torch.erf(x)
    return torch.erf(torch.tensor(x, dtype=torch.float32)).item()


def exp2(x):
    if isinstance(x, torch.Tensor):
        return torch.exp2(x)
    return torch.exp2(torch.tensor(x, dtype=torch.float32)).item()


def log2(x):
    if isinstance(x, torch.Tensor):
        return torch.log2(x)
    return torch.log2(torch.tensor(x, dtype=torch.float32)).item()


def sin(x):
    if isinstance(x, torch.Tensor):
        return torch.sin(x)
    return torch.sin(torch.tensor(x, dtype=torch.float32)).item()


def cos(x):
    if isinstance(x, torch.Tensor):
        return torch.cos(x)
    return torch.cos(torch.tensor(x, dtype=torch.float32)).item()
