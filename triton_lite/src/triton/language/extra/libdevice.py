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
