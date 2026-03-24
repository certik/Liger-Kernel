"""torch.nn.modules stubs."""

from torch.nn import Module, Linear, Parameter  # noqa: F401


def _pair(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def _single(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x,)


def _triple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x, x)
