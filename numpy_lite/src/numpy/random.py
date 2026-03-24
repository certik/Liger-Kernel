"""numpy.random — lightweight random array generation."""
import math
import random as _rng

_np = None

def _core():
    global _np
    if _np is None:
        import numpy._core as c
        _np = c
    return _np


def seed(s):
    _rng.seed(s)


def rand(*shape):
    c = _core()
    n = 1
    for s in shape:
        n *= s
    data = [_rng.random() for _ in range(n)]
    if not shape:
        return c.array(data[0]) if data else c.array(0.0)
    arr = c.array(data)
    return arr.reshape(shape)


def randn(*shape):
    c = _core()
    n = 1
    for s in shape:
        n *= s
    data = []
    for _ in range(0, n, 2):
        u1 = max(_rng.random(), 1e-300)
        u2 = _rng.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        data.append(z0)
        data.append(z1)
    data = data[:n]
    if not shape:
        return c.array(data[0]) if data else c.array(0.0)
    arr = c.array(data)
    return arr.reshape(shape)


def uniform(low=0.0, high=1.0, size=None):
    c = _core()
    if size is None:
        return c.array(low + (high - low) * _rng.random())
    if isinstance(size, int):
        size = (size,)
    n = 1
    for s in size:
        n *= s
    data = [low + (high - low) * _rng.random() for _ in range(n)]
    arr = c.array(data)
    return arr.reshape(size)


def normal(loc=0.0, scale=1.0, size=None):
    if size is None:
        size = (1,)
    elif isinstance(size, int):
        size = (size,)
    arr = randn(*size)
    if scale != 1.0:
        arr = arr * scale
    if loc != 0.0:
        arr = arr + loc
    return arr
