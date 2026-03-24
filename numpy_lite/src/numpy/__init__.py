"""numpy-lite: fast-loading NumPy replacement backed by a single C extension."""

from numpy._core import (
    # Types
    ndarray,
    dtype,
    # Dtype singletons
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    float16,
    float32,
    float64,
    # Creation
    array,
    asarray,
    zeros,
    ones,
    full,
    empty,
    empty_like,
    zeros_like,
    ones_like,
    arange,
    linspace,
    stack,
    concatenate,
    # Math
    abs,
    exp,
    log,
    sqrt,
    tanh,
    clip,
    maximum,
    minimum,
    where,
    # Reductions
    sum,
    cumsum,
    max,
    min,
    mean,
    argmax,
    argmin,
    # Predicates
    isnan,
    isinf,
    isposinf,
    isneginf,
    logical_not,
    logical_and,
    logical_or,
    logical_xor,
    allclose,
    array_equal,
    # Manipulation
    copyto,
    ascontiguousarray,
    broadcast_to,
    squeeze,
    expand_dims,
    transpose,
    swapaxes,
    argwhere,
)

import numpy.random  # noqa: F401


def shape(a):
    """Return the shape of an array."""
    return a.shape
