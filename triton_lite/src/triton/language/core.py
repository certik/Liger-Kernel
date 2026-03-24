"""
triton.language.core — Re-exports core symbols so that
`import triton.language.core as core` works.

In real Triton, triton.language.core is the underlying implementation module.
Here we simply re-export everything from triton.language.
"""

from triton.language import (
    # dtypes
    float16,
    float32,
    bfloat16,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    bool_ as bool,
    # constexpr
    constexpr,
    # operations
    arange,
    where,
    static_range,
)

import torch


def reshape(tensor, shape):
    """Reshape a tensor to a given shape."""
    if isinstance(shape, (list, tuple)):
        return tensor.reshape(*shape)
    return tensor.reshape(shape)


def broadcast_to(tensor, shape):
    """Broadcast a tensor to a given shape."""
    if isinstance(shape, (list, tuple)):
        return tensor.broadcast_to(*shape)
    return tensor.broadcast_to(shape)


def static_assert(condition, msg=""):
    """Compile-time assertion (just a runtime assert in interpreter mode)."""
    assert condition, msg
