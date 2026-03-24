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
    float64,
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
    float8e5,
    int1,
    # constexpr
    constexpr,
    # operations
    arange,
    where,
    static_range,
    reshape,
    broadcast_to,
    reduce,
    associative_scan,
    static_assert,
)

import torch


def static_assert(condition, msg=""):
    """Compile-time assertion (just a runtime assert in interpreter mode)."""
    assert condition, msg
