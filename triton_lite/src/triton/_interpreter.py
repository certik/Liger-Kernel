"""
Triton kernel interpreter.

Provides the JITFunction class that wraps kernel functions and
the TensorPointer abstraction for simulating Triton pointer semantics.
"""

import threading

import torch

# Thread-local storage for the current program_id during kernel execution
_ctx = threading.local()


class _PointerDtype:
    """Wraps a torch dtype to provide Triton's .element_ty attribute."""
    def __init__(self, torch_dtype):
        self._dtype = torch_dtype
        self.element_ty = torch_dtype  # In Triton, element_ty is the scalar type

    def __eq__(self, other):
        if isinstance(other, _PointerDtype):
            return self._dtype == other._dtype
        return self._dtype == other

    def __hash__(self):
        return hash(self._dtype)

    def __repr__(self):
        return repr(self._dtype)


class TensorPointer:
    """
    Simulates a Triton pointer into a flattened torch tensor.

    Supports pointer arithmetic (add scalar offset or array of offsets)
    so that kernel code like `a += program_id * stride` and
    `tl.load(a + col_offsets)` works unchanged.
    """

    def __init__(self, data, offset=0):
        # data: 1-D torch tensor (the flattened view of the original tensor)
        self.data = data
        self.offset = offset  # scalar base offset
        self.dtype = _PointerDtype(data.dtype)  # expose dtype with .element_ty

    def __add__(self, other):
        if isinstance(other, TensorPointer):
            return TensorPointer(self.data, self.offset + other.offset)
        if isinstance(other, torch.Tensor):
            if other.ndim == 0:
                # Scalar tensor → offset the pointer
                return TensorPointer(self.data, self.offset + int(other.item()))
            # 1-D offset tensor → PointerBlock (array of pointers)
            return PointerBlock(self.data, self.offset + other)
        # Plain scalar
        return TensorPointer(self.data, self.offset + int(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, torch.Tensor):
            if other.ndim == 0:
                self.offset += int(other.item())
            else:
                # 1-D offset tensor: can't iadd, should use __add__ instead
                raise TypeError(
                    "Cannot iadd a multi-element tensor to a pointer; use ptr + tensor"
                )
        else:
            self.offset += int(other)
        return self


class PointerBlock:
    """
    Represents a block of pointers (base tensor + array of absolute offsets).
    Produced by TensorPointer + offset_tensor.  Used by tl.load / tl.store.
    """

    def __init__(self, data, offsets):
        self.data = data  # 1-D torch tensor
        self.offsets = offsets  # torch tensor of integer offsets (any shape)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return PointerBlock(self.data, self.offsets + int(other))
        if isinstance(other, torch.Tensor):
            return PointerBlock(self.data, self.offsets + other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        # Needed for things like pointer_block * 2 (offset scaling)
        if isinstance(other, (int, float)):
            return PointerBlock(self.data, self.offsets * int(other))
        return NotImplemented


class JITFunction:
    """
    Wraps a Python function decorated with @triton.jit.

    Calling `kernel[(grid,)](arg1, arg2, ..., BLOCK_SIZE=128)` will:
    1. Convert torch.Tensor arguments to TensorPointer
    2. Loop over the grid, setting the program_id for each iteration
    3. Execute the kernel body as regular Python
    """

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __getitem__(self, grid):
        """kernel[(grid,)] → returns a launcher callable."""
        if not isinstance(grid, tuple):
            grid = (grid,)
        return _KernelLauncher(self.fn, grid)


class _KernelLauncher:
    """Returned by kernel[grid]; calling it executes the kernel."""

    def __init__(self, fn, grid):
        self.fn = fn
        self.grid = grid

    def __call__(self, *args, **kwargs):
        import inspect

        # Separate kernel kwargs (constexpr params) from launch kwargs
        launch_keys = {"num_warps", "num_stages", "num_ctas", "enable_warp_specialization", "grf_mode"}

        # Check which launch keys are actually kernel parameters
        sig = inspect.signature(self.fn)
        param_names = set(sig.parameters.keys())
        # If a launch key is also a kernel parameter, keep it as a kernel kwarg
        actual_launch_keys = launch_keys - param_names
        kernel_kwargs = {k: v for k, v in kwargs.items() if k not in actual_launch_keys}

        # Build final args list, converting tensors to pointers
        converted_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                converted_args.append(TensorPointer(arg.detach().reshape(-1), offset=0))
            else:
                converted_args.append(arg)

        # Also convert tensor kwargs
        converted_kwargs = {}
        for k, v in kernel_kwargs.items():
            if isinstance(v, torch.Tensor):
                converted_kwargs[k] = TensorPointer(v.detach().reshape(-1), offset=0)
            else:
                converted_kwargs[k] = v

        # Compute total grid size
        import copy

        total = 1
        for g in self.grid:
            if callable(g):
                raise NotImplementedError("Lambda grids not yet supported in triton-lite")
            total *= g

        # Iterate over grid
        for pid in range(total):
            # Set program IDs (support up to 3D grid)
            grid_ids = []
            remaining = pid
            for dim_size in reversed(self.grid):
                grid_ids.append(remaining % dim_size)
                remaining //= dim_size
            grid_ids.reverse()
            # Pad to 3D
            while len(grid_ids) < 3:
                grid_ids.append(0)

            _ctx.program_ids = grid_ids

            # Deep copy pointer args so each program gets fresh pointers
            call_args = []
            for arg in converted_args:
                if isinstance(arg, TensorPointer):
                    call_args.append(TensorPointer(arg.data, arg.offset))
                else:
                    call_args.append(arg)

            call_kwargs = {}
            for k, v in converted_kwargs.items():
                if isinstance(v, TensorPointer):
                    call_kwargs[k] = TensorPointer(v.data, v.offset)
                else:
                    call_kwargs[k] = v

            self.fn(*call_args, **call_kwargs)


def get_program_id(axis):
    """Called by tl.program_id(axis)."""
    return _ctx.program_ids[axis]
