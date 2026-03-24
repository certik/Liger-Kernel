"""
Triton kernel interpreter.

Provides the JITFunction class that wraps kernel functions and
the TensorPointer abstraction for simulating Triton pointer semantics.
"""

import threading

import torch

# Thread-local storage for the current program_id during kernel execution
_ctx = threading.local()


def _tensor_to_flat_storage(tensor):
    """Convert a tensor to a 1-D view of its underlying storage.

    For non-contiguous tensors (e.g. channels_last), reshape(-1) creates
    a contiguous copy which breaks stride-based indexing in kernels.
    This function returns the raw storage as a flat tensor instead.
    """
    t = tensor.detach()
    nbytes = t.untyped_storage().size()
    storage_numel = nbytes // t.element_size()
    flat = t.as_strided((storage_numel,), (1,), storage_offset=0)
    return flat, t.storage_offset()


class _PointerDtype:
    """Wraps a torch dtype to provide Triton's .element_ty attribute."""
    def __init__(self, torch_dtype):
        # Unwrap _DtypeWithBitwidth if present
        if hasattr(torch_dtype, '_dtype'):
            torch_dtype = torch_dtype._dtype
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
        self.type = self.dtype  # alias: some kernels use ptr.type.element_ty

    def _resolve_other(self, other):
        """Convert other to an integer offset value."""
        if isinstance(other, torch.Tensor):
            if other.ndim == 0:
                return int(other.item())
            return other  # multi-element tensor
        return int(other)

    def __add__(self, other):
        if isinstance(other, TensorPointer):
            return TensorPointer(self.data, self.offset + other.offset)
        resolved = self._resolve_other(other)
        if isinstance(resolved, torch.Tensor):
            return PointerBlock(self.data, self.offset + resolved)
        return TensorPointer(self.data, self.offset + resolved)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, torch.Tensor):
            if other.ndim == 0:
                return TensorPointer(self.data, self.offset - int(other.item()))
            return PointerBlock(self.data, self.offset - other)
        return TensorPointer(self.data, self.offset - int(other))

    def __rsub__(self, other):
        raise TypeError("Cannot subtract a pointer from a scalar")

    def __iadd__(self, other):
        if isinstance(other, torch.Tensor):
            if other.ndim == 0:
                self.offset += int(other.item())
            else:
                # Mutate into a PointerBlock — return new object
                return PointerBlock(self.data, self.offset + other)
        else:
            self.offset += int(other)
        return self

    def __isub__(self, other):
        if isinstance(other, torch.Tensor):
            if other.ndim == 0:
                self.offset -= int(other.item())
            else:
                return PointerBlock(self.data, self.offset - other)
        else:
            self.offset -= int(other)
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

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return PointerBlock(self.data, self.offsets - int(other))
        if isinstance(other, torch.Tensor):
            return PointerBlock(self.data, self.offsets - other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return PointerBlock(self.data, self.offsets * int(other))
        if isinstance(other, torch.Tensor):
            return PointerBlock(self.data, self.offsets * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)


class JITFunction:
    """
    Wraps a Python function decorated with @triton.jit.

    Calling `kernel[(grid,)](arg1, arg2, ..., BLOCK_SIZE=128)` will:
    1. Convert torch.Tensor arguments to TensorPointer
    2. Loop over the grid, setting the program_id for each iteration
    3. Execute the kernel body as regular Python

    Also supports direct calls (without grid) for helper JIT functions
    that are called from within other kernels.
    """

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __call__(self, *args, **kwargs):
        """Direct call for JIT helper functions invoked from within kernels."""
        return self.fn(*args, **kwargs)

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
        from triton.language import constexpr

        # Separate kernel kwargs (constexpr params) from launch kwargs
        launch_keys = {"num_warps", "num_stages", "num_ctas", "enable_warp_specialization", "grf_mode"}

        # Check which launch keys are actually kernel parameters
        sig = inspect.signature(self.fn)
        param_names = set(sig.parameters.keys())
        params_list = list(sig.parameters.values())
        # If a launch key is also a kernel parameter, keep it as a kernel kwarg
        actual_launch_keys = launch_keys - param_names
        kernel_kwargs = {k: v for k, v in kwargs.items() if k not in actual_launch_keys}

        def _is_constexpr_param(param):
            """Check if a parameter has tl.constexpr annotation."""
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                return False
            return ann is constexpr or (isinstance(ann, type) and ann.__name__ == 'constexpr')

        # Build final args list, converting tensors to pointers
        # and non-constexpr scalar floats to fp32 tensors (for type promotion)
        converted_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                flat, storage_off = _tensor_to_flat_storage(arg)
                converted_args.append(TensorPointer(flat, offset=storage_off))
            elif isinstance(arg, float) and i < len(params_list) and not _is_constexpr_param(params_list[i]):
                converted_args.append(torch.tensor(arg, dtype=torch.float32))
            else:
                converted_args.append(arg)

        # Also convert tensor kwargs
        converted_kwargs = {}
        for k, v in kernel_kwargs.items():
            if isinstance(v, torch.Tensor):
                flat, storage_off = _tensor_to_flat_storage(v)
                converted_kwargs[k] = TensorPointer(flat, offset=storage_off)
            elif isinstance(v, float) and k in sig.parameters and not _is_constexpr_param(sig.parameters[k]):
                converted_kwargs[k] = torch.tensor(v, dtype=torch.float32)
            else:
                converted_kwargs[k] = v

        # Resolve callable grid elements (lambda grids).
        # In real Triton, a grid can be a callable that receives a `meta` dict
        # mapping parameter names to their values and returns a grid tuple.
        resolved_grid = []
        for g in self.grid:
            if callable(g):
                # Build meta dict: map param names → original arg values
                meta = {}
                params = list(sig.parameters.keys())
                for param_name, arg_val in zip(params, args):
                    meta[param_name] = arg_val
                meta.update(kernel_kwargs)
                result = g(meta)
                if isinstance(result, (tuple, list)):
                    resolved_grid.extend(result)
                else:
                    resolved_grid.append(result)
            else:
                resolved_grid.append(g)
        grid = tuple(resolved_grid)

        # Compute total grid size and store grid dimensions
        total = 1
        for g in grid:
            total *= g
        # Pad grid to 3D for num_programs
        padded_grid = list(grid) + [1] * (3 - len(grid))
        _ctx.grid_dims = tuple(padded_grid[:3])

        # Iterate over grid
        for pid in range(total):
            # Set program IDs (support up to 3D grid)
            grid_ids = []
            remaining = pid
            for dim_size in reversed(grid):
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


def get_num_programs(axis):
    """Called by tl.num_programs(axis). Returns grid size for given axis."""
    # _ctx.grid_dims is set by _KernelLauncher before the loop
    return getattr(_ctx, 'grid_dims', (1, 1, 1))[axis]
