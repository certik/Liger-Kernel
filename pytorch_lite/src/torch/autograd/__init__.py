"""torch.autograd — Function base class for custom autograd ops."""

from torch._tensor import Tensor, _GradFn, _wrap


class _Context:
    """Minimal autograd context (the ``ctx`` argument to forward / backward)."""

    def __init__(self):
        self._saved = []

    def save_for_backward(self, *tensors):
        self._saved = list(tensors)

    @property
    def saved_tensors(self):
        return tuple(self._saved)

    # allow arbitrary attributes (e.g. ctx.foo = bar)
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


class Function:
    """
    Minimal stand-in for ``torch.autograd.Function``.

    Subclass and implement ``forward`` / ``backward`` as static methods,
    then call ``SubClass.apply(*args)`` to invoke.
    """

    @classmethod
    def apply(cls, *args):
        ctx = _Context()
        result = cls.forward(ctx, *args)

        # Check if any tensor arg requires grad
        needs_grad = any(isinstance(a, Tensor) and a._requires_grad for a in args)

        if needs_grad and isinstance(result, Tensor):
            result._requires_grad = True
            tensor_args = [a if isinstance(a, Tensor) else None for a in args]

            def backward_fn(grad):
                raw = cls.backward(ctx, grad)
                return raw if isinstance(raw, tuple) else (raw,)

            result._grad_fn = _GradFn(f"{cls.__name__}Backward", backward_fn, tensor_args)

        return result
