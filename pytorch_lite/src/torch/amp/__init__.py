"""torch.amp — mixed-precision stubs."""


def custom_fwd(fn=None, *, device_type=None, cast_inputs=None):
    if fn is not None:
        return fn
    return lambda f: f


def custom_bwd(fn=None, *, device_type=None):
    if fn is not None:
        return fn
    return lambda f: f
