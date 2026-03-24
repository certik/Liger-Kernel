"""torch.cuda.amp — mixed-precision stubs."""


def custom_fwd(fn=None, **kwargs):
    if fn is not None:
        return fn
    return lambda f: f


def custom_bwd(fn=None, **kwargs):
    if fn is not None:
        return fn
    return lambda f: f
