"""torch.cuda — CUDA stubs (always unavailable)."""

import torch.cuda.amp as amp  # noqa: F401


def is_available():
    return False


def get_device_capability(device=None):
    return (0, 0)


def get_device_properties(device=0):
    raise RuntimeError("No CUDA devices available (torch-lite)")


def device_count():
    return 0


def current_device():
    return 0
