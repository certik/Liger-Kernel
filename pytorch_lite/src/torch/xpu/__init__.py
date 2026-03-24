"""torch.xpu — XPU stubs (always unavailable)."""


def is_available():
    return False


def get_device_properties(device=0):
    raise RuntimeError("No XPU devices available (torch-lite)")
