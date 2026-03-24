"""triton.runtime — stubs for runtime utilities."""


class _DriverUtils:
    """Stub for triton.runtime.driver.active.utils."""

    def get_device_properties(self, device_id):
        return {"num_vectorcore": 20}


class _ActiveDriver:
    def __init__(self):
        self.utils = _DriverUtils()

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")


class _Driver:
    def __init__(self):
        self.active = _ActiveDriver()


driver = _Driver()
