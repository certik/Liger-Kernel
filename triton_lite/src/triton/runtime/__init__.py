"""triton.runtime — stubs for runtime utilities."""


class _DriverUtils:
    """Stub for triton.runtime.driver.active.utils."""

    def get_device_properties(self, device_id):
        return {"num_vectorcore": 20}


class _ActiveDriver:
    def __init__(self):
        self.utils = _DriverUtils()


class _Driver:
    def __init__(self):
        self.active = _ActiveDriver()


driver = _Driver()
