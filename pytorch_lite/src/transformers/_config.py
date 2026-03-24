"""Minimal PretrainedConfig stub."""


class PretrainedConfig:
    """Lightweight stand-in for transformers.PretrainedConfig."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        return self.__dict__.copy()
