"""Minimal tokenization_utils_base stub."""


class BatchEncoding(dict):
    """Lightweight stand-in for transformers.BatchEncoding."""

    def __init__(self, data=None, **kwargs):
        super().__init__(data or {})

    def to(self, device=None):
        return self
