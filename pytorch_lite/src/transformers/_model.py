"""Minimal PreTrainedModel stub."""

import torch.nn as nn


class PreTrainedModel(nn.Module):
    """Lightweight stand-in for transformers.PreTrainedModel."""

    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self.config = config
