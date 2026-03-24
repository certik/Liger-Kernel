"""
PyTorch-lite: a lightweight shim for HuggingFace transformers.

Only implements enough to run Liger-Kernel tests on macOS/CPU without
requiring the full transformers package.
"""

__version__ = "5.3.0"

from transformers._config import PretrainedConfig
from transformers._model import PreTrainedModel

__all__ = ["__version__", "PretrainedConfig", "PreTrainedModel"]
