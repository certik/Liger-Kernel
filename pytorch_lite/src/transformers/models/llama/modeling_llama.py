"""Minimal LlamaMLP for Liger-Kernel tests."""

import functools
import math

import torch
import torch.nn as nn


class _GELUTanh(nn.Module):
    """tanh-approximation GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""

    def forward(self, x):
        return nn.functional.gelu(x, approximate="tanh")


class _SiLU(nn.Module):
    def forward(self, x):
        return nn.functional.silu(x)


ACT2FN = {
    "gelu_pytorch_tanh": _GELUTanh(),
    "silu": _SiLU(),
}


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
