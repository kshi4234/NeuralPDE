import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable


STR_TO_ACT = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "softplus": F.softplus,
}


@dataclass
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    activation: Callable = F.relu


class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.activation = config.activation

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        self.body = nn.ModuleList([nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.num_layers - 1)])
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)


    def forward(self, x):
        x = self.input_proj(x)
        x = self.activation(x)
        for layer in self.body:
            x = layer(x)
            x = self.activation(x)
        x = self.output_proj(x)
        return x
