import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from neural_pde.transformers.common import Block, LayerNorm


@dataclass 
class TransformerConfig:
    input_dim: int = 1
    output_dim: int = 1
    n_embd: int = 128
    n_head: int = 2
    n_layer: int = 4
    bias: bool = True
    dropout: float = 0.1


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig): 
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.n_embd)
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.dropout),
                blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                head=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.output_proj = nn.Linear(config.n_embd, config.output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer.drop(x)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.head(x)
        x = self.output_proj(x)
        return x

    def no_seq_forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.transformer.drop(x)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.head(x)
        x = self.output_proj(x)
        return x[:, 0, :]


