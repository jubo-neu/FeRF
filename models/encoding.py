import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, frequencies, include_input=True):
        super().__init__()
        self.register_buffer("bands", math.pi * 2.0 ** torch.arange(frequencies))
        self.include_input = include_input
        self.out_dim = in_dim * (2 * frequencies + int(include_input))

    def forward(self, x):
        encoded = x[..., None] * self.bands
        encoded = torch.cat([encoded.sin(), encoded.cos()], -1).flatten(-2)
        return torch.cat([x, encoded], -1) if self.include_input else encoded
