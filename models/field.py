"""Small feature-conditioned radiance MLP, Eqs. (28)-(29)."""

import torch
from torch import nn
from torch.nn import functional as F
from .encoding import PositionalEncoding


class RadianceField(nn.Module):
    def __init__(self, feature_channels, hidden=128, layers=4, position_frequencies=10,
                 direction_frequencies=4, feature_frequencies=0, feature_dim=128):
        super().__init__()
        self.feature_projection = nn.Linear(feature_channels, feature_dim)
        self.feature_encoding = PositionalEncoding(feature_dim, feature_frequencies)
        self.position_encoding = PositionalEncoding(3, position_frequencies)
        self.direction_encoding = PositionalEncoding(3, direction_frequencies)
        width = self.feature_encoding.out_dim + self.position_encoding.out_dim
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        self.trunk = nn.Sequential(*modules)
        self.density = nn.Linear(hidden, 1)
        self.color = nn.Sequential(nn.Linear(hidden + self.direction_encoding.out_dim, hidden // 2),
                                   nn.ReLU(), nn.Linear(hidden // 2, 3), nn.Sigmoid())

    def forward(self, feature, point, direction):
        feature = self.feature_encoding(self.feature_projection(feature))
        h = self.trunk(torch.cat([feature, self.position_encoding(point)], -1))
        # Density does not depend on direction; positive density via softplus.
        sigma = F.softplus(self.density(h).squeeze(-1) - 1.0)
        color = self.color(torch.cat([h, self.direction_encoding(direction)], -1))
        return color, sigma
