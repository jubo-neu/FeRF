"""DPCAM, CSSAM, AFM; Eq. (14)-(21) with explicit pooling axes."""

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.shared = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1))

    def forward(self, x):
        # Fig. 5 specifies shared MLP on each pooled descriptor, then sum.
        weights = (self.shared(x.mean((2, 3), keepdim=True)) + self.shared(x.amax((2, 3), keepdim=True))).sigmoid()
        return x * weights


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        return self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)).sigmoid()


class DFFM(nn.Module):
    def __init__(self, global_channels, local_channels):
        super().__init__()
        self.global_channel = ChannelAttention(global_channels)
        self.local_channel = ChannelAttention(local_channels)
        self.global_spatial, self.local_spatial = SpatialAttention(), SpatialAttention()
        self.cross = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, global_feature, local_feature):
        if global_feature.shape[-2:] != local_feature.shape[-2:]:
            raise ValueError("Global/local feature scales must match")
        weights = self.cross(torch.cat([
            self.global_spatial(global_feature), self.local_spatial(local_feature)
        ], 1)).sigmoid()
        return torch.cat([
            self.global_channel(global_feature) * weights,
            self.local_channel(local_feature) * weights,
        ], 1)
