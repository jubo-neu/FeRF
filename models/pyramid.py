"""Top-down FFRM following the concatenation equations (23)-(25)."""

import torch
from torch import nn
from torch.nn import functional as F


class FFRM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projections = nn.ModuleList()
        current = channels[-1]
        for skip in reversed(channels[:-1]):
            # Preserve incoming channel count; unspecified in the paper.
            self.projections.append(nn.Conv2d(current, current, 1))
            current += skip
        self.out_channels = current

    def forward(self, features):
        x = features[-1]
        for projection, skip in zip(self.projections, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, projection(x)], 1)
        return x
