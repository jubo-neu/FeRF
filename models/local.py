"""LFEM: five residual convolution blocks, Eqs. (7)-(10)."""

from torch import nn


class LocalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.release = nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool2d(2))

    def forward(self, x):
        return self.release(self.main(x) + self.shortcut(x))


class LFEM(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.channels = [dim, 2 * dim, 4 * dim, 8 * dim]
        dims = [3, max(dim // 2, 4), *self.channels]
        self.blocks = nn.ModuleList([LocalBlock(a, b) for a, b in zip(dims, dims[1:])])

    def forward(self, x):
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= 1:
                outputs.append(x)
        return outputs
