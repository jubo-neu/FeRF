"""Self-contained hierarchical Swin encoder, with masked shifted windows."""

import torch
from torch import nn
from torch.nn import functional as F


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        if dim % heads:
            raise ValueError("Swin channels must be divisible by attention heads")
        self.heads, self.window = heads, window
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0) + window - 1
        index = relative[..., 0] * (2 * window - 1) + relative[..., 1]
        self.register_buffer("relative_index", index)
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x, mask):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q * self.scale) @ k.transpose(-1, -2)
        attention = attention + self.bias[self.relative_index].permute(2, 0, 1)[None]
        if mask is not None:
            nw = mask.shape[0]
            attention = (attention.view(-1, nw, self.heads, n, n) + mask[None, :, None]).flatten(0, 1)
        attention = attention.softmax(dim=-1)
        return self.proj((attention @ v).transpose(1, 2).reshape(b, n, c))


def partition(x, w):
    b, h, width, c = x.shape
    return x.reshape(b, h // w, w, width // w, w, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, c)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window=4, shifted=False):
        super().__init__()
        self.window, self.shifted = window, shifted
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, heads, window)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        b, h, width, c = x.shape
        w = self.window
        shift = w // 2 if self.shifted and min(h, width) > w else 0
        y = F.pad(self.norm1(x), (0, 0, 0, (-width) % w, 0, (-h) % w))
        hp, wp = y.shape[1:3]
        # Mask cyclic wrap-around AND padded key tokens.
        labels = y.new_zeros((1, hp, wp, 1))
        if shift:
            count = 0
            for hs in (slice(0, -w), slice(-w, -shift), slice(-shift, None)):
                for ws in (slice(0, -w), slice(-w, -shift), slice(-shift, None)):
                    labels[:, hs, ws] = count
                    count += 1
        labels = partition(labels, w).squeeze(-1)
        mask = (labels[:, :, None] != labels[:, None, :]).to(x.dtype) * -100.0
        valid = y.new_zeros((1, hp, wp, 1))
        valid[:, :h, :width] = 1
        if shift:
            y = torch.roll(y, (-shift, -shift), (1, 2))
            valid = torch.roll(valid, (-shift, -shift), (1, 2))
        valid = partition(valid, w).squeeze(-1)
        mask = mask + (1 - valid[:, None, :]) * -100.0
        windows = self.attention(partition(y, w), mask)
        y = windows.reshape(b, hp // w, wp // w, w, w, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
        if shift:
            y = torch.roll(y, (shift, shift), (1, 2))
        x = x + y[:, :h, :width]
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm, self.proj = nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        h, w = x.shape[1:3]
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.proj(self.norm(x))


class NativeSwin(nn.Module):
    def __init__(self, dim=96, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), window=4):
        super().__init__()
        if len(depths) != 4 or len(heads) != 4:
            raise ValueError("GFEM needs four stages")
        self.channels = [dim * 2 ** i for i in range(4)]
        self.patch = nn.Conv2d(3, dim, 4, 4)
        self.norm = nn.LayerNorm(dim)
        self.stages = nn.ModuleList([
            nn.Sequential(*[SwinBlock(c, heads[i], window, bool(j % 2)) for j in range(depths[i])])
            for i, c in enumerate(self.channels)
        ])
        self.merges = nn.ModuleList([PatchMerging(c) for c in self.channels[:-1]])

    def forward(self, image):
        x = self.norm(self.patch(image).permute(0, 2, 3, 1))
        features = []
        for i, stage in enumerate(self.stages):
            if i:
                x = self.merges[i - 1](x)
            x = stage(x)
            features.append(x.permute(0, 3, 1, 2).contiguous())
        return features


class PretrainedSwin(nn.Module):
    """Optional timm Swin-T; downloads ImageNet weights only when requested."""

    def __init__(self, image_size=64, pretrained=True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError('Install the pretrained extra: pip install -e ".[pretrained]"') from exc
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3), img_size=image_size,
        )
        self.channels = list(self.backbone.feature_info.channels())

    def forward(self, image):
        # timm's Swin feature API emits NHWC tensors.
        return [x.permute(0, 3, 1, 2).contiguous() for x in self.backbone(image)]
