"""OpenCV cameras: +x right, +y down, +z forward; integer pixel centers."""

import torch
from torch.nn import functional as F


def generate_rays(intrinsic, c2w, height, width, indices=None):
    b = intrinsic.shape[0]
    if indices is None:
        indices = torch.arange(height * width, device=intrinsic.device)[None].expand(b, -1)
    pixels = torch.stack([indices % width, indices // width, torch.ones_like(indices)], -1).to(intrinsic.dtype)
    camera_dirs = torch.einsum("bij,brj->bri", torch.linalg.inv(intrinsic), pixels)
    directions = F.normalize(torch.einsum("bij,brj->bri", c2w[:, :3, :3], camera_dirs), dim=-1)
    origins = c2w[:, None, :3, 3].expand_as(directions)
    return origins, directions


def world_to_camera(points, c2w):
    centered = points - c2w[:, None, None, :3, 3]
    return torch.einsum("bij,brsj->brsi", c2w[:, :3, :3].transpose(1, 2), centered)


def project_points(points, intrinsic, image_size):
    h, w = image_size
    projected = torch.einsum("bij,brsj->brsi", intrinsic, points)
    uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
    # align_corners=False: pixel index 0 maps to -1 + 1 / size.
    grid = torch.stack([2 * (uv[..., 0] + 0.5) / w - 1, 2 * (uv[..., 1] + 0.5) / h - 1], -1)
    valid = (points[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1)
    return grid, valid.to(points.dtype)
