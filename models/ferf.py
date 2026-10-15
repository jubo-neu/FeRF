"""Connect image-conditioned EGN with a queryable radiance field."""

import torch
from torch import nn
from torch.nn import functional as F
from .swin import NativeSwin, PretrainedSwin
from .local import LFEM
from .attention import DFFM
from .pyramid import FFRM
from .field import RadianceField
from ferf.rendering.camera import project_points, world_to_camera


class FeRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone = config.get("backbone", "native")
        if backbone == "native":
            self.gfem = NativeSwin(config.get("global_dim", 96), config.get("depths", [2, 2, 6, 2]),
                                   config.get("heads", [3, 6, 12, 24]), config.get("window", 4))
        elif backbone == "timm":
            self.gfem = PretrainedSwin(config.get("image_size", 64), config.get("pretrained", True))
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.lfem = LFEM(config.get("local_dim", 32))
        self.dffm = nn.ModuleList([DFFM(g, l) for g, l in zip(self.gfem.channels, self.lfem.channels)])
        self.ffrm = FFRM([g + l for g, l in zip(self.gfem.channels, self.lfem.channels)])
        self.field = RadianceField(self.ffrm.out_channels, **config.get("field", {}))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def encode(self, image):
        if min(image.shape[-2:]) < 32 or any(s % 32 for s in image.shape[-2:]):
            raise ValueError("Input height/width must be multiples of 32, at least 32")
        normalized = (image - self.image_mean) / self.image_std
        g, l = self.gfem(normalized), self.lfem(normalized)
        return self.ffrm([fusion(a, b) for fusion, a, b in zip(self.dffm, g, l)])

    def query(self, feature_map, points, directions, intrinsic, source_c2w, image_size):
        """points/directions [B,R,S,3]; features [B,C,H/4,W/4]."""
        b, r, s, _ = points.shape
        # Projection/division must stay float32 under AMP, especially behind the camera.
        with torch.autocast(device_type=points.device.type, enabled=False):
            camera_points = world_to_camera(points.float(), source_c2w.float())
            grid, valid = project_points(camera_points, intrinsic.float(), image_size)
            features = F.grid_sample(feature_map.float(), grid.reshape(b, r * s, 1, 2),
                                     mode="bilinear", padding_mode="zeros", align_corners=False)
            features = features.squeeze(-1).transpose(1, 2).reshape(b, r, s, -1)
            features = features * valid[..., None]
            camera_dirs = torch.einsum("bij,brsj->brsi", source_c2w[:, :3, :3].float().transpose(1, 2), directions.float())
        return self.field(features, camera_points, F.normalize(camera_dirs, dim=-1))
