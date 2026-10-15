import torch
from .sampling import stratified_samples, sample_pdf
from .volume import composite
from .camera import generate_rays


class Renderer:
    def __init__(self, near=0.8, far=3.5, coarse_samples=64, fine_samples=128,
                 ray_chunk=512, query_chunk=32768, white_background=True):
        if not 0 < near < far or coarse_samples < 3 or fine_samples < 0:
            raise ValueError("Require 0 < near < far, coarse_samples >= 3, fine_samples >= 0")
        if ray_chunk < 1 or query_chunk < 1:
            raise ValueError("Chunk sizes must be positive")
        self.near, self.far = near, far
        self.coarse_samples, self.fine_samples = coarse_samples, fine_samples
        self.ray_chunk, self.query_chunk = ray_chunk, query_chunk
        self.white_background = white_background

    def _pass(self, model, feature_map, origins, directions, depths, batch):
        b, r, s = depths.shape
        points = origins[..., None, :] + directions[..., None, :] * depths[..., None]
        dirs = directions[..., None, :].expand_as(points)
        flat_points, flat_dirs = points.reshape(b, -1, 1, 3), dirs.reshape(b, -1, 1, 3)
        colors, densities = [], []
        for start in range(0, r * s, self.query_chunk):
            color, sigma = model.query(feature_map, flat_points[:, start:start + self.query_chunk],
                                       flat_dirs[:, start:start + self.query_chunk], batch["source_K"],
                                       batch["source_c2w"], batch["source"].shape[-2:])
            colors.append(color)
            densities.append(sigma)
        color = torch.cat(colors, 1).reshape(b, r, s, 3)
        sigma = torch.cat(densities, 1).reshape(b, r, s)
        return composite(color, sigma, depths, self.far, self.white_background)

    def rays(self, model, feature_map, origins, directions, batch, perturb=False):
        depths = stratified_samples(origins, self.near, self.far, self.coarse_samples, perturb)
        coarse = self._pass(model, feature_map, origins, directions, depths, batch)
        if not self.fine_samples:
            return {**coarse, "coarse_rgb": coarse["rgb"]}
        mid = (depths[..., 1:] + depths[..., :-1]) * 0.5
        fine = sample_pdf(mid, coarse["weights"][..., 1:-1], self.fine_samples, deterministic=not perturb)
        depths = torch.sort(torch.cat([depths, fine.detach()], -1), -1).values
        output = self._pass(model, feature_map, origins, directions, depths, batch)
        return {**output, "coarse_rgb": coarse["rgb"]}

    @torch.no_grad()
    def image(self, model, batch, feature_map=None):
        h, w = batch["source"].shape[-2:]
        if feature_map is None:
            feature_map = model.encode(batch["source"])
        origins, directions = generate_rays(batch["target_K"], batch["target_c2w"], h, w)
        outputs = {k: [] for k in ("rgb", "depth", "opacity")}
        for start in range(0, h * w, self.ray_chunk):
            out = self.rays(model, feature_map, origins[:, start:start + self.ray_chunk],
                            directions[:, start:start + self.ray_chunk], batch)
            for key in outputs:
                outputs[key].append(out[key])
        return {k: torch.cat(v, 1).reshape(-1, h, w, 3 if k == "rgb" else 1).permute(0, 3, 1, 2)
                for k, v in outputs.items()}
