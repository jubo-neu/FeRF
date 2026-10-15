"""A portable scene-per-NPZ format; no implicit dataset camera conventions."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def read_scene(path):
    with np.load(path, allow_pickle=False) as archive:
        images = archive["images"]
        poses = archive["c2w"].astype(np.float32)
        intrinsic = archive["K"].astype(np.float32)
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) < 2:
        raise ValueError(f"{path}: images must be [V,H,W,3], V >= 2")
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    else:
        images = images.astype(np.float32)
    n = len(images)
    if intrinsic.shape == (3, 3):
        intrinsic = np.broadcast_to(intrinsic, (n, 3, 3)).copy()
    if poses.shape != (n, 4, 4) or intrinsic.shape != (n, 3, 3):
        raise ValueError(f"{path}: c2w must be [V,4,4], K [3,3] or [V,3,3]")
    if not all(np.isfinite(a).all() for a in (images, poses, intrinsic)):
        raise ValueError(f"{path}: non-finite scene values")
    if images.min() < 0 or images.max() > 1:
        raise ValueError(f"{path}: floating RGB values must be in [0,1]")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"{path}: invalid homogeneous c2w matrices")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-3) or not np.allclose(np.linalg.det(rotations), 1, atol=2e-3):
        raise ValueError(f"{path}: c2w rotations must be proper orthonormal rotations")
    if np.any(intrinsic[:, [0, 1], [0, 1]] <= 0):
        raise ValueError(f"{path}: positive focal lengths are required")
    return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(intrinsic), torch.from_numpy(poses)


class SceneDataset(Dataset):
    def __init__(self, root, split="train", image_size=64, source_view=0, evaluation=False):
        self.paths = sorted((Path(root) / split).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No NPZ scenes found in {Path(root) / split}")
        self.image_size, self.source_view = image_size, source_view
        self.evaluation = evaluation
        self.pairs = []
        if evaluation:
            for i, path in enumerate(self.paths):
                with np.load(path, allow_pickle=False) as archive:
                    count = len(archive["images"])
                if not 0 <= source_view < count:
                    raise ValueError(f"source_view={source_view} outside scene {path.name}")
                self.pairs.extend((i, target) for target in range(count) if target != source_view)

    def __len__(self):
        return len(self.pairs) if self.evaluation else len(self.paths)

    def __getitem__(self, index):
        if self.evaluation:
            scene, target = self.pairs[index]
            source = self.source_view
        else:
            scene = index
        images, intrinsic, poses = read_scene(self.paths[scene])
        n, _, h, w = images.shape
        if (h, w) != (self.image_size, self.image_size):
            raise ValueError(f"{self.paths[scene]}: expected {self.image_size}x{self.image_size}; convert/resize data first")
        if not self.evaluation:
            source, target = torch.randperm(n)[:2].tolist()
        return {"source": images[source], "target": images[target],
                "source_K": intrinsic[source], "target_K": intrinsic[target],
                "source_c2w": poses[source], "target_c2w": poses[target],
                "scene": self.paths[scene].stem, "source_index": source, "target_index": target}
