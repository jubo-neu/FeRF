"""Analytically render colored ellipsoids for an offline pipeline check."""

from pathlib import Path
import numpy as np


def look_at(eye):
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, np.array([0., 1., 0.]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.stack([right, down, forward], -1)
    pose[:3, 3] = eye
    return pose


def make_demo(root, size=64, views=8, scenes=3, seed=42):
    if size < 32 or size % 32 or views < 2 or scenes < 1:
        raise ValueError("size must be a positive multiple of 32; views >= 2; scenes >= 1")
    rng = np.random.default_rng(seed)
    k = np.array([[size * 1.2, 0, (size - 1) / 2], [0, size * 1.2, (size - 1) / 2], [0, 0, 1]], dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    pixels = np.stack([xx, yy, np.ones_like(xx)], -1)
    local = pixels @ np.linalg.inv(k).T
    for split in ("train", "val", "test"):
        directory = Path(root) / split
        directory.mkdir(parents=True, exist_ok=True)
        for scene in range(scenes):
            radius = rng.uniform(0.35, 0.6, 3)
            base = rng.uniform(0.15, 0.85, 3)
            images, poses = [], []
            for theta in np.linspace(0, 2 * np.pi, views, endpoint=False):
                eye = np.array([2 * np.sin(theta), 0.35, 2 * np.cos(theta)])
                pose = look_at(eye)
                direction = local @ pose[:3, :3].T
                direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
                o, d = eye / radius, direction / radius
                a = (d * d).sum(-1)
                b = 2 * (o * d).sum(-1)
                c = (o * o).sum() - 1
                discriminant = b * b - 4 * a * c
                t = (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
                hit = (discriminant >= 0) & (t > 0)
                point = eye + t[..., None] * direction
                normal = point / (radius * radius)
                normal /= np.linalg.norm(normal, axis=-1, keepdims=True).clip(1e-8)
                light = np.array([0.4, 0.8, 0.4]); light /= np.linalg.norm(light)
                shade = 0.4 + 0.6 * np.maximum(normal @ light, 0)
                texture = 0.85 + 0.15 * np.sin(point[..., 0] * 20) * np.cos(point[..., 1] * 18)
                rgb = np.ones((size, size, 3))
                rgb[hit] = (base * shade[..., None] * texture[..., None])[hit]
                images.append((rgb.clip(0, 1) * 255).round().astype(np.uint8)); poses.append(pose)
            np.savez_compressed(directory / f"{split}_{scene:03d}.npz", images=np.stack(images), K=k, c2w=np.stack(poses))
