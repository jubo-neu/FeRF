"""Render target cameras from a supplied trajectory or the scene cameras."""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ferf.checkpoint import model_from_checkpoint
from ferf.data.dataset import read_scene
from ferf.rendering import Renderer
from ferf.utils import resolve_device, save_rgb


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-view", type=int, default=0)
    parser.add_argument("--trajectory", help="NPZ containing c2w [T,4,4], optional K [T,3,3] or [3,3]")
    parser.add_argument("--output", default="runs/render")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gif", action="store_true", help="Also save an animated preview")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = resolve_device(args.device)
    model, state = model_from_checkpoint(args.checkpoint, device)
    images, intrinsic, poses = read_scene(args.scene)
    source = args.source_view
    if not 0 <= source < len(images): raise ValueError("Source view is out of range")
    if images.shape[-2:] != (state["config"]["model"]["image_size"],) * 2:
        raise ValueError("Scene resolution must match the checkpoint")
    target_poses, target_k = poses.numpy(), intrinsic.numpy()
    if args.trajectory:
        with np.load(args.trajectory, allow_pickle=False) as trajectory:
            target_poses = trajectory["c2w"].astype(np.float32)
            target_k = trajectory["K"].astype(np.float32) if "K" in trajectory else intrinsic[source].numpy()
        if target_k.shape == (3, 3): target_k = np.repeat(target_k[None], len(target_poses), 0)
    if target_poses.ndim != 3 or target_poses.shape[1:] != (4, 4) or target_k.shape != (len(target_poses), 3, 3):
        raise ValueError("Invalid trajectory shapes")
    if not np.isfinite(target_poses).all() or not np.isfinite(target_k).all() or not len(target_poses):
        raise ValueError("Trajectory must contain finite camera parameters and at least one view")
    batch = {"source": images[source:source + 1].to(device), "source_K": intrinsic[source:source + 1].to(device),
             "source_c2w": poses[source:source + 1].to(device)}
    features = model.encode(batch["source"])
    renderer = Renderer(**state["config"]["render"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    save_rgb(images[source], output / "source.png")
    frames = []
    for index, (pose, k) in enumerate(zip(target_poses, target_k)):
        batch["target_c2w"] = torch.from_numpy(pose)[None].to(device)
        batch["target_K"] = torch.from_numpy(k)[None].to(device)
        result = renderer.image(model, batch, features)
        filename = output / f"{index:04d}.png"
        save_rgb(result["rgb"][0], filename)
        np.savez_compressed(output / f"{index:04d}_geometry.npz", depth=result["depth"][0, 0].cpu().numpy(),
                            opacity=result["opacity"][0, 0].cpu().numpy())
        if args.gif:
            with Image.open(filename) as frame: frames.append(frame.copy())
    if args.gif:
        frames[0].save(output / "preview.gif", save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(f"Rendered {len(target_poses)} views to {output}")


if __name__ == "__main__":
    main()
