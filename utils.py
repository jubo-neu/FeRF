import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"model", "render", "data", "train"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"Configuration must include {sorted(required)}")
    return config


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu or install CUDA-enabled PyTorch")
    return device


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def save_rgb(image, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    array = (image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def write_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
