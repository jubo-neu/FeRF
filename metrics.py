"""Full-image PSNR, local-window SSIM, optional learned LPIPS."""

import torch
from torch.nn import functional as F


def psnr(prediction, target):
    mse = (prediction - target).square().mean(dim=(-3, -2, -1))
    return -10 * torch.log10(mse.clamp_min(1e-12))


def ssim(prediction, target):
    coords = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
    kernel = torch.exp(-coords.square() / (2 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, 11, 11)
    def smooth(x):
        return F.conv2d(x, kernel, groups=3)
    mu_x, mu_y = smooth(prediction), smooth(target)
    var_x, var_y = smooth(prediction.square()) - mu_x.square(), smooth(target.square()) - mu_y.square()
    cov = smooth(prediction * target) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / (
        (mu_x.square() + mu_y.square() + 0.01 ** 2) * (var_x + var_y + 0.03 ** 2))
    return score.mean(dim=(-3, -2, -1))


class ImageMetrics:
    def __init__(self, device, use_lpips=False):
        self.perceptual = None
        if use_lpips:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError('LPIPS requested; install pip install -e ".[metrics]"') from exc
            self.perceptual = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def __call__(self, prediction, target):
        out = {"psnr": psnr(prediction, target).mean().item(), "ssim": ssim(prediction, target).mean().item()}
        if self.perceptual is not None:
            out["lpips"] = self.perceptual(2 * prediction - 1, 2 * target - 1).mean().item()
        return out
