import torch


def composite(color, sigma, depths, far, white_background=True):
    """Finite near/far interval alpha compositing, Eq. (30)."""
    # Keep transmittance products stable when the field was evaluated under AMP.
    color, sigma, depths = color.float(), sigma.float(), depths.float()
    distances = torch.cat([depths[..., 1:] - depths[..., :-1], (far - depths[..., -1:]).clamp_min(0)], -1)
    alpha = -torch.expm1(-sigma * distances)
    transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1 - alpha + 1e-10], -1), -1)[..., :-1]
    weights = alpha * transmittance
    opacity = weights.sum(-1)
    rgb = (weights[..., None] * color).sum(-2)
    if white_background:
        rgb = rgb + 1 - opacity[..., None]
    return {"rgb": rgb, "depth": (weights * depths).sum(-1), "opacity": opacity, "weights": weights}
