import torch


def stratified_samples(origins, near, far, samples, perturb):
    edges = torch.linspace(near, far, samples + 1, device=origins.device, dtype=origins.dtype)
    shape = (*origins.shape[:-1], samples)
    offsets = torch.rand(shape, device=origins.device) if perturb else origins.new_full(shape, 0.5)
    return edges[:-1] + (edges[1:] - edges[:-1]) * offsets


def sample_pdf(bins, weights, count, deterministic=False):
    """Inverse-CDF sampling; bins has one more entry than weights."""
    weights = weights.detach() + 1e-5
    pdf = weights / weights.sum(-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[..., :1]), pdf.cumsum(-1)], -1)
    shape = (*cdf.shape[:-1], count)
    if deterministic:
        u = ((torch.arange(count, device=bins.device, dtype=bins.dtype) + 0.5) / count).expand(shape)
    else:
        u = torch.rand(shape, device=bins.device, dtype=bins.dtype)
    index = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    low, high = (index - 1).clamp_min(0), index.clamp_max(cdf.shape[-1] - 1)
    c0, c1 = torch.gather(cdf, -1, low), torch.gather(cdf, -1, high)
    b0, b1 = torch.gather(bins, -1, low), torch.gather(bins, -1, high)
    return b0 + (u - c0) / (c1 - c0).clamp_min(1e-8) * (b1 - b0)
