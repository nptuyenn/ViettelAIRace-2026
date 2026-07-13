import torch
import torch.nn.functional as F

_LPIPS_MODELS = {}


def l1_loss(pred, target):
    return torch.abs(pred - target).mean()


def _gaussian_kernel(window_size, sigma, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel


def ssim(pred, target, window_size=11, sigma=1.5):
    device = pred.device
    channels = pred.shape[0]
    kernel = _gaussian_kernel(window_size, sigma, device)
    kernel = kernel.expand(channels, 1, window_size, window_size)

    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def combined_loss(pred, target, lambda_ssim=0.2):
    l1 = l1_loss(pred, target)
    d_ssim = 1.0 - ssim(pred, target)
    return (1 - lambda_ssim) * l1 + lambda_ssim * d_ssim


def lpips_loss(pred, target, net="alex"):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("Install lpips to enable lambda_lpips > 0: pip install lpips") from exc

    device = pred.device
    key = (str(device), net)
    if key not in _LPIPS_MODELS:
        model = lpips.LPIPS(net=net).to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _LPIPS_MODELS[key] = model

    pred_n = pred.unsqueeze(0) * 2.0 - 1.0
    target_n = target.unsqueeze(0) * 2.0 - 1.0
    return _LPIPS_MODELS[key](pred_n, target_n).mean()


def combined_loss_with_perceptual(pred, target, lambda_ssim=0.2, lambda_lpips=0.0, lpips_net="alex"):
    base = combined_loss(pred, target, lambda_ssim=lambda_ssim)
    if lambda_lpips <= 0:
        return base
    return base + lambda_lpips * lpips_loss(pred, target, net=lpips_net)
