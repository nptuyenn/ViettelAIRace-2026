import numpy as np
import torch

from models.losses import ssim as ssim_torch


def psnr(pred, target, max_val=1.0):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def ssim(pred, target):
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).float()
    target_t = torch.from_numpy(target).permute(2, 0, 1).float()
    return ssim_torch(pred_t, target_t).item()


class LPIPSEvaluator:
    def __init__(self, device="cuda", net="alex"):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("Install lpips to compute LPIPS: pip install lpips") from exc

        self.device = device
        self.model = lpips.LPIPS(net=net).to(device).eval()

    def __call__(self, pred, target):
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        target_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        pred_t = pred_t * 2.0 - 1.0
        target_t = target_t * 2.0 - 1.0
        with torch.no_grad():
            return float(self.model(pred_t, target_t).item())


def normalize_psnr(psnr_value, max_psnr=50.0):
    if np.isinf(psnr_value):
        return 1.0
    return float(np.clip(psnr_value / max_psnr, 0.0, 1.0))


def competition_score(psnr_value, ssim_value, lpips_value, max_psnr=50.0):
    if lpips_value is None:
        return None
    return (
        0.4 * (1.0 - float(lpips_value))
        + 0.3 * float(ssim_value)
        + 0.3 * normalize_psnr(psnr_value, max_psnr)
    )


def evaluate_scene(rendered_dict, gt_dict, lpips_evaluator=None, max_psnr=50.0):
    psnr_values = []
    ssim_values = []
    lpips_values = []
    score_values = []
    for name, pred in rendered_dict.items():
        if name not in gt_dict:
            continue
        target = gt_dict[name]
        psnr_value = psnr(pred, target)
        ssim_value = ssim(pred, target)
        lpips_value = lpips_evaluator(pred, target) if lpips_evaluator is not None else None
        score_value = competition_score(psnr_value, ssim_value, lpips_value, max_psnr)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        if lpips_value is not None:
            lpips_values.append(lpips_value)
        if score_value is not None:
            score_values.append(score_value)

    return {
        "psnr_mean": float(np.mean(psnr_values)) if psnr_values else None,
        "ssim_mean": float(np.mean(ssim_values)) if ssim_values else None,
        "lpips_mean": float(np.mean(lpips_values)) if lpips_values else None,
        "score_mean": float(np.mean(score_values)) if score_values else None,
        "num_images": len(psnr_values),
    }
