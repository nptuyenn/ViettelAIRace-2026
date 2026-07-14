import torch
from pathlib import Path

from models.gaussian_splatting import inverse_sigmoid, num_sh_bases, rgb_to_sh, sh_to_rgb


def infer_sh_degree_from_colors(colors):
    if colors.ndim != 3:
        return 0
    bases = colors.shape[1]
    degree = int(round(bases ** 0.5)) - 1
    if num_sh_bases(degree) != bases:
        raise ValueError(f"Invalid SH color basis count: {bases}")
    return degree


def adapt_checkpoint_colors(colors, target_sh_degree, device):
    colors = colors.to(device)
    target_sh_degree = int(target_sh_degree)

    if target_sh_degree <= 0:
        if colors.ndim == 2:
            return colors
        return inverse_sigmoid(sh_to_rgb(colors))

    target_bases = num_sh_bases(target_sh_degree)
    if colors.ndim == 2:
        return rgb_to_sh(torch.sigmoid(colors), target_sh_degree)

    source_bases = colors.shape[1]
    if source_bases == target_bases:
        return colors

    adapted = torch.zeros(
        colors.shape[0],
        target_bases,
        3,
        dtype=colors.dtype,
        device=device,
    )
    copy_bases = min(source_bases, target_bases)
    adapted[:, :copy_bases, :] = colors[:, :copy_bases, :]
    return adapted


def save_checkpoint(path, model, optimizer, iteration):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": iteration,
        "means": model.means.detach().cpu(),
        "scales": model.scales.detach().cpu(),
        "quats": model.quats.detach().cpu(),
        "opacities": model.opacities.detach().cpu(),
        "colors": model.colors.detach().cpu(),
        "background": model.background.detach().cpu(),
        "sh_degree": getattr(model, "sh_degree", 0),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }, path)


def load_checkpoint(path, model, optimizer=None, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    with torch.no_grad():
        model.means.data = ckpt["means"].to(device)
        model.scales.data = ckpt["scales"].to(device)
        model.quats.data = ckpt["quats"].to(device)
        model.opacities.data = ckpt["opacities"].to(device)
        target_sh_degree = getattr(model, "sh_degree", ckpt.get("sh_degree", infer_sh_degree_from_colors(ckpt["colors"])))
        adapted_colors = adapt_checkpoint_colors(ckpt["colors"], target_sh_degree, device)
        optimizer_compatible = tuple(ckpt["colors"].shape) == tuple(adapted_colors.shape)
        model.colors.data = adapted_colors
        if "background" in ckpt:
            model.background.data = ckpt["background"].to(device)
    if optimizer is not None and ckpt.get("optimizer") is not None and optimizer_compatible:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (RuntimeError, ValueError):
            pass
    return ckpt["iteration"]


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    ckpts = sorted(checkpoint_dir.glob("iter_*.pth"))
    if len(ckpts) == 0:
        return None
    return ckpts[-1]
