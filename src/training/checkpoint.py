import torch
from pathlib import Path


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
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }, path)


def load_checkpoint(path, model, optimizer=None, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    with torch.no_grad():
        model.means.data = ckpt["means"].to(device)
        model.scales.data = ckpt["scales"].to(device)
        model.quats.data = ckpt["quats"].to(device)
        model.opacities.data = ckpt["opacities"].to(device)
        model.colors.data = ckpt["colors"].to(device)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    ckpts = sorted(checkpoint_dir.glob("iter_*.pth"))
    if len(ckpts) == 0:
        return None
    return ckpts[-1]
