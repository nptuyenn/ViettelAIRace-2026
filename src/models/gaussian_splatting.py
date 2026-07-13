import torch
import torch.nn as nn
import numpy as np
from gsplat import rasterization


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class GaussianModel(nn.Module):
    def __init__(self, points_xyz, points_rgb, device):
        super().__init__()
        self.device = device
        n = points_xyz.shape[0]

        means = torch.from_numpy(points_xyz).float().to(device)
        colors = torch.from_numpy(points_rgb).float().to(device).clamp(1e-4, 1 - 1e-4)

        dist = torch.cdist(means[:2000], means).min(dim=1).values.mean() if n > 2000 else torch.tensor(0.01)
        init_scale = torch.full((n, 3), float(torch.log(dist.clamp(min=1e-6))), device=device)

        quats = torch.zeros(n, 4, device=device)
        quats[:, 0] = 1.0

        opacities = inverse_sigmoid(torch.full((n, 1), 0.5, device=device))

        self.means = nn.Parameter(means)
        self.scales = nn.Parameter(init_scale)
        self.quats = nn.Parameter(quats)
        self.opacities = nn.Parameter(opacities)
        self.colors = nn.Parameter(inverse_sigmoid(colors))

    def get_scales(self):
        return torch.exp(self.scales)

    def get_opacities(self):
        return torch.sigmoid(self.opacities).squeeze(-1)

    def get_colors(self):
        return torch.sigmoid(self.colors)

    def get_quats(self):
        return self.quats / self.quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def render(self, viewmats, Ks, width, height, sh_degree=None):
        colors = self.get_colors()
        render_colors, render_alphas, meta = rasterization(
            means=self.means,
            quats=self.get_quats(),
            scales=self.get_scales(),
            opacities=self.get_opacities(),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            packed=False,
        )
        return render_colors[0], render_alphas[0], meta

    def optimizer_param_groups(self, lr_config):
        return [
            {"params": [self.means], "lr": lr_config["means_lr"], "name": "means"},
            {"params": [self.scales], "lr": lr_config["scales_lr"], "name": "scales"},
            {"params": [self.quats], "lr": lr_config["quats_lr"], "name": "quats"},
            {"params": [self.opacities], "lr": lr_config["opacities_lr"], "name": "opacities"},
            {"params": [self.colors], "lr": lr_config["colors_lr"], "name": "colors"},
        ]

    def num_points(self):
        return self.means.shape[0]
