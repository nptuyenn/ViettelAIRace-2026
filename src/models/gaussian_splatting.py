import math

import torch
import torch.nn as nn
import numpy as np
from gsplat import rasterization


def inverse_sigmoid(x):
    x = x.clamp(1e-6, 1 - 1e-6)
    return torch.log(x / (1 - x))


def estimate_initial_log_scale(means, sample_count=1000, ref_count=10000):
    n = means.shape[0]
    if n < 2:
        return torch.tensor(math.log(0.01), device=means.device)

    sample_count = min(sample_count, n)
    ref_count = min(ref_count, n)
    perm = torch.randperm(n, device=means.device)
    sample = means[perm[:sample_count]]
    ref = means[perm[:ref_count]]

    dists = torch.cdist(sample, ref)
    if ref_count > 1:
        nearest = torch.topk(dists, k=2, largest=False).values[:, 1]
    else:
        nearest = dists[:, 0]

    scale = nearest.median().clamp(min=1e-4)
    return torch.log(scale)


class GaussianModel(nn.Module):
    def __init__(self, points_xyz, points_rgb, device):
        super().__init__()
        self.device = device
        n = points_xyz.shape[0]

        means = torch.from_numpy(points_xyz).float().to(device)
        colors = torch.from_numpy(points_rgb).float().to(device).clamp(1e-4, 1 - 1e-4)

        init_log_scale = estimate_initial_log_scale(means)
        init_scale = torch.full((n, 3), float(init_log_scale), device=device)

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

    def _replace_parameters(self, means, scales, quats, opacities, colors):
        self.means = nn.Parameter(means.detach().to(self.device))
        self.scales = nn.Parameter(scales.detach().to(self.device))
        self.quats = nn.Parameter(quats.detach().to(self.device))
        self.opacities = nn.Parameter(opacities.detach().to(self.device))
        self.colors = nn.Parameter(colors.detach().to(self.device))

    @torch.no_grad()
    def append_gaussians(self, means, scales, quats, opacities, colors):
        self._replace_parameters(
            torch.cat([self.means.detach(), means.detach()], dim=0),
            torch.cat([self.scales.detach(), scales.detach()], dim=0),
            torch.cat([self.quats.detach(), quats.detach()], dim=0),
            torch.cat([self.opacities.detach(), opacities.detach()], dim=0),
            torch.cat([self.colors.detach(), colors.detach()], dim=0),
        )

    @torch.no_grad()
    def prune_gaussians(self, keep_mask):
        keep_mask = keep_mask.to(self.device)
        pruned = int((~keep_mask).sum().item())
        if pruned == 0:
            return 0
        self._replace_parameters(
            self.means.detach()[keep_mask],
            self.scales.detach()[keep_mask],
            self.quats.detach()[keep_mask],
            self.opacities.detach()[keep_mask],
            self.colors.detach()[keep_mask],
        )
        return pruned

    @torch.no_grad()
    def densify_from_gradients(self, densify_config):
        if self.means.grad is None:
            return 0

        max_points = int(densify_config.get("max_points", 300000))
        if self.num_points() >= max_points:
            return 0

        grad_threshold = float(densify_config.get("grad_threshold", 5e-5))
        max_new_points = int(densify_config.get("max_new_points", 5000))
        jitter_scale = float(densify_config.get("jitter_scale", 0.25))
        scale_shrink = float(densify_config.get("scale_shrink", 0.8))

        grad_norm = self.means.grad.detach().norm(dim=-1)
        candidates = torch.nonzero(grad_norm > grad_threshold, as_tuple=False).flatten()
        if candidates.numel() == 0:
            return 0

        remaining = max_points - self.num_points()
        max_new_points = max(0, min(max_new_points, remaining))
        if max_new_points == 0:
            return 0

        if candidates.numel() > max_new_points:
            candidate_grads = grad_norm[candidates]
            candidates = candidates[torch.topk(candidate_grads, k=max_new_points, largest=True).indices]

        parent_means = self.means.detach()[candidates]
        parent_scales_log = self.scales.detach()[candidates]
        parent_scales = torch.exp(parent_scales_log)
        jitter = torch.randn_like(parent_means) * parent_scales * jitter_scale

        new_means = parent_means + jitter
        new_scales = parent_scales_log + math.log(max(scale_shrink, 1e-3))
        new_quats = self.quats.detach()[candidates].clone()
        new_opacities = self.opacities.detach()[candidates].clone()
        new_colors = self.colors.detach()[candidates].clone()

        self.append_gaussians(new_means, new_scales, new_quats, new_opacities, new_colors)
        return int(candidates.numel())

    @torch.no_grad()
    def prune_low_opacity(self, pruning_config):
        opacity_threshold = float(pruning_config.get("opacity_threshold", 0.005))
        min_points = int(pruning_config.get("min_points", 5000))
        n = self.num_points()
        if n <= min_points:
            return 0

        opacities = self.get_opacities().detach()
        keep_mask = opacities >= opacity_threshold
        if int(keep_mask.sum().item()) < min_points:
            keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
            keep_indices = torch.topk(opacities, k=min_points, largest=True).indices
            keep_mask[keep_indices] = True

        return self.prune_gaussians(keep_mask)

    @torch.no_grad()
    def reset_opacity(self, value=0.1):
        value_tensor = torch.full_like(self.opacities, float(value)).clamp(1e-4, 1 - 1e-4)
        self.opacities.data.copy_(inverse_sigmoid(value_tensor))
