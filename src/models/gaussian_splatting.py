import math

import torch
import torch.nn as nn
import numpy as np
from gsplat import rasterization

SH_C0 = 0.28209479177387814


def inverse_sigmoid(x):
    x = x.clamp(1e-6, 1 - 1e-6)
    return torch.log(x / (1 - x))


def num_sh_bases(sh_degree):
    return (int(sh_degree) + 1) ** 2


def rgb_to_sh(rgb, sh_degree):
    coeffs = torch.zeros(
        rgb.shape[0],
        num_sh_bases(sh_degree),
        3,
        dtype=rgb.dtype,
        device=rgb.device,
    )
    coeffs[:, 0, :] = (rgb - 0.5) / SH_C0
    return coeffs


def sh_to_rgb(sh_coeffs):
    return (sh_coeffs[:, 0, :] * SH_C0 + 0.5).clamp(1e-4, 1 - 1e-4)


def rotate_vectors_by_quat(vectors, quats):
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    qvec = quats[:, 1:]
    uv = torch.cross(qvec, vectors, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vectors + 2.0 * (quats[:, :1] * uv + uuv)


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
    def __init__(
        self,
        points_xyz,
        points_rgb,
        device,
        sh_degree=0,
        background_color=(0.0, 0.0, 0.0),
        learn_background=False,
        rasterize_mode="classic",
        absgrad=False,
    ):
        super().__init__()
        self.device = device
        self.sh_degree = int(sh_degree)
        self.rasterize_mode = rasterize_mode
        self.absgrad = bool(absgrad)
        self.last_render_meta = None
        n = points_xyz.shape[0]

        means = torch.from_numpy(points_xyz).float().to(device)
        colors = torch.from_numpy(points_rgb).float().to(device).clamp(1e-4, 1 - 1e-4)
        background = torch.tensor(background_color, dtype=torch.float32, device=device).clamp(1e-4, 1 - 1e-4)

        init_log_scale = estimate_initial_log_scale(means)
        init_scale = torch.full((n, 3), float(init_log_scale), device=device)

        quats = torch.zeros(n, 4, device=device)
        quats[:, 0] = 1.0

        opacities = inverse_sigmoid(torch.full((n, 1), 0.5, device=device))

        self.means = nn.Parameter(means)
        self.scales = nn.Parameter(init_scale)
        self.quats = nn.Parameter(quats)
        self.opacities = nn.Parameter(opacities)
        if self.sh_degree > 0:
            self.colors = nn.Parameter(rgb_to_sh(colors, self.sh_degree))
        else:
            self.colors = nn.Parameter(inverse_sigmoid(colors))
        self.background = nn.Parameter(inverse_sigmoid(background), requires_grad=bool(learn_background))

    def get_scales(self):
        return torch.exp(self.scales)

    def get_opacities(self):
        return torch.sigmoid(self.opacities).squeeze(-1)

    def get_colors(self):
        if self.sh_degree > 0:
            return self.colors
        return torch.sigmoid(self.colors)

    def get_quats(self):
        return self.quats / self.quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def get_background(self):
        return torch.sigmoid(self.background)

    def render(self, viewmats, Ks, width, height, sh_degree=None):
        active_sh_degree = self.sh_degree if sh_degree is None else min(int(sh_degree), self.sh_degree)
        active_sh_degree = active_sh_degree if active_sh_degree > 0 else None
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
            sh_degree=active_sh_degree,
            rasterize_mode=self.rasterize_mode,
            absgrad=self.absgrad,
            packed=False,
        )
        self.last_render_meta = meta
        render_color = render_colors[0]
        render_alpha = render_alphas[0]
        render_color = render_color + (1.0 - render_alpha) * self.get_background().view(1, 1, 3)
        return render_color, render_alpha, meta

    def optimizer_param_groups(self, lr_config):
        groups = [
            {"params": [self.means], "lr": lr_config["means_lr"], "name": "means"},
            {"params": [self.scales], "lr": lr_config["scales_lr"], "name": "scales"},
            {"params": [self.quats], "lr": lr_config["quats_lr"], "name": "quats"},
            {"params": [self.opacities], "lr": lr_config["opacities_lr"], "name": "opacities"},
            {"params": [self.colors], "lr": lr_config["colors_lr"], "name": "colors"},
        ]
        if self.background.requires_grad:
            groups.append({
                "params": [self.background],
                "lr": lr_config.get("background_lr", lr_config["colors_lr"]),
                "name": "background",
            })
        return groups

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
        grad_norm = self._densification_grad_norm(densify_config)
        if grad_norm is None:
            return 0
        return self.densify_from_scores(grad_norm, densify_config)

    @torch.no_grad()
    def densify_from_scores(self, grad_scores, densify_config):
        max_points = int(densify_config.get("max_points", 300000))
        if self.num_points() >= max_points:
            return 0

        strategy = densify_config.get("strategy", "jitter_clone")
        if strategy == "clone_split":
            return self._densify_clone_split(grad_scores, densify_config)
        return self._densify_jitter_clone(grad_scores, densify_config)

    @torch.no_grad()
    def _densify_jitter_clone(self, grad_scores, densify_config):
        max_points = int(densify_config.get("max_points", 300000))
        grad_threshold = float(densify_config.get("grad_threshold", 5e-5))
        max_new_points = int(densify_config.get("max_new_points", 5000))
        jitter_scale = float(densify_config.get("jitter_scale", 0.25))
        scale_shrink = float(densify_config.get("scale_shrink", 0.8))

        grad_scores = grad_scores.detach().to(self.device)
        candidates = torch.nonzero(grad_scores > grad_threshold, as_tuple=False).flatten()
        if candidates.numel() == 0:
            return 0

        remaining = max_points - self.num_points()
        max_new_points = max(0, min(max_new_points, remaining))
        if max_new_points == 0:
            return 0

        if candidates.numel() > max_new_points:
            candidate_grads = grad_scores[candidates]
            candidates = candidates[torch.topk(candidate_grads, k=max_new_points, largest=True).indices]

        parent_means = self.means.detach()[candidates]
        parent_scales_log = self.scales.detach()[candidates]
        parent_scales = torch.exp(parent_scales_log)
        parent_quats = self.get_quats().detach()[candidates]
        local_jitter = torch.randn_like(parent_means) * parent_scales * jitter_scale
        jitter = rotate_vectors_by_quat(local_jitter, parent_quats)

        new_means = parent_means + jitter
        new_scales = parent_scales_log + math.log(max(scale_shrink, 1e-3))
        new_quats = parent_quats.clone()
        new_opacities = self.opacities.detach()[candidates].clone()
        new_colors = self.colors.detach()[candidates].clone()

        self.append_gaussians(new_means, new_scales, new_quats, new_opacities, new_colors)
        return int(candidates.numel())

    @torch.no_grad()
    def _densify_clone_split(self, grad_scores, densify_config):
        max_points = int(densify_config.get("max_points", 300000))
        grad_threshold = float(densify_config.get("grad_threshold", 5e-5))
        max_new_points = int(densify_config.get("max_new_points", 5000))
        jitter_scale = float(densify_config.get("jitter_scale", 0.25))
        split_samples = max(2, int(densify_config.get("split_samples", 2)))
        clone_scale_shrink = float(densify_config.get("clone_scale_shrink", 1.0))
        split_scale_shrink = float(densify_config.get("split_scale_shrink", 0.6))
        split_scale_quantile = float(densify_config.get("split_scale_quantile", 0.75))
        opacity_scale = float(densify_config.get("new_opacity_scale", 0.8))

        n = self.num_points()
        remaining = max_points - n
        if remaining <= 0:
            return 0
        max_new_points = max(0, min(max_new_points, remaining))
        if max_new_points == 0:
            return 0

        grad_scores = grad_scores.detach().to(self.device)
        candidates = torch.nonzero(grad_scores > grad_threshold, as_tuple=False).flatten()
        if candidates.numel() == 0:
            return 0

        if candidates.numel() > max_new_points:
            candidate_grads = grad_scores[candidates]
            candidates = candidates[torch.topk(candidate_grads, k=max_new_points, largest=True).indices]

        scales_log = self.scales.detach()
        scales = torch.exp(scales_log)
        max_scale = scales.max(dim=-1).values
        split_threshold = densify_config.get("split_scale_threshold")
        if split_threshold is None:
            split_threshold = torch.quantile(max_scale, split_scale_quantile).item()
        split_threshold = float(split_threshold)

        split_candidates = candidates[max_scale[candidates] > split_threshold]
        clone_candidates = candidates[max_scale[candidates] <= split_threshold]

        new_means = []
        new_scales = []
        new_quats = []
        new_opacities = []
        new_colors = []
        parents_to_remove = []
        created = 0

        split_parent_budget = max_new_points // split_samples
        if split_candidates.numel() > split_parent_budget:
            if split_parent_budget <= 0:
                split_candidates = split_candidates[:0]
            else:
                split_scores = grad_scores[split_candidates]
                split_candidates = split_candidates[torch.topk(split_scores, k=split_parent_budget, largest=True).indices]

        if split_candidates.numel() > 0:
            parent_means = self.means.detach()[split_candidates]
            parent_scales_log = scales_log[split_candidates]
            parent_scales = scales[split_candidates]
            parent_quats = self.get_quats().detach()[split_candidates]
            repeated = split_candidates.repeat_interleave(split_samples)
            child_count = int(repeated.numel())
            local_jitter = (
                torch.randn(child_count, 3, device=self.device)
                * parent_scales.repeat_interleave(split_samples, dim=0)
                * jitter_scale
            )
            jitter = rotate_vectors_by_quat(local_jitter, parent_quats.repeat_interleave(split_samples, dim=0))
            new_means.append(parent_means.repeat_interleave(split_samples, dim=0) + jitter)
            new_scales.append(parent_scales_log.repeat_interleave(split_samples, dim=0) + math.log(max(split_scale_shrink, 1e-3)))
            new_quats.append(parent_quats.repeat_interleave(split_samples, dim=0).clone())
            new_opacities.append(self._scaled_opacities(repeated, opacity_scale / split_samples))
            new_colors.append(self.colors.detach()[repeated].clone())
            parents_to_remove = split_candidates
            created += child_count

        remaining_new = max_new_points - created
        clone_budget = max(0, remaining_new)
        if clone_candidates.numel() > clone_budget:
            if clone_budget <= 0:
                clone_candidates = clone_candidates[:0]
            else:
                clone_scores = grad_scores[clone_candidates]
                clone_candidates = clone_candidates[torch.topk(clone_scores, k=clone_budget, largest=True).indices]
        if clone_candidates.numel() > 0:
            parent_scales = scales[clone_candidates]
            parent_quats = self.get_quats().detach()[clone_candidates]
            local_jitter = torch.randn_like(self.means.detach()[clone_candidates]) * parent_scales * jitter_scale
            jitter = rotate_vectors_by_quat(local_jitter, parent_quats)
            new_means.append(self.means.detach()[clone_candidates] + jitter)
            new_scales.append(scales_log[clone_candidates] + math.log(max(clone_scale_shrink, 1e-3)))
            new_quats.append(parent_quats.clone())
            new_opacities.append(self._scaled_opacities(clone_candidates, opacity_scale))
            new_colors.append(self.colors.detach()[clone_candidates].clone())
            created += int(clone_candidates.numel())

        if created == 0:
            return 0

        keep_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        if len(parents_to_remove) != 0:
            keep_mask[parents_to_remove] = False

        self._replace_parameters(
            torch.cat([self.means.detach()[keep_mask], *new_means], dim=0),
            torch.cat([self.scales.detach()[keep_mask], *new_scales], dim=0),
            torch.cat([self.quats.detach()[keep_mask], *new_quats], dim=0),
            torch.cat([self.opacities.detach()[keep_mask], *new_opacities], dim=0),
            torch.cat([self.colors.detach()[keep_mask], *new_colors], dim=0),
        )
        return created

    def _scaled_opacities(self, indices, scale):
        opacities = torch.sigmoid(self.opacities.detach()[indices]) * float(scale)
        return inverse_sigmoid(opacities.clamp(1e-4, 1 - 1e-4))

    def _densification_grad_norm(self, densify_config):
        if densify_config.get("use_absgrad", False) and self.last_render_meta is not None:
            means2d = self.last_render_meta.get("means2d")
            absgrad = getattr(means2d, "absgrad", None) if means2d is not None else None
            if absgrad is not None:
                grad_norm = absgrad.detach().norm(dim=-1)
                while grad_norm.ndim > 1:
                    grad_norm = grad_norm.max(dim=0).values
                if grad_norm.shape[0] == self.num_points():
                    return grad_norm

        if self.means.grad is None:
            return None
        return self.means.grad.detach().norm(dim=-1)

    def densification_grad_scores(self, densify_config):
        return self._densification_grad_norm(densify_config)

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
