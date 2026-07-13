import random
import torch
from pathlib import Path

from data.dataset import SceneTrainData
from data.transforms import pil_to_tensor
from models.gaussian_splatting import GaussianModel
from models.camera import build_viewmat, build_intrinsics_matrix
from models.losses import combined_loss_with_perceptual
from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.logger import get_logger

logger = get_logger("trainer")


class Trainer:
    def __init__(self, scene_paths, config, device="cuda", exclude_image_names=None):
        self.config = config
        self.device = device
        self.scene = SceneTrainData(scene_paths["train_images"], scene_paths["train_sparse"])
        self.checkpoint_dir = Path(scene_paths["scene_dir"]).name
        self.exclude_image_names = set(exclude_image_names or [])
        self.train_indices = [
            i for i, camera in enumerate(self.scene.cameras_list)
            if camera["image_name"] not in self.exclude_image_names
        ]
        if len(self.train_indices) == 0:
            raise ValueError("No training images left after applying holdout split")

        points_xyz, points_rgb = self.scene.get_point_cloud()
        if points_xyz is None:
            n = 20000
            points_xyz = (torch.rand(n, 3).numpy() - 0.5) * 2.0
            points_rgb = torch.rand(n, 3).numpy()

        self.model = GaussianModel(points_xyz, points_rgb, device, sh_degree=config.get("sh_degree", 0))
        self.lr_config = config["learning_rates"]
        self.optimizer = self._build_optimizer()
        self.gt_images = [None] * len(self.scene)

    def _build_optimizer(self):
        return torch.optim.Adam(self.model.optimizer_param_groups(self.lr_config), eps=1e-15)

    def _update_learning_rates(self, step, num_iters):
        decay_config = self.config.get("lr_decay", {})
        if not decay_config.get("enabled", False):
            return

        final_factor = float(decay_config.get("final_factor", 1.0))
        progress = min(max(step / max(num_iters, 1), 0.0), 1.0)
        factor = final_factor ** progress

        for group in self.optimizer.param_groups:
            name = group.get("name")
            base_lr = self.lr_config.get(f"{name}_lr")
            if base_lr is not None:
                group["lr"] = base_lr * factor

    def _maybe_refine_gaussians(self, step, num_iters):
        stats = {"added": 0, "pruned": 0, "opacity_reset": False}
        topology_changed = False

        densify_config = self.config.get("densification", {})
        if densify_config.get("enabled", False):
            from_iter = int(densify_config.get("from_iter", 0))
            until_iter = int(densify_config.get("until_iter", num_iters))
            every = int(densify_config.get("every", 0))
            if every > 0 and from_iter <= step <= until_iter and step % every == 0:
                stats["added"] = self.model.densify_from_gradients(densify_config)
                topology_changed = topology_changed or stats["added"] > 0

        pruning_config = self.config.get("pruning", {})
        if pruning_config.get("enabled", False):
            every = int(pruning_config.get("every", 0))
            if every > 0 and step > 0 and step % every == 0:
                stats["pruned"] = self.model.prune_low_opacity(pruning_config)
                topology_changed = topology_changed or stats["pruned"] > 0

        opacity_reset_config = self.config.get("opacity_reset", {})
        if opacity_reset_config.get("enabled", False):
            every = int(opacity_reset_config.get("every", 0))
            if every > 0 and step > 0 and step % every == 0:
                self.model.reset_opacity(opacity_reset_config.get("value", 0.1))
                stats["opacity_reset"] = True

        if topology_changed:
            self.optimizer = self._build_optimizer()
            self._update_learning_rates(step, num_iters)

        return stats

    def _get_gt_image_tensor(self, index):
        if self.gt_images[index] is None:
            image = self.scene.load_image(index)
            self.gt_images[index] = pil_to_tensor(image).to(self.device)
        return self.gt_images[index]

    def train(self, output_checkpoint_dir, resume=True):
        num_iters = self.config["num_iterations"]
        log_every = self.config.get("log_every", 100)
        save_every = self.config.get("save_every", 1000)

        start_iter = 0
        output_checkpoint_dir = Path(output_checkpoint_dir)
        output_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if resume:
            latest = find_latest_checkpoint(output_checkpoint_dir)
            if latest is not None:
                start_iter = load_checkpoint(latest, self.model, self.optimizer, self.device)
                logger.info(f"Resumed from {latest} at iteration {start_iter}")

        indices = self.train_indices

        for iteration in range(start_iter, num_iters):
            step = iteration + 1
            self._update_learning_rates(step, num_iters)

            index = random.choice(indices)
            camera = self.scene.get_camera(index)

            viewmat = build_viewmat(camera["R"], camera["t"]).unsqueeze(0).to(self.device)
            K = build_intrinsics_matrix(camera["fx"], camera["fy"], camera["cx"], camera["cy"]).unsqueeze(0).to(self.device)

            gt = self._get_gt_image_tensor(index)
            width, height = camera["width"], camera["height"]

            render_color, render_alpha, _ = self.model.render(viewmat, K, width, height)
            pred = render_color.permute(2, 0, 1)

            loss = combined_loss_with_perceptual(
                pred,
                gt,
                lambda_ssim=self.config.get("lambda_ssim", 0.2),
                lambda_lpips=self.config.get("lambda_lpips", 0.0),
                lpips_net=self.config.get("lpips_net", "alex"),
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            refine_stats = self._maybe_refine_gaussians(step, num_iters)

            if step % log_every == 0 or refine_stats["added"] > 0 or refine_stats["pruned"] > 0:
                logger.info(
                    f"iter={step} loss={loss.item():.4f} n_points={self.model.num_points()} "
                    f"added={refine_stats['added']} pruned={refine_stats['pruned']} "
                    f"opacity_reset={refine_stats['opacity_reset']}"
                )

            if step % save_every == 0:
                ckpt_path = output_checkpoint_dir / f"iter_{step:06d}.pth"
                save_checkpoint(ckpt_path, self.model, self.optimizer, step)

        final_path = output_checkpoint_dir / f"iter_{num_iters:06d}.pth"
        save_checkpoint(final_path, self.model, self.optimizer, num_iters)
        return final_path
