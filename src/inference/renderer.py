import torch
import numpy as np

from models.gaussian_splatting import GaussianModel
from models.camera import qvec_tvec_to_viewmat, build_intrinsics_matrix
from training.checkpoint import infer_sh_degree_from_colors, load_checkpoint
from data.transforms import tensor_to_array


def load_model_for_inference(checkpoint_path, points_xyz_dummy, points_rgb_dummy, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    n = ckpt["means"].shape[0]
    sh_degree = ckpt.get("sh_degree", infer_sh_degree_from_colors(ckpt["colors"]))
    dummy_xyz = np.zeros((n, 3), dtype=np.float32)
    dummy_rgb = np.full((n, 3), 0.5, dtype=np.float32)
    model = GaussianModel(dummy_xyz, dummy_rgb, device, sh_degree=sh_degree)
    load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    model.eval()
    return model


def render_pose(model, test_pose, device="cuda"):
    viewmat = qvec_tvec_to_viewmat(test_pose.qvec, test_pose.tvec).unsqueeze(0).to(device)
    K = build_intrinsics_matrix(test_pose.fx, test_pose.fy, test_pose.cx, test_pose.cy).unsqueeze(0).to(device)

    with torch.no_grad():
        render_color, render_alpha, _ = model.render(viewmat, K, test_pose.width, test_pose.height)

    pred = render_color.permute(2, 0, 1)
    return tensor_to_array(pred)


def render_all_poses(model, test_poses, device="cuda"):
    results = {}
    for pose in test_poses:
        results[pose.image_name] = render_pose(model, pose, device)
    return results
