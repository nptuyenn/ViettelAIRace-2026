import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import CHECKPOINT_DIR, OUTPUT_ROOT, SPLIT_CHOICES, get_scene_paths, get_split_root, list_scenes
from data.dataset import SceneTrainData
from data.splits import make_holdout_split
from utils.io_utils import read_image_as_array


def render_train_camera(model, camera, device):
    import torch
    from models.camera import build_intrinsics_matrix, build_viewmat

    viewmat = build_viewmat(camera["R"], camera["t"]).unsqueeze(0).to(device)
    K = build_intrinsics_matrix(camera["fx"], camera["fy"], camera["cx"], camera["cy"]).unsqueeze(0).to(device)

    with torch.no_grad():
        render_color, _, _ = model.render(viewmat, K, camera["width"], camera["height"])
    return render_color.detach().cpu().numpy()


def evaluate_local_scene(scene_name, scene_root, checkpoint_root, holdout_ratio, holdout_seed, lpips_evaluator, device, max_psnr):
    from evaluation.metrics import evaluate_scene
    from inference.renderer import load_model_for_inference
    from training.checkpoint import find_latest_checkpoint

    scene_paths = get_scene_paths(scene_root, scene_name)
    scene = SceneTrainData(scene_paths["train_images"], scene_paths["train_sparse"])
    _, val_indices = make_holdout_split(scene.cameras_list, holdout_ratio, holdout_seed)
    if len(val_indices) == 0:
        raise ValueError("Local validation needs holdout_ratio > 0")

    checkpoint_path = find_latest_checkpoint(Path(checkpoint_root) / scene_name)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found for {scene_name} in {Path(checkpoint_root) / scene_name}")

    model = load_model_for_inference(checkpoint_path, None, None, device=device)
    rendered = {}
    gt = {}
    shape_errors = []

    for index in val_indices:
        camera = scene.get_camera(index)
        name = camera["image_name"]
        pred = render_train_camera(model, camera, device)
        target = read_image_as_array(camera["image_path"])
        if pred.shape != target.shape:
            shape_errors.append(f"{name}: rendered {pred.shape[:2]} vs gt {target.shape[:2]}")
            continue
        rendered[name] = pred
        gt[name] = target

    if shape_errors:
        raise ValueError("Shape mismatch:\n" + "\n".join(shape_errors[:10]))

    result = evaluate_scene(rendered, gt, lpips_evaluator=lpips_evaluator, max_psnr=max_psnr)
    result.update({
        "scene": scene_name,
        "checkpoint": str(checkpoint_path),
        "holdout_ratio": holdout_ratio,
        "holdout_seed": holdout_seed,
    })
    return result


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    parser.add_argument("--split", default="private", choices=SPLIT_CHOICES)
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    parser.add_argument("--holdout-seed", type=int, default=2026)
    parser.add_argument("--max-psnr", type=float, default=50.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--require-lpips", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "local_eval.json"))
    args = parser.parse_args()

    scene_root = get_split_root(args.split)
    scenes = [args.scene] if args.scene else list_scenes(scene_root)

    lpips_evaluator = None
    if not args.skip_lpips:
        try:
            from evaluation.metrics import LPIPSEvaluator

            lpips_evaluator = LPIPSEvaluator(device=args.device)
        except ImportError:
            if args.require_lpips:
                raise
            print("LPIPS not available; PSNR/SSIM will be computed, score_mean will be null")

    results = []
    for scene_name in scenes:
        result = evaluate_local_scene(
            scene_name=scene_name,
            scene_root=scene_root,
            checkpoint_root=args.checkpoint_dir,
            holdout_ratio=args.holdout_ratio,
            holdout_seed=args.holdout_seed,
            lpips_evaluator=lpips_evaluator,
            device=args.device,
            max_psnr=args.max_psnr,
        )
        results.append(result)
        print(
            f"{scene_name}: n={result['num_images']} "
            f"score={result['score_mean']} psnr={result['psnr_mean']} "
            f"ssim={result['ssim_mean']} lpips={result['lpips_mean']}"
        )

    summary = {
        "split": args.split,
        "num_scenes": len(results),
        "score_mean": mean_or_none([r["score_mean"] for r in results]),
        "psnr_mean": mean_or_none([r["psnr_mean"] for r in results]),
        "ssim_mean": mean_or_none([r["ssim_mean"] for r in results]),
        "lpips_mean": mean_or_none([r["lpips_mean"] for r in results]),
        "scenes": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved local validation report to {output_path}")


if __name__ == "__main__":
    main()
