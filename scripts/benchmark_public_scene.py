import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import OUTPUT_ROOT, PUBLIC_SET_ROOT, get_scene_paths, list_scenes
from data.dataset import SceneTrainData
from data.splits import image_names_for_indices, make_holdout_split
from eval_local import evaluate_local_scene
from training.trainer import Trainer
from utils.io_utils import read_yaml


def first_public_scene():
    scenes = list_scenes(PUBLIC_SET_ROOT)
    if not scenes:
        raise FileNotFoundError(f"No public scenes found in {PUBLIC_SET_ROOT}")
    return scenes[0]


def fail_if_threshold_missed(result, args):
    failures = []
    if args.min_score is not None:
        score = result.get("score_mean")
        if score is None or score < args.min_score:
            failures.append(f"score_mean {score} < {args.min_score}")
    if args.min_psnr is not None:
        psnr = result.get("psnr_mean")
        if psnr is None or psnr < args.min_psnr:
            failures.append(f"psnr_mean {psnr} < {args.min_psnr}")
    if args.min_ssim is not None:
        ssim = result.get("ssim_mean")
        if ssim is None or ssim < args.min_ssim:
            failures.append(f"ssim_mean {ssim} < {args.min_ssim}")
    if args.max_lpips is not None:
        lpips = result.get("lpips_mean")
        if lpips is None or lpips > args.max_lpips:
            failures.append(f"lpips_mean {lpips} > {args.max_lpips}")

    if failures:
        print("Benchmark failed:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None)
    parser.add_argument("--config", default="configs/competitive.yaml")
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    parser.add_argument("--holdout-seed", type=int, default=2026)
    parser.add_argument("--checkpoint-root", default=str(OUTPUT_ROOT / "public_benchmark_checkpoints"))
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "public_benchmark.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--require-lpips", action="store_true")
    parser.add_argument("--max-psnr", type=float, default=50.0)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--min-psnr", type=float, default=None)
    parser.add_argument("--min-ssim", type=float, default=None)
    parser.add_argument("--max-lpips", type=float, default=None)
    args = parser.parse_args()

    scene_name = args.scene or first_public_scene()
    scene_paths = get_scene_paths(PUBLIC_SET_ROOT, scene_name)
    config = read_yaml(args.config)

    scene = SceneTrainData(scene_paths["train_images"], scene_paths["train_sparse"])
    _, val_indices = make_holdout_split(scene.cameras_list, args.holdout_ratio, args.holdout_seed)
    exclude_names = image_names_for_indices(scene.cameras_list, val_indices)
    print(f"{scene_name}: holdout {len(exclude_names)} / {len(scene.cameras_list)} train images")

    trainer = Trainer(scene_paths, config, device=args.device, exclude_image_names=exclude_names)
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_dir = checkpoint_root / scene_name
    final_checkpoint = trainer.train(checkpoint_dir, resume=not args.no_resume)
    print(f"{scene_name}: final checkpoint {final_checkpoint}")

    lpips_evaluator = None
    if not args.skip_lpips:
        try:
            from evaluation.metrics import LPIPSEvaluator

            lpips_evaluator = LPIPSEvaluator(device=args.device)
        except ImportError:
            if args.require_lpips:
                raise
            print("LPIPS not available; score_mean will be null")

    result = evaluate_local_scene(
        scene_name=scene_name,
        scene_root=PUBLIC_SET_ROOT,
        checkpoint_root=checkpoint_root,
        holdout_ratio=args.holdout_ratio,
        holdout_seed=args.holdout_seed,
        lpips_evaluator=lpips_evaluator,
        device=args.device,
        max_psnr=args.max_psnr,
    )
    result.update({"config": args.config, "final_checkpoint": str(final_checkpoint)})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"{scene_name}: score={result['score_mean']} psnr={result['psnr_mean']} "
        f"ssim={result['ssim_mean']} lpips={result['lpips_mean']}"
    )
    print(f"Saved benchmark report to {output_path}")
    fail_if_threshold_missed(result, args)


if __name__ == "__main__":
    main()
