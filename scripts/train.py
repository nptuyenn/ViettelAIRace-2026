import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import CHECKPOINT_DIR, SPLIT_CHOICES, get_scene_paths, get_split_root
from data.dataset import SceneTrainData
from data.splits import image_names_for_indices, make_holdout_split
from utils.io_utils import read_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", default="private", choices=SPLIT_CHOICES)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--holdout-ratio", type=float, default=0.0)
    parser.add_argument("--holdout-seed", type=int, default=2026)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    scene_root = get_split_root(args.split)
    scene_paths = get_scene_paths(scene_root, args.scene)
    config = read_yaml(args.config)

    exclude_names = set()
    if args.holdout_ratio > 0:
        scene = SceneTrainData(scene_paths["train_images"], scene_paths["train_sparse"])
        _, val_indices = make_holdout_split(scene.cameras_list, args.holdout_ratio, args.holdout_seed)
        exclude_names = image_names_for_indices(scene.cameras_list, val_indices)
        print(f"Holdout validation: excluded {len(exclude_names)} images from training")

    from training.trainer import Trainer

    trainer = Trainer(scene_paths, config, device="cuda", exclude_image_names=exclude_names)
    checkpoint_dir = CHECKPOINT_DIR / args.scene
    final_path = trainer.train(checkpoint_dir, resume=not args.no_resume)
    print(f"Training xong, checkpoint cuoi: {final_path}")


if __name__ == "__main__":
    main()
