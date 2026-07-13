import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import PUBLIC_SET_ROOT, PRIVATE_TEST_ROOT, CHECKPOINT_DIR, get_scene_paths, list_scenes
from data.dataset import SceneTrainData
from data.splits import image_names_for_indices, make_holdout_split
from utils.io_utils import read_yaml
from utils.logger import get_logger

logger = get_logger("train_all_scenes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="private", choices=["public", "private"])
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--holdout-ratio", type=float, default=0.0)
    parser.add_argument("--holdout-seed", type=int, default=2026)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    scene_root = PUBLIC_SET_ROOT if args.split == "public" else PRIVATE_TEST_ROOT
    config = read_yaml(args.config)
    scenes = list_scenes(scene_root)

    for scene_name in scenes:
        logger.info(f"Training scene {scene_name}")
        scene_paths = get_scene_paths(scene_root, scene_name)
        exclude_names = set()
        if args.holdout_ratio > 0:
            scene = SceneTrainData(scene_paths["train_images"], scene_paths["train_sparse"])
            _, val_indices = make_holdout_split(scene.cameras_list, args.holdout_ratio, args.holdout_seed)
            exclude_names = image_names_for_indices(scene.cameras_list, val_indices)
            logger.info(f"{scene_name}: excluded {len(exclude_names)} holdout images from training")
        from training.trainer import Trainer

        trainer = Trainer(scene_paths, config, device="cuda", exclude_image_names=exclude_names)
        checkpoint_dir = CHECKPOINT_DIR / scene_name
        trainer.train(checkpoint_dir, resume=not args.no_resume)


if __name__ == "__main__":
    main()
