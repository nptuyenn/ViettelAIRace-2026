import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import CHECKPOINT_DIR, RENDERED_DIR, SPLIT_CHOICES, get_scene_paths, get_split_root, list_scenes
from data.pose_loader import load_test_poses
from inference.postprocess import save_scene_renders
from utils.logger import get_logger

logger = get_logger("render_all_scenes")


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    ckpts = sorted(checkpoint_dir.glob("iter_*.pth"))
    return ckpts[-1] if ckpts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="private", choices=SPLIT_CHOICES)
    args = parser.parse_args()

    scene_root = get_split_root(args.split)
    scenes = list_scenes(scene_root)

    for scene_name in scenes:
        scene_paths = get_scene_paths(scene_root, scene_name)
        checkpoint_dir = CHECKPOINT_DIR / scene_name
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            logger.warning(f"Bo qua {scene_name}, chua co checkpoint")
            continue

        test_poses = load_test_poses(scene_paths["test_poses_csv"])

        from inference.renderer import load_model_for_inference, render_all_poses

        model = load_model_for_inference(checkpoint_path, None, None, device="cuda")
        rendered_dict = render_all_poses(model, test_poses, device="cuda")

        scene_output_dir = RENDERED_DIR / scene_name
        save_scene_renders(rendered_dict, test_poses, scene_output_dir)
        logger.info(f"{scene_name}: da render {len(rendered_dict)} anh")


if __name__ == "__main__":
    main()
