import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import PUBLIC_SET_ROOT, PRIVATE_TEST_ROOT, CHECKPOINT_DIR, RENDERED_DIR, get_scene_paths
from data.pose_loader import load_test_poses
from inference.renderer import load_model_for_inference, render_all_poses
from inference.postprocess import save_scene_renders
from training.checkpoint import find_latest_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", default="private", choices=["public", "private"])
    args = parser.parse_args()

    scene_root = PUBLIC_SET_ROOT if args.split == "public" else PRIVATE_TEST_ROOT
    scene_paths = get_scene_paths(scene_root, args.scene)

    checkpoint_dir = CHECKPOINT_DIR / args.scene
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"Khong tim thay checkpoint trong {checkpoint_dir}")

    test_poses = load_test_poses(scene_paths["test_poses_csv"])
    model = load_model_for_inference(checkpoint_path, None, None, device="cuda")

    rendered_dict = render_all_poses(model, test_poses, device="cuda")

    scene_output_dir = RENDERED_DIR / args.scene
    save_scene_renders(rendered_dict, test_poses, scene_output_dir)
    print(f"Da render {len(rendered_dict)} anh vao {scene_output_dir}")


if __name__ == "__main__":
    main()
