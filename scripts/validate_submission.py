import sys
import argparse
import io
import zipfile
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import PUBLIC_SET_ROOT, PRIVATE_TEST_ROOT, RENDERED_DIR, get_scene_paths, list_scenes
from data.pose_loader import load_test_poses


def _expected_scene_files(scene_name, scene_root):
    scene_paths = get_scene_paths(scene_root, scene_name)
    test_poses = load_test_poses(scene_paths["test_poses_csv"])
    return {pose.image_name: pose for pose in test_poses}, test_poses


def validate_scene_dir(scene_name, scene_root, rendered_root):
    pose_lookup, test_poses = _expected_scene_files(scene_name, scene_root)
    scene_rendered_dir = Path(rendered_root) / scene_name

    errors = []
    if not scene_rendered_dir.exists():
        return [f"{scene_name}: thieu thu muc render"]

    if len(pose_lookup) != len(test_poses):
        errors.append(f"{scene_name}: test_poses.csv co ten anh bi trung")

    actual_files = {p.name for p in scene_rendered_dir.iterdir() if p.is_file()}
    expected_files = set(pose_lookup)

    for image_name in sorted(expected_files - actual_files):
        errors.append(f"{scene_name}/{image_name}: thieu file")
    for image_name in sorted(actual_files - expected_files):
        errors.append(f"{scene_name}/{image_name}: file thua")

    for pose in test_poses:
        image_path = scene_rendered_dir / pose.image_name
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            if img.size != (pose.width, pose.height):
                errors.append(
                    f"{scene_name}/{pose.image_name}: kich thuoc {img.size} khac yeu cau {(pose.width, pose.height)}"
                )
            if img.mode not in ("RGB", "RGBA"):
                errors.append(f"{scene_name}/{pose.image_name}: mode {img.mode} khong phai RGB/RGBA")
    return errors


def validate_rendered_dir(scene_root, rendered_root):
    expected_scenes = set(list_scenes(scene_root))
    rendered_root = Path(rendered_root)
    actual_scenes = {p.name for p in rendered_root.iterdir() if p.is_dir()} if rendered_root.exists() else set()

    errors = []
    for scene_name in sorted(expected_scenes - actual_scenes):
        errors.append(f"{scene_name}: thieu scene")
    for scene_name in sorted(actual_scenes - expected_scenes):
        errors.append(f"{scene_name}: scene thua")
    for scene_name in sorted(expected_scenes):
        errors.extend(validate_scene_dir(scene_name, scene_root, rendered_root))
    return errors


def validate_zip(scene_root, zip_path):
    expected_scenes = set(list_scenes(scene_root))
    errors = []
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return [f"{zip_path}: khong ton tai"]

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        actual_scenes = {Path(name).parts[0] for name in names if len(Path(name).parts) >= 2}
        actual_files_by_scene = {}
        for name in names:
            parts = Path(name).parts
            if len(parts) != 2:
                errors.append(f"{name}: duong dan trong zip phai co dang <scene>/<image>")
                continue
            actual_files_by_scene.setdefault(parts[0], set()).add(parts[1])

        for scene_name in sorted(expected_scenes - actual_scenes):
            errors.append(f"{scene_name}: thieu scene trong zip")
        for scene_name in sorted(actual_scenes - expected_scenes):
            errors.append(f"{scene_name}: scene thua trong zip")

        for scene_name in sorted(expected_scenes):
            pose_lookup, test_poses = _expected_scene_files(scene_name, scene_root)
            actual_files = actual_files_by_scene.get(scene_name, set())
            expected_files = set(pose_lookup)
            for image_name in sorted(expected_files - actual_files):
                errors.append(f"{scene_name}/{image_name}: thieu file trong zip")
            for image_name in sorted(actual_files - expected_files):
                errors.append(f"{scene_name}/{image_name}: file thua trong zip")
            for pose in test_poses:
                arcname = f"{scene_name}/{pose.image_name}"
                if arcname not in zf.namelist():
                    continue
                with zf.open(arcname) as f:
                    data = io.BytesIO(f.read())
                with Image.open(data) as img:
                    if img.size != (pose.width, pose.height):
                        errors.append(
                            f"{arcname}: kich thuoc {img.size} khac yeu cau {(pose.width, pose.height)}"
                        )
                    if img.mode not in ("RGB", "RGBA"):
                        errors.append(f"{arcname}: mode {img.mode} khong phai RGB/RGBA")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="private", choices=["public", "private"])
    parser.add_argument("--rendered_dir", default=str(RENDERED_DIR))
    parser.add_argument("--zip", default=None)
    args = parser.parse_args()

    scene_root = PUBLIC_SET_ROOT if args.split == "public" else PRIVATE_TEST_ROOT
    all_errors = validate_zip(scene_root, args.zip) if args.zip else validate_rendered_dir(scene_root, args.rendered_dir)

    if len(all_errors) == 0:
        print("Submission hop le, khong co loi")
    else:
        print(f"Tim thay {len(all_errors)} loi:")
        for e in all_errors:
            print(f"  {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
