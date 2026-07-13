from pathlib import Path
from utils.io_utils import save_image_array


def save_rendered_image(array, width, height, output_path):
    h, w = array.shape[0], array.shape[1]
    if (w, h) != (width, height):
        raise ValueError(f"{output_path}: rendered size {(w, h)} does not match required {(width, height)}")
    save_image_array(array, output_path)


def save_scene_renders(rendered_dict, test_poses, scene_output_dir):
    scene_output_dir = Path(scene_output_dir)
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    pose_lookup = {p.image_name: p for p in test_poses}
    for image_name, array in rendered_dict.items():
        pose = pose_lookup[image_name]
        output_path = scene_output_dir / image_name
        save_rendered_image(array, pose.width, pose.height, output_path)
