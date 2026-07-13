import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TestPose:
    image_name: str
    qvec: np.ndarray
    tvec: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def load_test_poses(csv_path):
    csv_path = Path(csv_path)
    poses = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append(TestPose(
                image_name=row["image_name"],
                qvec=np.array([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])]),
                tvec=np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])]),
                fx=float(row["fx"]),
                fy=float(row["fy"]),
                cx=float(row["cx"]),
                cy=float(row["cy"]),
                width=int(float(row["width"])),
                height=int(float(row["height"])),
            ))
    return poses


def load_all_scenes_test_poses(test_root, scene_names, subpath=("test", "test_poses.csv")):
    result = {}
    for scene_name in scene_names:
        csv_path = Path(test_root) / scene_name / Path(*subpath)
        if csv_path.exists():
            result[scene_name] = load_test_poses(csv_path)
    return result
