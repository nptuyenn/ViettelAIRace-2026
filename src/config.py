import os
from pathlib import Path

IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    DEFAULT_DATA_ROOT = Path("/kaggle/input/datasets/trucnguyen0504/vai-nvs-data/data")
    DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/outputs")
else:
    DEFAULT_DATA_ROOT = Path("./data")
    DEFAULT_OUTPUT_ROOT = Path("./outputs")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))

PUBLIC_SET_ROOT = DATA_ROOT / "public_set"
PRIVATE_TEST_ROOT = DATA_ROOT / "private_set1"
DEFAULT_ROUND2_ROOT = Path("./VAI_NVS_DATA_ROUND2") if Path("./VAI_NVS_DATA_ROUND2").exists() else DATA_ROOT / "round2"
ROUND2_ROOT = Path(os.environ.get("ROUND2_ROOT", DEFAULT_ROUND2_ROOT))
SPLIT_CHOICES = ("public", "private", "round2")

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
RENDERED_DIR = OUTPUT_ROOT / "rendered"
SUBMISSION_ZIP = OUTPUT_ROOT / "submission_round1.zip"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"


def get_split_root(split):
    if split == "public":
        return PUBLIC_SET_ROOT
    if split == "private":
        return PRIVATE_TEST_ROOT
    if split == "round2":
        return ROUND2_ROOT
    raise ValueError(f"Unknown split: {split}")


def get_scene_paths(scene_root, scene_name):
    scene_dir = Path(scene_root) / scene_name
    return {
        "scene_dir": scene_dir,
        "train_images": scene_dir / "train" / "images",
        "train_sparse": scene_dir / "train" / "sparse" / "0",
        "test_images": scene_dir / "test" / "images",
        "test_poses_csv": scene_dir / "test" / "test_poses.csv",
    }


def list_scenes(scene_root):
    scene_root = Path(scene_root)
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene root does not exist: {scene_root}")
    return sorted([p.name for p in scene_root.iterdir() if p.is_dir()])
