import json
import yaml
import numpy as np
from pathlib import Path
from PIL import Image


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def read_image(path):
    return Image.open(path).convert("RGB")


def read_image_as_array(path):
    return np.array(read_image(path), dtype=np.float32) / 255.0


def save_image_array(array, path):
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).astype(np.uint8)
    ensure_dir(Path(path).parent)
    Image.fromarray(array).save(path)


def resize_image_array(array, width, height):
    img = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    img = img.resize((width, height), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
