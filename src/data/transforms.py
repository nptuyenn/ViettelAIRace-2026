import numpy as np
import torch
from PIL import Image


def pil_to_tensor(image):
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_array(tensor):
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(array, 0.0, 1.0)


def resize_pil(image, width, height):
    return image.resize((width, height), Image.LANCZOS)


def downscale_intrinsics(fx, fy, cx, cy, scale):
    return fx * scale, fy * scale, cx * scale, cy * scale
