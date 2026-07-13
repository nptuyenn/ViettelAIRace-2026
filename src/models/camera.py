import numpy as np
import torch


def build_intrinsics_matrix(fx, fy, cx, cy):
    K = torch.zeros(3, 3, dtype=torch.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1.0
    return K


def build_viewmat(R, t):
    viewmat = torch.eye(4, dtype=torch.float32)
    viewmat[:3, :3] = torch.from_numpy(R).float()
    viewmat[:3, 3] = torch.from_numpy(t).float()
    return viewmat


def qvec_tvec_to_viewmat(qvec, tvec):
    w, x, y, z = qvec
    R = np.array([
        [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
    ])
    return build_viewmat(R, tvec)


class CameraBatch:
    def __init__(self, cameras, device):
        self.device = device
        self.viewmats = torch.stack([
            build_viewmat(c["R"], c["t"]) for c in cameras
        ]).to(device)
        self.Ks = torch.stack([
            build_intrinsics_matrix(c["fx"], c["fy"], c["cx"], c["cy"]) for c in cameras
        ]).to(device)
        self.widths = [c["width"] for c in cameras]
        self.heights = [c["height"] for c in cameras]
        self.names = [c["image_name"] for c in cameras]

    def __len__(self):
        return len(self.names)

    def get(self, index):
        return (
            self.viewmats[index:index + 1],
            self.Ks[index:index + 1],
            self.widths[index],
            self.heights[index],
            self.names[index],
        )
