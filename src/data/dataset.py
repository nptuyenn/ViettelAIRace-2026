import struct
import numpy as np
from pathlib import Path
from collections import namedtuple
from PIL import Image

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
ImageEntry = namedtuple("ImageEntry", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5,
    "OPENCV": 8, "OPENCV_FISHEYE": 8, "FULL_OPENCV": 12, "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4, "RADIAL_FISHEYE": 5, "THIN_PRISM_FISHEYE": 12,
}
CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
    4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE",
}


def _read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
    ])


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            props = _read_next_bytes(fid, 24, "iiQQ")
            camera_id, model_id, width, height = props
            model_name = CAMERA_MODEL_IDS[model_id]
            num_params = CAMERA_MODEL_NUM_PARAMS[model_name]
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(camera_id, model_name, width, height, np.array(params))
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = ""
            c = fid.read(1)
            while c != b"\x00":
                name += c.decode("utf-8")
                c = fid.read(1)
            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            data = _read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([data[0::3], data[1::3]])
            point3D_ids = np.array(data[2::3])
            images[image_id] = ImageEntry(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)
    return images


def read_points3D_binary(path):
    points = {}
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read_next_bytes(fid, 43, "QdddBBBd")
            point_id = props[0]
            xyz = np.array(props[1:4])
            rgb = np.array(props[4:7])
            track_length = _read_next_bytes(fid, 8, "Q")[0]
            _read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            points[point_id] = {"xyz": xyz, "rgb": rgb}
    return points


def get_intrinsics(camera):
    if camera.model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = camera.params[0], camera.params[1], camera.params[2], camera.params[3]
    elif camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"):
        fx = fy = camera.params[0]
        cx, cy = camera.params[1], camera.params[2]
    else:
        raise NotImplementedError(camera.model)
    return fx, fy, cx, cy


class SceneTrainData:
    def __init__(self, images_dir, sparse_dir):
        self.images_dir = Path(images_dir)
        self.sparse_dir = Path(sparse_dir)
        self.colmap_cameras = read_cameras_binary(self.sparse_dir / "cameras.bin")
        self.colmap_images = read_images_binary(self.sparse_dir / "images.bin")
        points3D_path = self.sparse_dir / "points3D.bin"
        self.points3D = read_points3D_binary(points3D_path) if points3D_path.exists() else {}
        self.cameras_list = self._build_camera_list()

    def _build_camera_list(self):
        cam_list = []
        for image_id, image in self.colmap_images.items():
            image_path = self.images_dir / image.name
            if not image_path.exists():
                continue
            camera = self.colmap_cameras[image.camera_id]
            fx, fy, cx, cy = get_intrinsics(camera)
            R = qvec2rotmat(image.qvec)
            t = image.tvec
            cam_list.append({
                "image_id": image_id,
                "image_name": image.name,
                "image_path": self.images_dir / image.name,
                "width": camera.width,
                "height": camera.height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "R": R, "t": t,
            })
        cam_list.sort(key=lambda c: c["image_name"])
        return cam_list

    def get_point_cloud(self):
        if len(self.points3D) == 0:
            return None, None
        xyz = np.stack([p["xyz"] for p in self.points3D.values()])
        rgb = np.stack([p["rgb"] for p in self.points3D.values()]) / 255.0
        return xyz.astype(np.float32), rgb.astype(np.float32)

    def __len__(self):
        return len(self.cameras_list)

    def load_image(self, index):
        path = self.cameras_list[index]["image_path"]
        return Image.open(path).convert("RGB")

    def get_camera(self, index):
        return self.cameras_list[index]
