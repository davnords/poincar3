from __future__ import annotations

import os
import os.path as osp

import cv2
import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image, ImageFile
from tqdm import tqdm

from ..geometry import unproject_depth_map_to_point_map
from .cropping import resize_image, resize_image_depth_and_intrinsic

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
_to_tensor = tvf.ToTensor()

_TEST_SCAN_NUMBERS = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]


def _load_cam_mvsnet(words: str, interval_scale: float = 1) -> tuple[np.ndarray, np.ndarray]:
    """Reads a DTU MVSNet-format camera txt file's contents."""
    words = words.split()
    cam = np.zeros((2, 4, 4))
    for i in range(4):
        for j in range(4):
            cam[0][i][j] = words[4 * i + j + 1]
    for i in range(3):
        for j in range(3):
            cam[1][i][j] = words[3 * i + j + 18]

    extrinsic = cam[0].astype(np.float32)
    intrinsic = cam[1].astype(np.float32)
    return intrinsic, extrinsic


class DTU(torch.utils.data.Dataset):
    def __init__(
        self,
        DTU_DIR: str,
        load_img_size: int = 518,
        patch_multiple: int = 16,
        cache_file: str = "data/dataset_cache/dtu_mv_recon_cache.npy",
    ) -> None:
        self.DTU_DIR = DTU_DIR
        self.load_img_size = load_img_size
        self.patch_multiple = patch_multiple

        if osp.exists(cache_file):
            self.metadata = np.load(cache_file, allow_pickle=True).item()
            self.sequence_list = sorted(self.metadata.keys())
        else:
            self.sequence_list = [f"scan{n}" for n in _TEST_SCAN_NUMBERS]
            self.metadata = {}
            for seq in tqdm(self.sequence_list, desc="[DTU] indexing"):
                rgb_root = osp.join(DTU_DIR, seq, "images")
                all_imgs = sorted(d for d in os.listdir(rgb_root) if d.endswith(".jpg"))
                self.metadata[seq] = len(all_imgs)
            os.makedirs(osp.dirname(cache_file), exist_ok=True)
            np.save(cache_file, self.metadata)

    def __len__(self) -> int:
        return len(self.sequence_list)

    def get_data(self, sequence_name: str, ids: list[int] | np.ndarray) -> dict:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        seq_len = self.metadata[sequence_name]

        image_path = osp.join(self.DTU_DIR, sequence_name, "images")
        depth_path = osp.join(self.DTU_DIR, sequence_name, "depths")
        mask_path = osp.join(self.DTU_DIR, sequence_name, "binary_masks")
        cam_path = osp.join(self.DTU_DIR, sequence_name, "cams")

        image_paths: list[str] = [""] * len(ids)
        images: list[torch.Tensor] = [None] * len(ids)  # type: ignore[list-item]
        depths: list[np.ndarray] = [None] * len(ids)  # type: ignore[list-item]
        extrinsics = np.zeros((len(ids), 3, 4))
        intrinsics = np.zeros((len(ids), 3, 3))

        for id_index, frame_id in enumerate(ids):
            impath = osp.join(image_path, f"{frame_id:08d}.jpg")
            depthpath = osp.join(depth_path, f"{frame_id:08d}.npy")
            campath = osp.join(cam_path, f"{frame_id:08d}_cam.txt")
            maskpath = osp.join(mask_path, f"{frame_id:08d}.png")

            rgb_image = Image.open(impath)
            depthmap = np.load(depthpath)
            rgb_image = resize_image(rgb_image, (depthmap.shape[1], depthmap.shape[0]))
            depthmap = np.nan_to_num(depthmap.astype(np.float32), nan=0.0)

            mask = (cv2.imread(maskpath, cv2.IMREAD_UNCHANGED) / 255.0).astype(np.float32)
            mask[mask > 0.5] = 1.0
            mask[mask < 0.5] = 0.0
            mask = cv2.resize(mask, (depthmap.shape[1], depthmap.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = cv2.erode(mask, np.ones((10, 10), np.uint8), iterations=1)
            depthmap = depthmap * mask

            intrinsic, extrinsic = _load_cam_mvsnet(open(campath, "r").read())
            intrinsic = intrinsic[:3, :3]

            rgb_image, depthmap, intrinsic = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=intrinsic,
                output_width=self.load_img_size,
                patch_multiple=self.patch_multiple,
            )

            image_paths[id_index] = impath
            images[id_index] = _to_tensor(rgb_image)
            depths[id_index] = depthmap
            intrinsics[id_index] = intrinsic
            extrinsics[id_index] = extrinsic[:3, :]

        depths_arr = np.array(depths)  # (S, H, W)
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths_arr[..., None], extrinsics_cam=extrinsics, intrinsics_cam=intrinsics
        )

        return {
            "seq_id": sequence_name,
            "seq_len": seq_len,
            "image_paths": image_paths,
            "images": torch.stack(images, dim=0),
            "pointclouds": pointclouds,  # (S, H, W, 3), numpy
            "valid_mask": depths_arr > 1e-4,  # (S, H, W)
        }
