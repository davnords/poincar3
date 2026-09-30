from __future__ import annotations

import os
import os.path as osp

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image, ImageFile

from ..geometry import unproject_depth_map_to_point_map
from .cropping import resize_image_depth_and_intrinsic

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
_to_tensor = tvf.ToTensor()


class ETH3D(torch.utils.data.Dataset):
    def __init__(
        self,
        ETH3D_DIR: str,
        load_img_size: int = 518,
        patch_multiple: int = 16,
        cache_file: str = "data/dataset_cache/eth3d_mv_recon_cache.npy",
    ) -> None:
        self.ETH3D_DIR = ETH3D_DIR
        self.load_img_size = load_img_size
        self.patch_multiple = patch_multiple

        if osp.exists(cache_file):
            self.metadata = np.load(cache_file, allow_pickle=True).item()
            self.sequence_list = sorted(self.metadata.keys())
        else:
            self.sequence_list = sorted(
                seq for seq in os.listdir(ETH3D_DIR) if os.path.isdir(osp.join(ETH3D_DIR, seq))
            )
            self.metadata = {}
            for seq in self.sequence_list:
                seq_image_root = osp.join(ETH3D_DIR, seq, "images", "custom_undistorted")
                self.metadata[seq] = sorted(f for f in os.listdir(seq_image_root) if f.endswith(".JPG"))
            os.makedirs(osp.dirname(cache_file), exist_ok=True)
            np.save(cache_file, self.metadata)

    def __len__(self) -> int:
        return len(self.sequence_list)

    def get_data(self, sequence_name: str, ids: list[int] | np.ndarray) -> dict:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        image_list = self.metadata[sequence_name]
        seq_len = len(image_list)

        image_paths: list[str] = [""] * len(ids)
        images: list[torch.Tensor] = [None] * len(ids)  # type: ignore[list-item]
        depths: list[np.ndarray] = [None] * len(ids)  # type: ignore[list-item]
        extrinsics = np.zeros((len(ids), 3, 4))
        intrinsics = np.zeros((len(ids), 3, 3))

        for id_index, frame_id in enumerate(ids):
            img_name = image_list[frame_id]
            impath = osp.join(self.ETH3D_DIR, sequence_name, "images", "custom_undistorted", img_name)
            depthpath = osp.join(self.ETH3D_DIR, sequence_name, "ground_truth_depth", "custom_undistorted", img_name)
            campath = osp.join(self.ETH3D_DIR, sequence_name, "custom_undistorted_cam", img_name.replace("JPG", "npz"))

            cam = np.load(campath)
            intrinsic, extrinsic = cam["intrinsics"], cam["extrinsics"]

            rgb_image = Image.open(impath)
            width, height = rgb_image.size
            depthmap = np.fromfile(depthpath, dtype=np.float32).reshape(height, width)
            depthmap[~np.isfinite(depthmap)] = -1

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
            "pointclouds": pointclouds,
            "valid_mask": depths_arr > 1e-4,
        }
