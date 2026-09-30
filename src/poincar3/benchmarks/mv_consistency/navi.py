from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as transform_F
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode


def read_image(image_path: Path, exif_transpose: bool = True) -> Image.Image:
    """Reads a NAVI image (and rotates it according to the metadata)."""
    with open(image_path, "rb") as f:
        with Image.open(f) as image:
            if exif_transpose:
                image = ImageOps.exif_transpose(image)
            image.convert("RGB")
            return image


def read_depth(path: str, scale_factor: float = 10.0) -> np.ndarray:
    depth_image = Image.open(path)
    max_val = (2**16) - 1
    disparity = np.array(depth_image).astype("uint16")
    disparity = disparity.astype(np.float32) / (max_val * scale_factor)
    disparity[disparity == 0] = np.inf
    return 1 / disparity


def bbox_crop(
    image: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crops image/depth to the valid-depth bounding box and shifts intrinsics
    to match. `image`: (C,H,W), `depth`: (1,H,W), `intrinsics`: (3,3).
    """
    mask = depth > 0
    if not mask.any():
        return image, depth, intrinsics

    mask_coords = mask.nonzero()
    tl_coord = mask_coords.min(dim=0).values[1:]
    br_coord = mask_coords.max(dim=0).values[1:]

    box_size = br_coord - tl_coord
    img_size = torch.tensor(mask.shape[1:], device=image.device)
    max_dim = box_size.max()

    pad_size = max_dim - box_size
    tl_cent = tl_coord - pad_size // 2

    tl_final = tl_cent.clip(min=0)
    br_final = (tl_final + max_dim).clip(max=img_size)
    tl_final = (br_final - max_dim).clip(min=0)

    x_start, y_start = tl_final[0], tl_final[1]
    x_end, y_end = br_final[0], br_final[1]

    image_cropped = image[:, x_start:x_end, y_start:y_end]
    depth_cropped = depth[:, x_start:x_end, y_start:y_end]

    intrinsics_cropped = intrinsics.clone()
    intrinsics_cropped[0, 2] -= y_start  # cx' = cx - u_min
    intrinsics_cropped[1, 2] -= x_start  # cy' = cy - v_min

    return image_cropped, depth_cropped, intrinsics_cropped


def translate(v: torch.Tensor) -> torch.Tensor:
    """Homogeneous translation matrix from a translation vector, `float32[N]` -> `float32[N+1,N+1]`."""
    result = torch.as_tensor(v, dtype=torch.float32)
    dimensions = result.shape[-1]
    result = result[..., None, :].transpose(-1, -2)
    result = torch.constant_pad_nd(result, [dimensions, 0, 0, 1])
    id_matrix = torch.diag(result.new_ones([dimensions + 1]))
    id_matrix = id_matrix.expand_as(result)
    return result + id_matrix


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrix from a quaternion, `float32[4]` -> `float32[4,4]`."""
    q = torch.as_tensor(q, dtype=torch.float32)
    w, x, y, z = torch.unbind(q, dim=-1)
    ZERO = torch.zeros_like(z)
    ONES = torch.ones_like(z)
    s = 2.0 / (q * q).sum(dim=-1)
    R_00 = 1 - s * (y**2 + z**2)
    R_01 = s * (x * y - z * w)
    R_02 = s * (x * z + y * w)
    R_10 = s * (x * y + z * w)
    R_11 = 1 - s * (x**2 + z**2)
    R_12 = s * (y * z - x * w)
    R_20 = s * (x * z - y * w)
    R_21 = s * (y * z + x * w)
    R_22 = 1 - s * (x**2 + y**2)
    rotation = torch.stack(
        [R_00, R_01, R_02, ZERO, R_10, R_11, R_12, ZERO, R_20, R_21, R_22, ZERO, ZERO, ZERO, ZERO, ONES],
        dim=-1,
    )
    return rotation.reshape(q.shape[:-1] + (4, 4))


def camera_matrices_from_annotation(annotation: dict) -> torch.Tensor:
    """Converts a NAVI annotation's camera pose to a 4x4 matrix (world-to-camera)."""
    translation = translate(annotation["camera"]["t"])
    rotation = quaternion_to_rotation_matrix(annotation["camera"]["q"])
    return translation @ rotation


def _build_transforms(image_mean: str, image_size: tuple[int, int]):
    if image_mean == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    elif image_mean == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif image_mean == "None":
        # Poincar3.forward already normalizes internally -- keep raw [0,1] here.
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    else:
        raise ValueError(f"Unknown image_mean {image_mean!r}")

    image_transform = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=mean, std=std),
            tv_transforms.Resize(min(image_size), interpolation=InterpolationMode.NEAREST),
            tv_transforms.CenterCrop(min(image_size)),
        ]
    )
    target_transform = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Resize(min(image_size), interpolation=InterpolationMode.NEAREST),
            tv_transforms.CenterCrop(min(image_size)),
        ]
    )
    return image_transform, target_transform


class NAVI(torch.utils.data.Dataset):
    """Clusters of `num_views` nearest-in-rotation frames of one NAVI object/scene."""

    def __init__(
        self,
        path: str,
        split: str = "valid",
        model: str = "all",
        image_mean: str = "None",
        num_views: int = 8,
    ) -> None:
        super().__init__()
        if split == "train":
            collection, subpart = "multiview", "train"
        elif split == "valid":
            collection, subpart = "multiview", "test"
        elif split == "trainval":
            collection, subpart = "multiview", "all"
        elif split == "test":
            collection, subpart = "wild", "all"
        else:
            raise ValueError(f"Unknown split: {split}")

        self.data_root = Path(path)
        self.image_size = (512, 512)

        self.data_dict = self.parse_dataset()
        self.define_instances_split(model, collection, subpart)
        self.image_transform, self.target_transform = _build_transforms(image_mean, self.image_size)

        self.num_views = num_views
        self.instance_clusters = self.generate_instance_clusters(self.instances, num_views)

    def __len__(self) -> int:
        return len(self.instance_clusters)

    def __getitem__(self, index: int) -> dict:
        cluster_info = self.instance_clusters[index]
        all_data = [self.get_single(*ids) for ids in cluster_info]
        batch = {k: [d[k] for d in all_data] for k in all_data[0]}
        for key in batch:
            if isinstance(batch[key][0], torch.Tensor):
                batch[key] = torch.stack(batch[key], dim=0)
        return batch

    def generate_instance_clusters(self, instances, num_views: int) -> list:
        """Build, per object/scene, clusters of `num_views` frames closest in rotation."""
        inst_dict: dict = {}
        for obj_id, coll_id, img_id in instances:
            inst_dict.setdefault(obj_id, {}).setdefault(coll_id, []).append(img_id)

        clusters = []
        for obj_id in inst_dict:
            for col_id in inst_dict[obj_id]:
                scene_img_ids = inst_dict[obj_id][col_id]
                if len(scene_img_ids) < num_views:
                    continue
                rots = []
                for img_id in scene_img_ids:
                    anno = self.data_dict[obj_id][col_id]["annotations"][img_id]
                    Rt = camera_matrices_from_annotation(anno)
                    rots.append(Rt[:3, :3])
                rots = torch.stack(rots, dim=0)

                for i, _ in enumerate(scene_img_ids):
                    rot_i = rots[i, None].repeat(len(rots), 1, 1)
                    rots_ij = torch.bmm(rot_i, rots.permute(0, 2, 1))
                    rots_tr = torch.einsum("bii->b", rots_ij)
                    rel_ang_rad = (0.5 * rots_tr - 0.5).clamp(min=-1, max=1).acos()
                    _, nearest_indices = torch.topk(rel_ang_rad, k=num_views, largest=False)
                    cluster_img_ids = [scene_img_ids[j] for j in nearest_indices]
                    clusters.append([(obj_id, col_id, img_id) for img_id in cluster_img_ids])
        return clusters

    def get_single(self, obj_id: str, scene_id: str, img_id: str) -> dict:
        anno = self.data_dict[obj_id][scene_id]["annotations"][img_id]
        scene_path = self.data_root / obj_id / scene_id
        image_path = scene_path / f"images/{img_id}.jpg"
        depth_path = scene_path / f"depth/{img_id}.png"

        image = read_image(image_path)
        image = self.image_transform(image)

        depth = read_depth(str(depth_path)) / 1000  # NAVI convention: mm -> m
        min_depth = depth[depth > 0].min()
        depth = self.target_transform(depth)

        orig_h, orig_w = anno["image_size"]
        image_h, image_w = image.shape[1:]
        orig_fx = anno["camera"]["focal_length"]
        aug_fx = orig_fx * min(image_h, image_w) / min(orig_h, orig_w)

        intrinsics = torch.eye(3)
        intrinsics[0, 0] = aug_fx
        intrinsics[1, 1] = aug_fx
        intrinsics[0, 2] = image_w / 2.0
        intrinsics[1, 2] = image_h / 2.0

        image, depth, intrinsics_after_crop = bbox_crop(image, depth, intrinsics)
        bbox_h, bbox_w = image.shape[1:]
        final_h, final_w = self.image_size

        # Resize the (variable-size) bbox crop back to (final_h, final_w).
        # Nearest interpolation to avoid inventing depth values at edges.
        image = transform_F.resize(image, [final_h, final_w], interpolation=InterpolationMode.NEAREST)
        depth = transform_F.resize(depth, [final_h, final_w], interpolation=InterpolationMode.NEAREST)

        scale_w, scale_h = final_w / bbox_w, final_h / bbox_h
        intrinsics_final = intrinsics_after_crop.clone()
        intrinsics_final[0, 0] *= scale_w
        intrinsics_final[1, 1] *= scale_h
        intrinsics_final[0, 2] *= scale_w
        intrinsics_final[1, 2] *= scale_h

        depth[depth < min_depth] = 0

        Rt = camera_matrices_from_annotation(anno)  # NAVI pose is world-to-camera.
        Rt[:3, 3] = Rt[:3, 3] / 1000.0

        mask = (depth > 0).float()
        return {"image": image, "depth": depth, "intrinsics": intrinsics_final, "Rt": Rt, "mask": mask}

    def parse_dataset(self) -> dict:
        """Parses `<object_id>/<collection>/` folders into a nested dict of instances."""
        data_dict: dict = {}
        all_collections = []
        all_collections += sorted(glob.glob(str(self.data_root / "*/multiview_*")))
        all_collections += sorted(glob.glob(str(self.data_root / "*/wild_set")))

        for collection_path in all_collections:
            object_id, collection_id = collection_path.split("/")[-2:]
            img_files = sorted(os.listdir(os.path.join(collection_path, "images")))
            img_ids = [f.split(".")[0] for f in img_files if "jpg" in f]
            img_ids = [i for i in img_ids if "_" not in i]  # drop "small_" thumbnails

            with open(os.path.join(collection_path, "annotations.json")) as f:
                annotations = json.load(f)
                annotations = {a["filename"].split(".")[0]: a for a in annotations}

            data_dict.setdefault(object_id, {})[collection_id] = {
                "views": img_ids,
                "annotations": annotations,
            }
        return data_dict

    def define_instances_split(self, model: str, collection: str, subpart: str) -> None:
        object_names = list(self.data_dict.keys()) if model == "all" else [model]
        assert collection in ["multiview", "wild"]
        assert subpart in ["train", "test", "all"]

        self.instances = []
        for obj_id in object_names:
            scenes = sorted(self.data_dict[obj_id].keys())
            if "wild_set" not in scenes or len(scenes) == 1:
                continue

            if collection == "wild":
                image_ids = sorted(self.data_dict[obj_id]["wild_set"]["views"])
                image_ann = self.data_dict[obj_id]["wild_set"]["annotations"]
                for _id in image_ids:
                    if subpart == "all":
                        self.instances.append((obj_id, "wild_set", _id))
                    else:
                        im_split = image_ann[_id]["split"]
                        if subpart == "train" and im_split == "train":
                            self.instances.append((obj_id, "wild_set", _id))
                        elif subpart == "test" and im_split == "val":
                            self.instances.append((obj_id, "wild_set", _id))
            else:
                scenes = sorted(s for s in scenes if "multiview" in s)
                train_split = int(0.9 * len(scenes))
                if subpart == "train":
                    scenes = scenes[:train_split]
                elif subpart == "test":
                    scenes = scenes[train_split:]
                for scene in scenes:
                    image_ids = sorted(self.data_dict[obj_id][scene]["views"])
                    self.instances.extend((obj_id, scene, _id) for _id in image_ids)
