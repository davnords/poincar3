from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

try:
    _LANCZOS = Image.Resampling.LANCZOS
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover -- older Pillow
    _LANCZOS = Image.LANCZOS
    _BICUBIC = Image.BICUBIC


def _round_to_patch_multiple(value: float, patch_multiple: int) -> int:
    return max(patch_multiple, int(round(float(value) / patch_multiple)) * patch_multiple)


def resize_image(image: Image.Image, output_resolution: tuple[int, int]) -> Image.Image:
    max_resize_scale = max(output_resolution[0] / image.size[0], output_resolution[1] / image.size[1])
    return image.resize(output_resolution, resample=_LANCZOS if max_resize_scale < 1 else _BICUBIC)


def resize_image_depth_and_intrinsic(
    image: Image.Image,
    depth_map: np.ndarray,
    intrinsic: np.ndarray,
    output_width: int,
    patch_multiple: int = 16,
    pixel_center: bool = True,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Resizes `image`/`depth_map` to roughly `output_width` wide, with both
    sides rounded to the nearest multiple of `patch_multiple`, adjusting
    `intrinsic` to match (pixel-center convention, i.e. pixel `(0,0)`'s
    center is at `(0.5, 0.5)`).

    The width is rounded too (not just the height): a patch-16 model fed a
    518-wide image drops the leftover 6 columns and predicts a 512-wide
    depth map, which then no longer lines up with the ground-truth point
    cloud. `518` is a no-op under the original port's `patch_multiple=14`."""
    if depth_map.ndim != 2:
        raise ValueError(f"Depth map must be a 2D array, but found depthmap.shape = {depth_map.shape}")
    input_resolution = np.array(depth_map.shape[::-1], dtype=np.float32)  # (H, W) -> (W, H)
    output_width = _round_to_patch_multiple(output_width, patch_multiple)
    output_resolution = np.array(
        [
            output_width,
            _round_to_patch_multiple(input_resolution[1] * (output_width / input_resolution[0]), patch_multiple),
        ]
    )

    image = resize_image(image, tuple(output_resolution))
    depth_map = cv2.resize(depth_map, output_resolution, interpolation=cv2.INTER_NEAREST)

    intrinsic = np.copy(intrinsic)
    if pixel_center:
        intrinsic[0, 2] += 0.5
        intrinsic[1, 2] += 0.5

    resize_scale = np.max(output_resolution / input_resolution)
    intrinsic[:2, :] = intrinsic[:2, :] * resize_scale

    if pixel_center:
        intrinsic[0, 2] -= 0.5
        intrinsic[1, 2] -= 0.5

    assert image.size == depth_map.shape[::-1], f"Image size {image.size} does not match depth map shape {depth_map.shape[::-1]}"
    return image, depth_map, intrinsic
