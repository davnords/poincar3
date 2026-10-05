from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image


def _round_to_patch(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(value / patch_size)) * patch_size)


def get_target_shape(aspect_ratio: float, img_size: int, patch_size: int) -> tuple[int, int]:
    """Target ``(H, W)`` for a given aspect ratio ``H/W``, long side = ``img_size``.

    ``aspect_ratio <= 1`` -> landscape (W is the long side = img_size); ``> 1`` ->
    portrait (H is the long side). The short side is rounded to a patch multiple.
    This spans both orientations in one continuous range, so no separate
    transpose step is needed.
    """
    if aspect_ratio <= 1.0:
        return _round_to_patch(img_size * aspect_ratio, patch_size), img_size
    return img_size, _round_to_patch(img_size / aspect_ratio, patch_size)


def _resize_to_cover(
    image: np.ndarray,
    K: np.ndarray,
    target_hw: tuple[int, int],
    safe_bound: float,
    rescale_aug: bool,
    rng: np.random.Generator,
    depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Resize preserving aspect so the image covers ``target_hw`` (+ margin).

    If ``depth`` is given, it is resized in lockstep (nearest-neighbor, to
    avoid interpolating across depth discontinuities) and returned as the
    third element; otherwise the third element is ``None``.
    """
    th, tw = target_hw
    H, W = image.shape[:2]
    if rescale_aug:
        safe_bound = safe_bound + float(rng.triangular(0.0, 0.0, 0.3)) * max(th, tw)

    scale = max((th + safe_bound) / H, (tw + safe_bound) / W)
    out_h = int(np.floor(H * scale))
    out_w = int(np.floor(W * scale))

    pil = Image.fromarray(image).resize(
        (out_w, out_h), Image.LANCZOS if scale < 1 else Image.BICUBIC
    )
    image = np.asarray(pil)

    if depth is not None:
        depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    actual_scale = max(out_h / H, out_w / W)
    K = K.copy()
    # Account for the 0.5 pixel-center convention when scaling the principal point.
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    K[:2, :] *= actual_scale
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return image, K, depth


def _crop_to_principal_point(
    image: np.ndarray,
    K: np.ndarray,
    target_hw: tuple[int, int],
    depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Crop to exactly ``target_hw`` centered on the principal point (pad if short).

    ``depth``, if given, is cropped/padded identically to ``image``.
    """
    th, tw = target_hw
    H, W = image.shape[:2]
    K = K.copy()
    cx, cy = K[0, 2], K[1, 2]  # cols, rows

    half_h = min(th / 2, cy)
    half_w = min(tw / 2, cx)
    start_r = math.floor(cy) - math.floor(half_h)
    start_c = math.floor(cx) - math.floor(half_w)
    start_r = max(0, start_r)
    start_c = max(0, start_c)

    image = image[start_r : start_r + th, start_c : start_c + tw]
    K[1, 2] -= start_r
    K[0, 2] -= start_c

    # Pad (bottom/right) if the crop came up short -- mirrors VGGT's strict mode.
    ph, pw = th - image.shape[0], tw - image.shape[1]
    if ph > 0 or pw > 0:
        image = np.pad(image, ((0, max(0, ph)), (0, max(0, pw)), (0, 0)), constant_values=0)

    if depth is not None:
        depth = depth[start_r : start_r + th, start_c : start_c + tw]
        dph, dpw = th - depth.shape[0], tw - depth.shape[1]
        if dph > 0 or dpw > 0:
            depth = np.pad(depth, ((0, max(0, dph)), (0, max(0, dpw))), constant_values=0)

    return image, K, depth


def process_frame(
    image: Image.Image,
    K: np.ndarray,
    target_hw: tuple[int, int],
    *,
    rng: np.random.Generator,
    rescale_aug: bool = True,
    safe_bound: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Aspect-preserving resize + principal-point crop to ``target_hw``.

    Returns ``(image_uint8 (H, W, 3), K (3, 3))``.
    """
    arr = np.asarray(image.convert("RGB"))
    arr, K, _ = _resize_to_cover(arr, K, target_hw, safe_bound, rescale_aug, rng)
    arr, K, _ = _crop_to_principal_point(arr, K, target_hw)
    # Contiguous copy: PIL/np views can be read-only, which torch.from_numpy warns on.
    return np.ascontiguousarray(arr), K


def process_frame_with_depth(
    image: Image.Image,
    K: np.ndarray,
    depth: np.ndarray,
    target_hw: tuple[int, int],
    *,
    rng: np.random.Generator,
    rescale_aug: bool = True,
    safe_bound: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as ``process_frame``, but also resizes/crops ``depth`` in lockstep
    (nearest-neighbor) so it stays pixel-aligned with the returned image/K.

    Returns ``(image_uint8 (H, W, 3), K (3, 3), depth_f32 (H, W))``.
    """
    arr = np.asarray(image.convert("RGB"))
    depth = np.asarray(depth, dtype=np.float32)
    arr, K, depth = _resize_to_cover(arr, K, target_hw, safe_bound, rescale_aug, rng, depth)
    arr, K, depth = _crop_to_principal_point(arr, K, target_hw, depth)
    return np.ascontiguousarray(arr), K, np.ascontiguousarray(depth)
