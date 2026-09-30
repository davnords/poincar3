from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro


def build_eth3d(data_dir: str, out_path: str) -> None:
    annotation: dict[str, list[list[dict]]] = {}
    for scene_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        cam_dir = scene_dir / "custom_undistorted_cam"
        if not cam_dir.exists():
            continue
        frames = []
        for cam_path in sorted(cam_dir.glob("*.npz")):
            img_name = cam_path.stem + ".JPG"
            filepath = f"{scene_dir.name}/images/custom_undistorted/{img_name}"
            if not (scene_dir / "images" / "custom_undistorted" / img_name).exists():
                continue
            extri = np.load(cam_path)["extrinsics"][:3, :].tolist()  # (4,4) w2c -> (3,4)
            frames.append({"filepath": filepath, "extri": extri})
        if len(frames) >= 2:  # need at least one pair
            annotation[scene_dir.name] = [frames]  # one sequence spanning the whole scene

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(annotation, f)
    print(f"Wrote {len(annotation)} scenes to {out}")


def build_co3d(anno_dir: str, split: str, out_path: str) -> None:
    ann_dir = Path(anno_dir)
    suffix = f"_{split}.jgz"
    annotation: dict[str, list[list[dict]]] = {}
    for path in sorted(ann_dir.glob(f"*{suffix}")):
        category = path.name[: -len(suffix)]
        with gzip.open(path, "rt") as f:
            objects: dict[str, list[dict]] = json.load(f)
        annotation[category] = [
            [{"filepath": frame["filepath"], "extri": frame["extri"]} for frame in frames]
            for frames in objects.values()
        ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(annotation, f)
    print(f"Wrote {len(annotation)} categories to {out}")


def build_scannet(data_dir: str, out_path: str) -> None:
    annotation: dict[str, list[list[dict]]] = {}
    for scene_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir() and p.name.startswith("scene")):
        pose_dir = scene_dir / "pose"
        if not pose_dir.exists():
            continue
        frames = []
        for pose_path in sorted(pose_dir.glob("*.txt"), key=lambda p: int(p.stem)):
            img_path = scene_dir / "color" / f"{pose_path.stem}.jpg"
            if not img_path.exists():
                continue
            c2w = np.loadtxt(pose_path)
            w2c = np.linalg.inv(c2w)[:3, :].tolist()
            frames.append({"filepath": f"{scene_dir.name}/color/{pose_path.stem}.jpg", "extri": w2c})
        if len(frames) >= 2:  # need at least one pair
            annotation[scene_dir.name] = [frames]  # one sequence spanning the whole scene

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(annotation, f)
    print(f"Wrote {len(annotation)} scenes to {out}")


# ScanNet++ (50 held-out `nvs_sem_val` / `nvs_test` scenes). The annotation
# this writes reproduces the reference protocol under
# `RelposeBenchmark` -- same scene list (straight from the split file), same
# frame set (`frames` + `test_frames`, minus `is_bad`), same pose convention,
# and one sequence per scene, which `RelposeBenchmark` then samples
# `num_frames` from uniformly just as that benchmark's `process_sequence`
# does (`np.random.choice(len(metadata), num_frames, replace=False)`).
#
# Deliberately *not* built from ScanNet++'s precomputed pairwise overlaps
# (`overlaps/<scene>.npy`), even though this is the one dataset here that has
# them and `poincar3.data.ffreconstruction.scannetpp` reads them for training:
# the reference benchmark ignores them, and matching it is what keeps the
# resulting AUC comparable to externally reported numbers.
#
# ScanNet++ is also in the ffrecon *training* mixture (`split="train"`, the
# 856 `nvs_sem_train` scenes). val/test are scene-disjoint from that, so there
# is no leakage, but this measures fit *within* the training distribution and
# complements the out-of-domain ETH3D/ScanNet benchmarks rather than replacing
# them.

# nerfstudio stores `transform_matrix` as camera-to-world in OpenGL axes;
# `RelposeAnnotationDataset` wants cam-from-world in OpenCV/COLMAP axes. Same
# flip as `poincar3.data.ffreconstruction.sequence`'s `_GL_TO_COLMAP`, so the
# GT here matches both the reference benchmark and our training convention.
_GL_TO_COLMAP = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)

_SCANNETPP_SPLIT_FILES = {"val": "nvs_sem_val.txt", "test": "nvs_test.txt"}

_SCANNETPP_DEFAULT_OUT = {
    "reference": "data/scannet++/data_download/scannetpp/annotations/relpose_val.jgz",
    "local": "data/scannet++/data_download/scannetpp/annotations/relpose_val_local.jgz",
}

# `local` mode only: how much camera-centre separation counts against
# viewing-direction similarity when ranking a scene's frames by proximity to a
# seed. 30 means "one median inter-camera distance is worth 30 degrees of
# viewing-direction difference" -- a scale-free way to prefer frames that both
# look the same way *and* stand near the seed, rather than either alone.
_SCANNETPP_DIST_WEIGHT_DEG = 30.0


def _scannetpp_scene_frames(root: Path, scene: str) -> list[dict]:
    """Every non-bad frame of `scene` as a `RelposeAnnotationDataset` frame
    dict. `frames` + `test_frames` (the held-out NVS query views) and the
    `is_bad` filter all match the reference benchmark's `_scene_metadata`."""
    meta_path = root / scene / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    frames = []
    for frame in meta["frames"] + meta.get("test_frames", []):
        if frame.get("is_bad", False):
            continue
        w2c = np.linalg.inv(np.array(frame["transform_matrix"], dtype=np.float64) @ _GL_TO_COLMAP)
        frames.append(
            {
                "filepath": f"{scene}/dslr/resized_undistorted_images/{frame['file_path']}",
                "extri": w2c[:3, :4].tolist(),
            }
        )
    return frames


def _scannetpp_local_sequences(
    frames: list[dict], num_frames: int, sequences_per_scene: int, pool: int, rng: np.random.Generator
) -> list[list[int]]:
    """Short-baseline sequences: from a random seed frame, rank the scene by
    viewing-direction angle plus normalized camera-centre distance to that
    seed, then draw `num_frames` at random from the nearest `num_frames *
    pool`. `pool` is the difficulty dial -- 1 would take the seed's very
    closest neighbours, larger values reach further out.
    """
    extri = np.array([f["extri"] for f in frames], dtype=np.float64)
    rot, trans = extri[:, :3, :3], extri[:, :3, 3]
    centers = np.einsum("nij,nj->ni", rot.transpose(0, 2, 1), -trans)
    forward = rot[:, 2, :]  # camera viewing direction in world coordinates
    keep = min(len(frames), num_frames * pool)

    sequences = []
    for _ in range(sequences_per_scene):
        seed = int(rng.integers(len(frames)))
        angle = np.degrees(np.arccos(np.clip(forward @ forward[seed], -1.0, 1.0)))
        distance = np.linalg.norm(centers - centers[seed], axis=1)
        score = angle + _SCANNETPP_DIST_WEIGHT_DEG * distance / (np.median(distance) + 1e-9)
        candidates = np.argsort(score)[:keep]
        sequences.append([int(i) for i in rng.choice(candidates, min(num_frames, len(candidates)), replace=False)])
    return sequences


def build_scannetpp(
    data_root: str,
    split: str,
    out_path: str,
    mode: str,
    num_frames: int,
    sequences_per_scene: int,
    pool: int,
    seed: int,
) -> None:
    root = Path(data_root)
    scenes = [s for s in (root / ".." / "splits" / _SCANNETPP_SPLIT_FILES[split]).read_text().splitlines() if s]
    rng = np.random.default_rng(seed)

    annotation: dict[str, list[list[dict]]] = {}
    for scene in scenes:
        frames = _scannetpp_scene_frames(root, scene)
        if mode == "reference":
            if len(frames) >= 2:  # need at least one pair
                annotation[scene] = [frames]  # one sequence spanning the whole scene
        elif len(frames) >= num_frames:
            annotation[scene] = [
                [frames[i] for i in indices]
                for indices in _scannetpp_local_sequences(frames, num_frames, sequences_per_scene, pool, rng)
            ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(annotation, f)
    total = sum(len(v) for v in annotation.values())
    print(f"Wrote {len(annotation)} scenes / {total} sequences ({mode} mode) to {out}")
    if mode == "local":
        print(
            "NOTE: `local` sequences are selected using the GT poses, so the resulting AUC is a "
            "within-suite signal -- comparable across our own runs, not to outside numbers. "
            "Evaluate with `--relpose.dataset scannetpp` (the whole-scene build is `scannetpp_reference`)."
        )


@dataclass(frozen=True)
class Cfg:
    which: Literal["eth3d", "co3d", "scannet", "scannetpp"]

    eth3d_data_dir: str = "data/eth3d"
    eth3d_out_path: str = "data/eth3d/annotations/test.jgz"

    co3d_anno_dir: str = "data/CO3D/annotations/co3dv2"
    co3d_split: str = "test"
    co3d_out_path: str = "data/CO3D/annotations/co3dv2/relpose_test.jgz"

    scannet_data_dir: str = "data/scannet_test_1500"
    scannet_out_path: str = "data/scannet_test_1500/annotations/test.jgz"

    scannetpp_data_root: str = "data/scannet++/data_download/scannetpp/data"
    # The 50-scene `nvs_sem_val`/`nvs_test` splits, both scene-disjoint from
    # the `train` split the ffrecon mixture trains on. Same two the reference
    # `ScanNetPlusPlusBenchmark.Cfg` offers.
    scannetpp_split: Literal["val", "test"] = "val"
    # None -> `relpose_val.jgz` for `reference`, `relpose_val_local.jgz` for
    # `local`, so the two modes can never overwrite each other's file.
    scannetpp_out_path: str | None = None
    # `local` (default): short-baseline sequences, each drawn from one seed
    # frame's nearest neighbours by viewing direction and camera position.
    # Median GT relative rotation ~17 degrees vs `reference`'s 88, so AUC
    # comes out far higher and training runs separate more cleanly. The
    # selection reads the GT poses, part of what the metric scores, so the
    # result is a within-suite signal rather than a portable benchmark number.
    #
    # `reference`: one whole-scene sequence per scene, the protocol described
    # above.
    scannetpp_mode: Literal["local", "reference"] = "local"
    # `local` mode only (ignored by `reference`).
    scannetpp_num_frames: int = 10
    scannetpp_sequences_per_scene: int = 10
    # Candidate pool as a multiple of `scannetpp_num_frames` -- the difficulty
    # dial. 3 gives a median GT relative rotation of ~17 degrees; 2 is harder
    # to beat still (~13), 5 is closer to MegaDepth (~22).
    scannetpp_pool: int = 3
    scannetpp_seed: int = 0


def main(cfg: Cfg) -> None:
    if cfg.which == "eth3d":
        build_eth3d(cfg.eth3d_data_dir, cfg.eth3d_out_path)
    elif cfg.which == "co3d":
        build_co3d(cfg.co3d_anno_dir, cfg.co3d_split, cfg.co3d_out_path)
    elif cfg.which == "scannetpp":
        build_scannetpp(
            cfg.scannetpp_data_root,
            cfg.scannetpp_split,
            cfg.scannetpp_out_path or _SCANNETPP_DEFAULT_OUT[cfg.scannetpp_mode],
            cfg.scannetpp_mode,
            cfg.scannetpp_num_frames,
            cfg.scannetpp_sequences_per_scene,
            cfg.scannetpp_pool,
            cfg.scannetpp_seed,
        )
    else:
        build_scannet(cfg.scannet_data_dir, cfg.scannet_out_path)


if __name__ == "__main__":
    main(tyro.cli(Cfg))
