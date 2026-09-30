from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from PIL import Image

from poincar3 import Poincar3
from poincar3.benchmarks.dense_features import extract_global_attention_qk, get_attention_probe
from poincar3.device import device

PATCH = 16  # ViT-L/16: one token per 16x16 pixels.
COLORS = ("#2a78d6", "#2f9e6b", "#d6572a")  # one per query point

# Row-major fill into the (nrows, ncols) grid. The first view of a cluster is
# the query image and the rest are attention targets, so every entry must be
# the same length. Any replacement has to be genuinely covisible views, not
# just views of the same scene.
VIEWS = (
    ("assets/0002_A.jpg", "assets/0002_B.jpg", "assets/0002_C.jpg"),
    ("assets/0010_A.jpg", "assets/0010_B.jpg", "assets/0010_C.jpg"),
    ("assets/0015_A.jpg", "assets/0015_B.jpg", "assets/0015_C.jpg"),
    ("assets/0016_A.png", "assets/0016_B.png", "assets/0016_C.png"),
)


@dataclass(frozen=True)
class Cfg:
    views: tuple[tuple[str, ...], ...] = VIEWS
    nrows: int = 2
    ncols: int = 2
    num_queries: int = len(COLORS)
    # A pretraining run directory, or `None` for the released checkpoint.
    run_path: str | None = None
    # Every view is resized and center-cropped to this square side, a multiple
    # of PATCH, so every cell in the grid is pixel-for-pixel the same size.
    size: int = 448
    max_keypoints: int = 400
    # Candidates stay this many patches off the border, where a patch has half
    # its receptive field cropped away.
    margin: int = 3
    # A candidate survives only if the round trip view 0 -> view v -> view 0
    # lands within this many patches of where it started, in *every* view.
    max_cycle_error: int = 1
    # No two chosen queries sit within this many patches of each other, or they
    # all pile onto the single most distinctive object.
    min_separation: int = 8
    out_path: str = "demo.png"


def load_views(paths: tuple[str, ...], size: int) -> torch.Tensor:
    """-> (V, 3, size, size) in [0, 1]. Resize short side, then center-crop."""
    views = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        scale = size / min(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
        left, top = (img.width - size) // 2, (img.height - size) // 2
        img = img.crop((left, top, left + size, top + size))
        views.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(views)


def keypoint_candidates(image: torch.Tensor, grid: int, cfg: Cfg) -> np.ndarray:
    """Patches holding a SIFT keypoint, strongest first, one entry per patch.

    SIFT rather than a learned detector on purpose: it needs no extra weights,
    and its keypoints are the corners and blobs a reader recognises as "the
    same point" across views. The response only dedupes and orders patches --
    which candidates actually get drawn is decided by the model below.
    """
    rgb = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    detector = cv2.SIFT_create(nfeatures=cfg.max_keypoints)
    strongest: dict[tuple[int, int], float] = {}
    for kp in detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None):
        row, col = int(kp.pt[1]) // PATCH, int(kp.pt[0]) // PATCH
        if cfg.margin <= row < grid - cfg.margin and cfg.margin <= col < grid - cfg.margin:
            strongest[(row, col)] = max(kp.response, strongest.get((row, col), -np.inf))
    return np.array(sorted(strongest, key=lambda rc: -strongest[rc]), dtype=int).reshape(-1, 2)


@torch.no_grad()
def cross_view_attention(model: Poincar3, views: torch.Tensor, queries: np.ndarray, grid: int):
    """For every candidate patch in view 0, where its cross-view attention peaks
    in each other view, how much mass landed there, and its round-trip error.

    One forward pass per cluster, however many candidates: `q`/`k` are captured
    for the whole `V * num_tokens` sequence, so each extra candidate -- in
    either direction -- is just another row indexed out of that same capture.
    The softmax spans every view at once, exactly as the model itself runs, so
    a query's attention mass is *split* across the target views rather than
    renormalised per pair.
    """
    num_views = views.shape[0]
    probe = get_attention_probe(model)
    q, k, scale = extract_global_attention_qk(probe, views)  # (heads, V * tokens, head_dim)

    # Each view has a few prefix tokens (registers + a camera token) before its
    # patch tokens, so view v's patch (row, col) sits at this flat index.
    start = probe.patch_token_start
    tokens_per_view = start + grid * grid

    def attend_from(token_idx: list[int]) -> torch.Tensor:
        """Head-averaged attention from each given token to the whole sequence."""
        logits = torch.einsum("qhd,hnd->qhn", q[:, token_idx].transpose(0, 1), k) * scale
        return logits.softmax(dim=-1).mean(dim=1)

    def in_view(probs: torch.Tensor, view: int) -> torch.Tensor:
        offset = view * tokens_per_view + start
        return probs[:, offset : offset + grid * grid]

    def tokens_at(view: int, flat: torch.Tensor) -> list[int]:
        return [view * tokens_per_view + start + int(i) for i in flat]

    probs = attend_from(tokens_at(0, queries[:, 0] * grid + queries[:, 1]))
    match = np.zeros((len(queries), num_views - 1, 2), dtype=int)
    peak = np.zeros((len(queries), num_views - 1))
    cycle_error = np.zeros((len(queries), num_views - 1), dtype=int)
    for view in range(1, num_views):
        view_probs = in_view(probs, view)
        flat = view_probs.argmax(dim=-1)
        best = flat.cpu().numpy()
        match[:, view - 1] = np.stack([best // grid, best % grid], axis=1)
        peak[:, view - 1] = view_probs.max(dim=-1).values.cpu().numpy()
        # Round trip: query the matched patch back, and see whether view 0's
        # argmax returns to where this candidate started.
        back = in_view(attend_from(tokens_at(view, flat)), 0).argmax(dim=-1).cpu().numpy()
        back_rc = np.stack([back // grid, back % grid], axis=1)
        cycle_error[:, view - 1] = np.abs(back_rc - queries).max(axis=1)  # Chebyshev, in patches
    return match, peak, cycle_error


def select_queries(queries: np.ndarray, peak: np.ndarray, cycle_error: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Indices of the candidates worth drawing, using only the model itself --
    no ground truth anywhere.

    Cycle consistency does the real work: a query on textureless or
    non-covisible content still produces an argmax somewhere, and drawing it
    yields a confident-looking wrong line, but its round trip lands far from
    where it started. Among survivors, rank by each candidate's *weakest*
    target view: a query is only worth showing if it works everywhere, not just
    in the easiest view.
    """
    consistent = np.flatnonzero(cycle_error.max(axis=1) <= cfg.max_cycle_error)
    if len(consistent) == 0:
        return np.empty(0, dtype=int)
    ranked = consistent[np.argsort(-peak[consistent].min(axis=1))]

    def thin(separation: int) -> list[int]:
        chosen: list[int] = []
        for idx in ranked:
            if len(chosen) == cfg.num_queries:
                break
            if all(np.abs(queries[idx] - queries[j]).max() >= separation for j in chosen):
                chosen.append(int(idx))
        return chosen

    # Try the requested spacing, then relax a patch at a time: a hard
    # `min_separation` on a scene with few consistent candidates otherwise
    # quietly returns two queries where three were asked for, which reads as a
    # half-empty cell rather than a deliberate choice.
    for separation in range(cfg.min_separation, 0, -1):
        chosen = thin(separation)
        if len(chosen) == cfg.num_queries:
            break
    return np.array(chosen, dtype=int)


def draw_cell(fig, cell, views: torch.Tensor, queries: np.ndarray, match: np.ndarray, grid: int) -> None:
    """One grid cell: every view side by side, each query point in view 0 joined
    to its attention argmax in each of the others."""
    num_views, size = views.shape[0], views.shape[-1]
    inner = cell.subgridspec(1, num_views, wspace=0.01)
    axes = [fig.add_subplot(inner[i]) for i in range(num_views)]
    # aspect="auto" lets each axes fill its box exactly; with imshow's default
    # "equal", any mismatch between box and image aspect is padded out as
    # visible whitespace instead of tightening the grid.
    for ax, view in zip(axes, views):
        ax.imshow(view.permute(1, 2, 0).clamp(0, 1).cpu().numpy(), aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def pixel(row: int, col: int) -> tuple[float, float]:
        return (col + 0.5) / grid * size, (row + 0.5) / grid * size

    # Black halo under every line and marker, so a correspondence stays legible
    # on both bright sky and dark interiors.
    outline = [pe.withStroke(linewidth=1.6 + 1.6, foreground="black")]

    for i, (row, col) in enumerate(queries):
        color = COLORS[i % len(COLORS)]
        points = [pixel(row, col)] + [pixel(*match[i, v]) for v in range(num_views - 1)]
        # Only adjacent panels are joined: a view-0-to-view-2 line drawn
        # straight would cross view 1's image and read as a match *there*.
        for ax_a, ax_b, point_a, point_b in zip(axes, axes[1:], points, points[1:]):
            fig.add_artist(
                ConnectionPatch(
                    xyA=point_a, coordsA=ax_a.transData, xyB=point_b, coordsB=ax_b.transData,
                    color=color, linewidth=1.6, zorder=4, path_effects=outline,
                )
            )
        # Markers go on the figure, after the lines: a figure-level artist draws
        # after every axes has finished, so an in-axes scatter would always end
        # up under the ConnectionPatches no matter how high its zorder. Line2D
        # keeps `transform=ax.transData`, so these still land in image pixel
        # coordinates while drawing last, and marker size stays in points.
        for ax, point in zip(axes, points):
            fig.add_artist(
                Line2D(
                    [point[0]], [point[1]], marker="o", markersize=6.0, linestyle="none",
                    color=color, markeredgecolor="black", markeredgewidth=0.8,
                    transform=ax.transData, zorder=6,
                )
            )


def load_model(run_path: str | None) -> Poincar3:
    if run_path is None:
        # No config -> download the released checkpoint, already the trained
        # (teacher) weights.
        return Poincar3().to(device).eval()

    from poincar3.model import SSLModel
    from poincar3.run import load_run

    run_cfg, _, weights, _, _ = load_run(Path(run_path))
    ssl_model = SSLModel(run_cfg.model).to(device)
    ssl_model.load_state_dict(weights)
    return ssl_model.teacher.eval()


def main(cfg: Cfg) -> None:
    assert cfg.size % PATCH == 0, f"--size must be a multiple of {PATCH}"
    assert len(cfg.views) == cfg.nrows * cfg.ncols, f"{len(cfg.views)} clusters for a {cfg.nrows}x{cfg.ncols} grid"
    num_views = len(cfg.views[0])
    assert num_views >= 2 and all(len(v) == num_views for v in cfg.views), "every cluster needs the same view count"

    model = load_model(cfg.run_path)
    grid = cfg.size // PATCH

    fig = plt.figure(figsize=(2.6 * num_views * cfg.ncols, 2.6 * cfg.nrows))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    outer = fig.add_gridspec(cfg.nrows, cfg.ncols, wspace=0.03, hspace=0.03)

    for i, paths in enumerate(cfg.views):
        views = load_views(paths, cfg.size).to(device)
        candidates = keypoint_candidates(views[0], grid, cfg)
        match, peak, cycle_error = cross_view_attention(model, views, candidates, grid)
        keep = select_queries(candidates, peak, cycle_error, cfg)
        share = " ".join(f"{Path(p).stem}={m:.3f}" for p, m in zip(paths[1:], peak[keep].mean(axis=0))) if len(keep) else "-"
        print(f"  {Path(paths[0]).stem}: {len(keep)}/{cfg.num_queries} queries, mean peak  {share}")
        draw_cell(fig, outer[i // cfg.ncols, i % cfg.ncols], views, candidates[keep], match[keep], grid)

    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
