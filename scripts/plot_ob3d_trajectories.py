#!/usr/bin/env python3
"""Plot OB3D Egocentric vs Non-Egocentric camera-center trajectories.

OB3D camera JSON files store world-to-camera extrinsics. We therefore derive
the camera center as C = -R^T t, matching the rig ATE evaluator convention.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OTFRIG_ROOT = Path(os.environ.get("OTFRIG_ROOT", REPO_ROOT.parent))


@dataclass(frozen=True)
class Trajectory:
    name: str
    root: Path
    frames: list[int]
    centers: np.ndarray
    image_paths: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one OB3D scene's Egocentric and Non-Egocentric trajectories."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_OTFRIG_ROOT / "raw_eqr")
    parser.add_argument("--scene", default="emerald-square")
    parser.add_argument("--ego-subdir", default="ob3d")
    parser.add_argument("--non-ego-subdir", default="ob3d_ne")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ob3d_emerald-square_ego_vs_non_ego_trajectory.png"),
    )
    parser.add_argument("--samples", default="0,50,99", help="Comma-separated frame ids to show.")
    return parser.parse_args()


def frame_id_from_camera_path(path: Path) -> int:
    stem = path.stem
    if not stem.endswith("_cam"):
        raise ValueError(f"Unexpected camera filename: {path.name}")
    return int(stem[: -len("_cam")])


def load_center_from_cam_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Unexpected OB3D camera JSON schema: {path}")
    cam = payload[0]
    rotation = np.asarray(cam["extrinsics"]["rotation"], dtype=np.float64)
    translation = np.asarray(cam["extrinsics"]["translation"], dtype=np.float64)
    return -rotation.T @ translation


def load_trajectory(raw_root: Path, subdir: str, scene: str, name: str) -> Trajectory:
    root = raw_root / subdir / scene
    camera_dir = root / "cameras"
    image_dir = root / "images"
    if not camera_dir.is_dir():
        raise FileNotFoundError(camera_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    cam_paths = sorted(camera_dir.glob("*_cam.json"))
    if not cam_paths:
        raise FileNotFoundError(f"No *_cam.json files under {camera_dir}")

    frames: list[int] = []
    centers: list[np.ndarray] = []
    image_paths: list[Path] = []
    for cam_path in cam_paths:
        frame_id = frame_id_from_camera_path(cam_path)
        image_path = image_dir / f"{frame_id:05d}_rgb.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        frames.append(frame_id)
        centers.append(load_center_from_cam_json(cam_path))
        image_paths.append(image_path)

    if len(frames) != len(set(frames)):
        raise ValueError(f"Duplicate frame ids in {camera_dir}")

    return Trajectory(name=name, root=root, frames=frames, centers=np.stack(centers), image_paths=image_paths)


def load_thumbnail(path: Path, size: tuple[int, int] = (220, 110)) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (255, 255, 255))
    left = (size[0] - image.width) // 2
    top = (size[1] - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def frame_to_path(traj: Trajectory, frame_id: int) -> Path:
    try:
        index = traj.frames.index(frame_id)
    except ValueError as exc:
        raise ValueError(f"Frame {frame_id} is not present in {traj.name}") from exc
    return traj.image_paths[index]


def equalize_xy_axes(ax: plt.Axes, trajectories: list[Trajectory]) -> None:
    all_centers = np.concatenate([traj.centers for traj in trajectories], axis=0)
    x_min, y_min = all_centers[:, 0].min(), all_centers[:, 1].min()
    x_max, y_max = all_centers[:, 0].max(), all_centers[:, 1].max()
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    radius = max(x_max - x_min, y_max - y_min) / 2.0
    margin = max(radius * 0.08, 0.1)
    ax.set_xlim(cx - radius - margin, cx + radius + margin)
    ax.set_ylim(cy - radius - margin, cy + radius + margin)
    ax.set_aspect("equal", adjustable="box")


def summarize(traj: Trajectory) -> dict[str, object]:
    centers = traj.centers
    bbox_min = centers.min(axis=0)
    bbox_max = centers.max(axis=0)
    span = bbox_max - bbox_min
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return {
        "name": traj.name,
        "root": str(traj.root),
        "num_frames": len(traj.frames),
        "frame_min": min(traj.frames),
        "frame_max": max(traj.frames),
        "center_formula": "C = -R^T t from OB3D world-to-camera extrinsics",
        "bbox_min_xyz": bbox_min.tolist(),
        "bbox_max_xyz": bbox_max.tolist(),
        "bbox_span_xyz": span.tolist(),
        "bbox_span_norm": float(np.linalg.norm(span)),
        "mean_step": float(step.mean()) if len(step) else 0.0,
        "median_step": float(np.median(step)) if len(step) else 0.0,
        "max_step": float(step.max()) if len(step) else 0.0,
    }


def main() -> None:
    args = parse_args()
    samples = [int(token.strip()) for token in args.samples.split(",") if token.strip()]

    ego = load_trajectory(args.raw_root, args.ego_subdir, args.scene, "Egocentric")
    non_ego = load_trajectory(args.raw_root, args.non_ego_subdir, args.scene, "Non-Egocentric")
    trajectories = [ego, non_ego]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 6, height_ratios=[1.4, 1.0, 0.95])

    ax_xy = fig.add_subplot(grid[0:2, 0:3])
    colors = {"Egocentric": "#1f77b4", "Non-Egocentric": "#d62728"}
    for traj in trajectories:
        xy = traj.centers[:, :2]
        ax_xy.plot(xy[:, 0], xy[:, 1], "-", color=colors[traj.name], linewidth=2.2, label=traj.name)
        ax_xy.scatter(xy[0, 0], xy[0, 1], color=colors[traj.name], marker="o", s=55)
        ax_xy.scatter(xy[-1, 0], xy[-1, 1], color=colors[traj.name], marker="X", s=70)
        ax_xy.text(xy[0, 0], xy[0, 1], f" {traj.name} start", fontsize=8)
        ax_xy.text(xy[-1, 0], xy[-1, 1], f" {traj.name} end", fontsize=8)
    equalize_xy_axes(ax_xy, trajectories)
    ax_xy.set_title(f"OB3D {args.scene}: top-down camera centers")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(loc="best")

    ax_3d = fig.add_subplot(grid[0:2, 3:6], projection="3d")
    for traj in trajectories:
        c = traj.centers
        ax_3d.plot(c[:, 0], c[:, 1], c[:, 2], color=colors[traj.name], linewidth=2.0, label=traj.name)
        ax_3d.scatter(c[0, 0], c[0, 1], c[0, 2], color=colors[traj.name], marker="o", s=45)
        ax_3d.scatter(c[-1, 0], c[-1, 1], c[-1, 2], color=colors[traj.name], marker="X", s=55)
    ax_3d.set_title("3D center path, C = -R^T t")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.legend(loc="best")
    ax_3d.view_init(elev=23, azim=-60)

    for row, traj in enumerate(trajectories):
        for col, frame_id in enumerate(samples):
            ax = fig.add_subplot(grid[2, row * 3 + col])
            thumb = load_thumbnail(frame_to_path(traj, frame_id))
            ax.imshow(thumb)
            ax.set_title(f"{traj.name}\nframe {frame_id:05d}", fontsize=9)
            ax.axis("off")

    fig.suptitle(
        "Egocentric keeps a small local path; Non-Egocentric traverses a much wider baseline",
        fontsize=14,
    )
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "scene": args.scene,
                "samples": samples,
                "trajectories": [summarize(traj) for traj in trajectories],
                "output_image": str(args.out),
            },
            indent=2,
        )
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
