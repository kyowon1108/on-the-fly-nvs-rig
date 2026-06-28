#!/usr/bin/env python3
"""Compare our EQR->pinhole outputs with COLMAP panorama_sfm-style renders.

This is a CPU-only diagnostic. It reimplements the relevant image sampling from
COLMAP's panorama_sfm.py example and compares each official virtual perspective
view against the precomputed images under pinhole_rig/ob3d_rig.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OTFRIG_ROOT = Path(os.environ.get("OTFRIG_ROOT", REPO_ROOT.parent))


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def official_panorama_rotations() -> list[tuple[str, np.ndarray]]:
    rotations: list[tuple[str, np.ndarray]] = []
    for pitch_deg in (-35.0, 0.0, 35.0):
        yaw_offset_deg = 45.0 if pitch_deg > 0.0 else 0.0
        for yaw_deg in np.linspace(0.0, 360.0, 4, endpoint=False) + yaw_offset_deg:
            rotation = Rotation.from_euler(
                "XY", [-pitch_deg, -yaw_deg], degrees=True
            ).as_matrix()
            rotations.append(
                (f"pitch{pitch_deg:+.0f}_yaw{yaw_deg:03.0f}", rotation)
            )
    return rotations


def render_official_style(
    eqr_path: Path,
    cam_from_pano_rotation: np.ndarray,
    width: int = 400,
    height: int = 400,
    hfov_deg: float = 90.0,
) -> np.ndarray:
    pano = np.asarray(Image.open(eqr_path).convert("RGB"))
    pano_h, pano_w = pano.shape[:2]
    focal = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))

    x, y = np.indices((width, height)).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)
    xy += 0.5

    xy_norm = np.empty_like(xy)
    xy_norm[:, 0] = (xy[:, 0] - width / 2.0) / focal
    xy_norm[:, 1] = (xy[:, 1] - height / 2.0) / focal
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    rays_in_pano = rays @ cam_from_pano_rotation
    rays_in_pano = rays_in_pano.T
    yaw = np.arctan2(rays_in_pano[0], rays_in_pano[2])
    pitch = -np.arctan2(
        rays_in_pano[1], np.linalg.norm(rays_in_pano[[0, 2]], axis=0)
    )
    u = (1.0 + yaw / np.pi) / 2.0 * pano_w
    v = (1.0 - pitch * 2.0 / np.pi) / 2.0 * pano_h

    xy_in_pano = np.stack([u, v], axis=-1).reshape(width, height, 2)
    xy_in_pano = xy_in_pano.astype(np.float32)
    xy_in_pano -= 0.5
    x_coords, y_coords = np.moveaxis(xy_in_pano, [0, 1, 2], [2, 1, 0])
    return cv2.remap(
        pano,
        x_coords,
        y_coords,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def load_view_names(rig_config: Path) -> list[str]:
    raw = json.loads(rig_config.read_text())
    return [cam["name"] for ring in raw for cam in ring["cameras"]]


def compare_one(
    repo_root: Path,
    rig_config: Path,
    scene: str,
    timestep: int,
) -> list[dict[str, str | float | int]]:
    view_names = load_view_names(rig_config)
    eqr_path = repo_root / "raw_eqr/ob3d" / scene / "images" / f"{timestep:05d}_rgb.png"
    ours_root = (
        repo_root
        / "pinhole_rig/ob3d_rig"
        / f"{scene}_100"
        / "images"
    )
    ours = {
        view: np.asarray(
            Image.open(ours_root / view / f"frame_{timestep:06d}.png").convert("RGB")
        )
        for view in view_names
    }

    rows: list[dict[str, str | float | int]] = []
    for official_label, rotation in official_panorama_rotations():
        rendered = render_official_style(eqr_path, rotation)
        scored = []
        for view, image in ours.items():
            abs_diff = np.abs(rendered.astype(np.int16) - image.astype(np.int16))
            scored.append(
                (
                    psnr_uint8(rendered, image),
                    float(abs_diff.mean()),
                    int(abs_diff.max()),
                    view,
                )
            )
        best_psnr, best_mean_abs, best_max_abs, best_view = max(
            scored, key=lambda item: item[0]
        )
        rows.append(
            {
                "scene": scene,
                "timestep": timestep,
                "official_view": official_label,
                "best_ours_view": best_view,
                "psnr_db": best_psnr,
                "mean_abs_rgb": best_mean_abs,
                "max_abs_rgb": best_max_abs,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_OTFRIG_ROOT),
        help="otfrig root containing raw_eqr and pinhole_rig.",
    )
    parser.add_argument(
        "--rig-config",
        default=str(REPO_ROOT / "examples/panoramic_rig/rig12_panosfm.json"),
        help="Blender-style rig config used by our converter.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Sample as scene:timestep. May be passed multiple times.",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    samples = args.sample or [
        "classroom:0",
        "emerald-square:6",
        "sponza:0",
    ]
    rows: list[dict[str, str | float | int]] = []
    for sample in samples:
        scene, timestep_text = sample.split(":", 1)
        rows.extend(
            compare_one(
                Path(args.repo_root),
                Path(args.rig_config),
                scene,
                int(timestep_text),
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
