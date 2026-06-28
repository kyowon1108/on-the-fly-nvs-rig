#!/usr/bin/env python3
"""Evaluate native-EQR SfM camera centers on OB3D.

This script is for spherical/EQR SfM baselines, not for OTF virtual-rig runs.
It reads either OpenSfM ``reconstruction.json`` or a COLMAP/SphereSfM
``images.bin`` model, extracts one camera center per original EQR frame, aligns
the estimated trajectory to OB3D GT centers with Sim(3), and reports ATE plus
registration completeness.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.read_write_model import read_images_binary  # noqa: E402


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def parse_source_ts(name: str) -> int:
    match = re.search(r"(\d{5,})", Path(name).name)
    if not match:
        raise ValueError(f"Cannot parse OB3D frame index from {name!r}")
    return int(match.group(1))


def center_from_w2c(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return -rotation.T @ translation


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rodrigues_to_rotmat(rotation: list[float] | np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = r / theta
    kx, ky, kz = k
    skew = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * skew + (1 - math.cos(theta)) * (skew @ skew)


def load_gt_centers_from_raw_scene(raw_scene: Path) -> dict[int, np.ndarray]:
    camera_dir = raw_scene / "cameras"
    if not camera_dir.exists():
        raise FileNotFoundError(f"OB3D camera directory not found: {camera_dir}")
    centers: dict[int, np.ndarray] = {}
    for path in sorted(camera_dir.glob("*_cam.json")):
        ts = parse_source_ts(path.name)
        raw = json.loads(path.read_text())
        meta = raw[0] if isinstance(raw, list) else raw
        extrinsics = meta["extrinsics"]
        rotation = np.asarray(extrinsics["rotation"], dtype=np.float64)
        translation = np.asarray(extrinsics["translation"], dtype=np.float64)
        centers[ts] = center_from_w2c(rotation, translation)
    if not centers:
        raise ValueError(f"No OB3D camera JSON files found under {camera_dir}")
    return centers


def load_gt_centers_json(gt_path: Path) -> dict[int, np.ndarray]:
    raw = json.loads(gt_path.read_text())
    centers: dict[int, np.ndarray] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            item = value["center"] if isinstance(value, dict) else value
            centers[int(key)] = np.asarray(item, dtype=np.float64)
    elif isinstance(raw, list):
        for ts, value in enumerate(raw):
            item = value["center"] if isinstance(value, dict) else value
            centers[ts] = np.asarray(item, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported GT center format: {type(raw)!r}")
    return centers


def load_opensfm_centers(reconstruction_json: Path) -> dict[int, np.ndarray]:
    raw = json.loads(reconstruction_json.read_text())
    if not raw:
        raise ValueError(f"No reconstructions in {reconstruction_json}")
    reconstruction = raw[0]
    centers: dict[int, np.ndarray] = {}
    for name, shot in reconstruction.get("shots", {}).items():
        ts = parse_source_ts(name)
        rotation = rodrigues_to_rotmat(shot["rotation"])
        translation = np.asarray(shot["translation"], dtype=np.float64)
        centers[ts] = center_from_w2c(rotation, translation)
    return centers


def load_colmap_centers(model_path: Path) -> dict[int, np.ndarray]:
    images_path = model_path / "images.bin" if model_path.is_dir() else model_path
    if images_path.name != "images.bin":
        images_path = images_path / "images.bin"
    if not images_path.exists():
        raise FileNotFoundError(f"COLMAP images.bin not found: {images_path}")
    images = read_images_binary(str(images_path))
    centers: dict[int, np.ndarray] = {}
    for image in images.values():
        ts = parse_source_ts(image.name)
        rotation = qvec_to_rotmat(image.qvec)
        translation = np.asarray(image.tvec, dtype=np.float64)
        centers[ts] = center_from_w2c(rotation, translation)
    return centers


def umeyama_sim3(est: np.ndarray, gt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if est.shape != gt.shape or est.ndim != 2 or est.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 trajectories, got {est.shape} and {gt.shape}")
    if len(est) < 3:
        raise ValueError("Need at least three registered frames for Sim3 ATE")
    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    x = est - mu_est
    y = gt - mu_gt
    covariance = (y.T @ x) / len(est)
    u, singular_values, vt = np.linalg.svd(covariance)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1
    rotation = u @ d @ vt
    var_est = np.sum(x * x) / len(est)
    scale = float(np.trace(np.diag(singular_values) @ d) / max(var_est, 1e-12))
    translation = mu_gt - scale * rotation @ mu_est
    return scale, rotation, translation


def evaluate(est_centers: dict[int, np.ndarray], gt_centers: dict[int, np.ndarray]) -> dict[str, Any]:
    expected_ts = sorted(gt_centers)
    registered_ts = sorted(ts for ts in est_centers if ts in gt_centers)
    missing_ts = sorted(set(expected_ts) - set(registered_ts))
    extra_ts = sorted(set(est_centers) - set(expected_ts))
    if len(registered_ts) < 3:
        raise ValueError(f"Need at least 3 registered GT-matched frames, got {len(registered_ts)}")

    est = np.stack([est_centers[ts] for ts in registered_ts], axis=0)
    gt = np.stack([gt_centers[ts] for ts in registered_ts], axis=0)
    scale, rotation, translation = umeyama_sim3(est, gt)
    aligned = (scale * (rotation @ est.T)).T + translation
    errors = np.linalg.norm(gt - aligned, axis=1)

    gt_all = np.stack([gt_centers[ts] for ts in expected_ts], axis=0)
    span = float(np.linalg.norm(gt_all.max(axis=0) - gt_all.min(axis=0)))
    ate_rmse = float(np.sqrt(np.mean(errors**2)))
    return {
        "expected_frames": len(expected_ts),
        "registered_frames": len(registered_ts),
        "registration_recall": float(len(registered_ts) / max(len(expected_ts), 1)),
        "first_registered_ts": int(registered_ts[0]),
        "last_registered_ts": int(registered_ts[-1]),
        "registered_timesteps": registered_ts,
        "missing_timesteps": missing_ts,
        "extra_timesteps": extra_ts,
        "ATE_RMSE_m": ate_rmse,
        "ATE_mean_m": float(np.mean(errors)),
        "ATE_median_m": float(np.median(errors)),
        "ATE_max_m": float(np.max(errors)),
        "GT_bbox_diag_m": span,
        "ATE_RMSE_pct_span": float(100.0 * ate_rmse / max(span, 1e-12)),
        "Sim3_scale_est_to_gt": scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["opensfm", "colmap"], required=True)
    parser.add_argument(
        "--input",
        required=True,
        help="OpenSfM reconstruction.json or COLMAP/SphereSfM model directory/images.bin.",
    )
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument("--raw-scene", help="OB3D raw EQR scene containing cameras/*.json.")
    gt_group.add_argument("--gt-centers", help="gt_centers.json path.")
    parser.add_argument("--output", default="", help="Output JSON path.")
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    input_path = _as_path(args.input)
    if args.method == "opensfm":
        reconstruction_json = input_path / "reconstruction.json" if input_path.is_dir() else input_path
        est_centers = load_opensfm_centers(reconstruction_json)
        input_record = str(reconstruction_json)
    else:
        est_centers = load_colmap_centers(input_path)
        input_record = str(input_path)

    gt_centers = (
        load_gt_centers_from_raw_scene(_as_path(args.raw_scene))
        if args.raw_scene
        else load_gt_centers_json(_as_path(args.gt_centers))
    )
    result = evaluate(est_centers, gt_centers)
    result.update(
        {
            "method": args.method,
            "input": input_record,
            "gt_source": str(_as_path(args.raw_scene or args.gt_centers)),
        }
    )

    output_path = _as_path(args.output) if args.output else input_path.parent / "ate_ob3d_native_eqr.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    if args.fail_on_missing and result["missing_timesteps"]:
        raise SystemExit(f"Missing registered timesteps: {result['missing_timesteps']}")

    print(
        f"{args.method} native-EQR ATE: "
        f"{result['registered_frames']}/{result['expected_frames']} registered, "
        f"ATE={result['ATE_RMSE_m']:.6f} m "
        f"({result['ATE_RMSE_pct_span']:.4f}% span)"
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
