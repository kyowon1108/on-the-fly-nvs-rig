#!/usr/bin/env python3
"""Evaluate OB3D rig-center ATE from an OTF rig run.

The rig convention has N keyframes per timestep but only one shared rig center.
This script groups metadata keyframes by timestep, averages their estimated
centers, checks same-timestep center spread, aligns the trajectory to OB3D GT
centers with Sim(3) Umeyama, and reports ATE in meters plus percent scene span.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


_TS_RE = re.compile(r"^(?P<view>.+)__ts(?P<ts>\d+)(?:\.[A-Za-z0-9]+)?$")


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def center_from_w2c(rt: Any) -> np.ndarray:
    """Camera center C = -R^T t for a world-to-camera transform."""
    mat = np.asarray(rt, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 Rt, got shape {mat.shape}")
    r = mat[:3, :3]
    t = mat[:3, 3]
    return -r.T @ t


def parse_ts_and_view(info: dict[str, Any]) -> tuple[int, str]:
    if "rig_ts" in info:
        return int(info["rig_ts"]), str(info.get("rig_view", ""))
    name = str(info.get("name", ""))
    match = _TS_RE.match(name)
    if not match:
        raise ValueError(
            "Cannot recover rig timestep from keyframe info. Expected saved "
            "info['rig_ts'] or a name like '<view>__ts00037'. Got: "
            f"{info!r}"
        )
    return int(match.group("ts")), match.group("view")


def load_estimated_centers(metadata_path: Path) -> tuple[np.ndarray, list[int], dict[str, float]]:
    metadata = json.loads(metadata_path.read_text())
    grouped: dict[int, list[np.ndarray]] = {}
    views_by_ts: dict[int, set[str]] = {}
    for keyframe in metadata.get("keyframes", []):
        info = keyframe.get("info", {})
        ts, view = parse_ts_and_view(info)
        grouped.setdefault(ts, []).append(center_from_w2c(keyframe["Rt"]))
        views_by_ts.setdefault(ts, set()).add(view)

    if not grouped:
        raise ValueError(f"No rig keyframes found in {metadata_path}")

    ts_list = sorted(grouped)
    centers = []
    spreads = []
    view_counts = []
    for ts in ts_list:
        per_view = np.stack(grouped[ts], axis=0)
        center = per_view.mean(axis=0)
        centers.append(center)
        spreads.append(float(np.linalg.norm(per_view - center[None], axis=1).max()))
        view_counts.append(len(views_by_ts.get(ts, set())) or len(per_view))

    stats = {
        "same_ts_spread_max_m": float(np.max(spreads)),
        "same_ts_spread_mean_m": float(np.mean(spreads)),
        "views_per_ts_min": int(np.min(view_counts)),
        "views_per_ts_max": int(np.max(view_counts)),
    }
    return np.stack(centers, axis=0), ts_list, stats


def load_gt_centers(gt_path: Path, ts_list: list[int]) -> np.ndarray:
    raw = json.loads(gt_path.read_text())
    centers = []
    if isinstance(raw, dict):
        for ts in ts_list:
            value = raw[str(ts)] if str(ts) in raw else raw[ts]
            if isinstance(value, dict):
                value = value["center"]
            centers.append(np.asarray(value, dtype=np.float64))
    elif isinstance(raw, list):
        for ts in ts_list:
            value = raw[ts]
            if isinstance(value, dict):
                value = value["center"]
            centers.append(np.asarray(value, dtype=np.float64))
    else:
        raise ValueError(f"Unsupported GT center JSON shape: {type(raw)!r}")
    return np.stack(centers, axis=0)


def umeyama_sim3(est: np.ndarray, gt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation such that gt ~= scale * R * est + t."""
    if est.shape != gt.shape or est.ndim != 2 or est.shape[1] != 3:
        raise ValueError(f"Expected Nx3 trajectories, got est={est.shape}, gt={gt.shape}")
    if len(est) < 3:
        raise ValueError("Need at least three timesteps for Sim3 alignment")

    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    x = est - mu_est
    y = gt - mu_gt
    cov = (y.T @ x) / len(est)
    u, singular_values, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1
    r = u @ d @ vt
    var_est = np.sum(x * x) / len(est)
    scale = float(np.trace(np.diag(singular_values) @ d) / max(var_est, 1e-12))
    t = mu_gt - scale * r @ mu_est
    return scale, r, t


def evaluate(metadata_path: Path, gt_path: Path) -> dict[str, Any]:
    est, ts_list, spread_stats = load_estimated_centers(metadata_path)
    gt = load_gt_centers(gt_path, ts_list)
    scale, rotation, translation = umeyama_sim3(est, gt)
    aligned = (scale * (rotation @ est.T)).T + translation
    errors = np.linalg.norm(gt - aligned, axis=1)
    span = float(np.linalg.norm(gt.max(axis=0) - gt.min(axis=0)))
    ate_rmse = float(np.sqrt(np.mean(errors**2)))

    return {
        "metadata_path": str(metadata_path),
        "gt_centers_path": str(gt_path),
        "num_timesteps": len(ts_list),
        "first_timestep": int(ts_list[0]),
        "last_timestep": int(ts_list[-1]),
        **spread_stats,
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
    parser.add_argument(
        "--run",
        required=True,
        help="OTF run directory containing metadata.json, or the metadata.json path itself.",
    )
    parser.add_argument(
        "--gt-centers",
        required=True,
        help="OB3D rig scene gt_centers.json.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path. Defaults to <run>/ate_ob3d_rig.json.",
    )
    args = parser.parse_args()

    run_path = _as_path(args.run)
    metadata_path = run_path if run_path.name == "metadata.json" else run_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    gt_path = _as_path(args.gt_centers)
    if not gt_path.exists():
        raise FileNotFoundError(f"gt_centers.json not found: {gt_path}")

    result = evaluate(metadata_path, gt_path)
    output_path = (
        _as_path(args.output)
        if args.output
        else metadata_path.parent / "ate_ob3d_rig.json"
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(
        "OB3D rig ATE: "
        f"n={result['num_timesteps']} "
        f"ATE={result['ATE_RMSE_m']:.6f} m "
        f"({result['ATE_RMSE_pct_span']:.4f}% span), "
        f"same-ts spread max={result['same_ts_spread_max_m']:.3e} m"
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
