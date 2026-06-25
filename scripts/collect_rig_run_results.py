#!/usr/bin/env python3
"""Collect OTF rig run outputs into a compact CSV table.

Inputs are run directories that may contain:
- metadata.json
- ate_ob3d_rig.json
- render_eval/split_metrics.json
- train.log

The script is CPU-only and does not render or touch CUDA.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


RUN_RE = re.compile(
    r"(?P<prefix>.*?)(?P<scene>[A-Za-z0-9_-]+)?_(?P<variant>default|freeze|depth0|proba0|holdout)?"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def metric(summary: dict[str, Any], key: str) -> Any:
    value = summary.get(key)
    return "" if value is None else value


def infer_ablation(run: Path) -> str:
    name = run.name.lower()
    if "freeze" in name:
        return "freeze_rig_poses"
    if "depth0" in name or "depth_loss_weight_init_0" in name:
        return "depth_loss_weight_init_0"
    if "proba0" in name or "init_proba_scaler_0" in name:
        return "init_proba_scaler_0"
    return "default"


def infer_scene(run: Path, metadata: dict[str, Any]) -> str:
    source = metadata.get("config", {}).get("source_path")
    if source:
        return Path(str(source)).name
    return run.name


def collect_one(run: Path) -> dict[str, Any]:
    metadata = read_json(run / "metadata.json")
    ate = read_json(run / "ate_ob3d_rig.json")
    split = read_json(run / "render_eval" / "split_metrics.json")
    train = split.get("train_views", {})
    test = split.get("test_frames", split.get("holdout_views_summary", {}))
    tracking = split.get("tracking_frames", {})
    all_views = split.get("all_views", {})
    return {
        "run": str(run),
        "scene": infer_scene(run, metadata),
        "ablation": infer_ablation(run),
        "render_split_mode": split.get("split_mode", ""),
        "num_keyframes": metadata.get("num keyframes", ""),
        "runtime_sec": metadata.get("time", ""),
        "fps": metadata.get("FPS", ""),
        "registered_timesteps": ate.get("num_timesteps", ""),
        "views_per_ts_min": ate.get("views_per_ts_min", ""),
        "views_per_ts_max": ate.get("views_per_ts_max", ""),
        "same_ts_spread_max_m": ate.get("same_ts_spread_max_m", ""),
        "ATE_RMSE_m": ate.get("ATE_RMSE_m", ""),
        "ATE_RMSE_pct_span": ate.get("ATE_RMSE_pct_span", ""),
        "train_num_frames": metric(train, "num_frames"),
        "train_psnr": metric(train, "psnr_mean"),
        "train_ssim": metric(train, "ssim_mean"),
        "train_lpips": metric(train, "lpips_mean"),
        "test_num_frames": metric(test, "num_frames"),
        "test_psnr": metric(test, "psnr_mean"),
        "test_ssim": metric(test, "ssim_mean"),
        "test_lpips": metric(test, "lpips_mean"),
        "tracking_num_frames": metric(tracking, "num_frames"),
        "holdout_num_frames": metric(test, "num_frames"),
        "holdout_psnr": metric(test, "psnr_mean"),
        "holdout_ssim": metric(test, "ssim_mean"),
        "holdout_lpips": metric(test, "lpips_mean"),
        "all_psnr": metric(all_views, "psnr_mean"),
        "all_ssim": metric(all_views, "ssim_mean"),
        "all_lpips": metric(all_views, "lpips_mean"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run directories to collect.")
    parser.add_argument("--output", default="", help="CSV path. Defaults to stdout.")
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        path = Path(run).expanduser().resolve()
        if path.is_dir():
            rows.append(collect_one(path))
    fieldnames = list(rows[0].keys()) if rows else []

    if args.output:
        out = Path(args.output).expanduser().resolve()
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out}")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
