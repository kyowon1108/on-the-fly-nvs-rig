#!/usr/bin/env python3
"""Summarize rig post-hoc render metrics by train/holdout split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_frames": 0,
            "psnr_mean": None,
            "ssim_mean": None,
            "lpips_mean": None,
            "psnr_min": None,
            "psnr_max": None,
        }
    psnr = [float(r["psnr"]) for r in rows]
    ssim = [float(r["ssim"]) for r in rows]
    lpips = _finite([float(r["lpips"]) for r in rows])
    return {
        "num_frames": len(rows),
        "psnr_mean": mean(psnr),
        "ssim_mean": mean(ssim),
        "lpips_mean": mean(lpips) if lpips else None,
        "psnr_min": min(psnr),
        "psnr_max": max(psnr),
    }


def load_metrics(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    rows = data.get("per_frame", [])
    if not rows:
        raise ValueError(f"No per_frame rows found in {path}")
    train_rows = [r for r in rows if not bool(r.get("is_test", False))]
    holdout_rows = [r for r in rows if bool(r.get("is_test", False))]
    views = sorted({str(r.get("rig_view", "")) for r in rows})
    holdout_views = sorted({str(r.get("rig_view", "")) for r in holdout_rows})
    return {
        "metrics_path": str(path),
        "views": views,
        "holdout_views": holdout_views,
        "all_views": summarize_rows(rows),
        "train_views": summarize_rows(train_rows),
        "holdout_views_summary": summarize_rows(holdout_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        help="OTF run directory, render_eval directory, or metrics.json path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path. Defaults to <render_eval>/split_metrics.json.",
    )
    args = parser.parse_args()

    input_path = _as_path(args.run)
    if input_path.name == "metrics.json":
        metrics_path = input_path
    elif input_path.name == "render_eval":
        metrics_path = input_path / "metrics.json"
    else:
        metrics_path = input_path / "render_eval" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")

    summary = load_metrics(metrics_path)
    output_path = (
        _as_path(args.output)
        if args.output
        else metrics_path.parent / "split_metrics.json"
    )
    output_path.write_text(json.dumps(summary, indent=2) + "\n")

    holdout = summary["holdout_views_summary"]
    train = summary["train_views"]
    print(
        "Rig render split: "
        f"train n={train['num_frames']} PSNR={train['psnr_mean']}, "
        f"holdout n={holdout['num_frames']} PSNR={holdout['psnr_mean']}"
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
