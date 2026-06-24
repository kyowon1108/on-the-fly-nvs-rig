#!/usr/bin/env python3
"""Summarize rig post-hoc render metrics by train/test split."""

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
    def split_of(row: dict[str, Any]) -> str:
        return str(
            row.get(
                "rig_eval_split",
                "test" if bool(row.get("is_test", False)) else "train",
            )
        )

    train_rows = [r for r in rows if split_of(r) == "train"]
    test_rows = [r for r in rows if split_of(r) == "test"]
    tracking_rows = [r for r in rows if split_of(r) == "tracking"]
    views = sorted({str(r.get("rig_view", "")) for r in rows})
    test_views = sorted({str(r.get("rig_view", "")) for r in test_rows})
    test_timesteps = sorted({
        int(r["rig_ts"]) for r in test_rows if r.get("rig_ts") is not None
    })
    split_meta = data.get("split", {})
    split_mode = split_meta.get(
        "mode",
        "timestep" if len(test_views) == len(views) and test_timesteps else (
            "view" if test_rows else "none"
        ),
    )
    test_summary = summarize_rows(test_rows)
    return {
        "metrics_path": str(path),
        "split_mode": split_mode,
        "views": views,
        "test_views": test_views,
        "test_timesteps": test_timesteps,
        "all_views": summarize_rows(rows),
        "train_views": summarize_rows(train_rows),
        "test_frames": test_summary,
        "tracking_frames": summarize_rows(tracking_rows),
        # Backward-compatible key used by older collectors.
        "holdout_views": test_views,
        "holdout_views_summary": test_summary,
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

    test = summary["test_frames"]
    train = summary["train_views"]
    tracking = summary["tracking_frames"]
    print(
        "Rig render split: "
        f"mode={summary['split_mode']} "
        f"train n={train['num_frames']} PSNR={train['psnr_mean']}, "
        f"test n={test['num_frames']} PSNR={test['psnr_mean']}, "
        f"tracking n={tracking['num_frames']} excluded"
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
