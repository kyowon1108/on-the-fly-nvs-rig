#!/usr/bin/env python3
"""Fail-fast checks for protocol-safe OTF-Rig run artifacts.

This script does not evaluate whether a scene looks good. It checks whether a
rig run satisfies the protocol invariants needed for reporting: no geometry
leakage, no hidden missing timesteps, one complete view packet per timestep,
and separate test-split and diagnostic metric artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ZERO_AUDIT_KEYS = (
    "triangulation_partner_count_same_ts",
    "triangulation_partner_count_test",
    "triangulation_partner_count_tracking",
    "triangulation_partner_count_invalid_id",
    "mvs_partner_count_same_ts",
    "spawn_count_test",
    "spawn_count_tracking",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _center_from_w2c(rt: Any) -> tuple[float, float, float]:
    if len(rt) != 4 or any(len(row) != 4 for row in rt):
        raise ValueError("Expected 4x4 Rt")
    r = [[float(rt[i][j]) for j in range(3)] for i in range(3)]
    t = [float(rt[i][3]) for i in range(3)]
    return tuple(-sum(r[row][col] * t[row] for row in range(3)) for col in range(3))


def _keyframe_source_ts(keyframe: dict[str, Any]) -> int:
    info = keyframe.get("info", {})
    if "source_ts" not in info:
        raise ValueError(f"Rig keyframe is missing source_ts: {info!r}")
    return int(info["source_ts"])


def _center_spread_stats(metadata: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[int, list[tuple[float, float, float]]] = {}
    views: dict[int, set[str]] = {}
    for keyframe in metadata.get("keyframes", []):
        info = keyframe.get("info", {})
        if "rig_view" not in info:
            continue
        if "Rt" not in keyframe:
            raise ValueError(f"Rig keyframe is missing Rt: {info!r}")
        ts = _keyframe_source_ts(keyframe)
        grouped.setdefault(ts, []).append(_center_from_w2c(keyframe["Rt"]))
        views.setdefault(ts, set()).add(str(info.get("rig_view", "")))

    if not grouped:
        raise ValueError("No rig keyframes with Rt/source_ts found in metadata.json")

    spreads = []
    view_counts = []
    for ts, centers in grouped.items():
        mean = tuple(sum(center[i] for center in centers) / len(centers) for i in range(3))
        spreads.append(max(
            math.sqrt(sum((center[i] - mean[i]) ** 2 for i in range(3)))
            for center in centers
        ))
        view_counts.append(len(views.get(ts, set())) or len(centers))
    return {
        "same_ts_spread_max_m": float(max(spreads)),
        "views_per_ts_min": int(min(view_counts)),
        "views_per_ts_max": int(max(view_counts)),
    }


def _extract_completeness(metadata: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    _ = metadata
    return _load_json(run_dir / "render_eval" / "rig_completeness.json")


def _extract_leakage_audit(metadata: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    _ = metadata
    return _load_json(run_dir / "render_eval" / "rig_leakage_audit.json")


def _extract_incremental_refinement(metadata: dict[str, Any]) -> dict[str, Any]:
    return metadata.get("extra", {}).get("rig_incremental_refinement", {})


def _protocol_failures(
    run_dir: Path,
    expected_num_views: int,
    max_center_spread: float,
    fail_on_missing: bool,
) -> tuple[list[str], dict[str, Any]]:
    metadata_path = run_dir / "metadata.json"
    metadata = _load_json(metadata_path)
    eval_dir = run_dir / "render_eval"
    failures: list[str] = []

    leakage = _extract_leakage_audit(metadata, run_dir)
    completeness = _extract_completeness(metadata, run_dir)
    refinement = _extract_incremental_refinement(metadata)
    if not leakage:
        failures.append("missing rig leakage audit")
    if not completeness:
        failures.append("missing rig completeness metadata")
    if not refinement:
        failures.append("missing rig incremental refinement metadata")
    elif int(refinement.get("fallbacks", 0) or 0) != 0:
        failures.append(
            "rig incremental MiniBA fallback count must be 0, "
            f"got {refinement.get('fallbacks')}"
        )

    for key in ZERO_AUDIT_KEYS:
        if key not in leakage:
            failures.append(f"missing required leakage audit key: {key}")
            continue
        value = int(leakage.get(key, 0) or 0)
        if value != 0:
            failures.append(f"{key} must be 0, got {value}")

    missing_test = [int(ts) for ts in completeness.get("missing_timesteps_test", [])]
    missing_all = [int(ts) for ts in completeness.get("missing_timesteps_all", [])]
    if fail_on_missing:
        if missing_all:
            failures.append(f"missing_timesteps_all must be empty, got {missing_all}")
        if missing_test:
            failures.append(f"missing_test_timesteps must be empty, got {missing_test}")

    for key in ("views_per_timestep_min", "views_per_timestep_max"):
        if key in completeness and int(completeness[key]) != expected_num_views:
            failures.append(f"{key} must be {expected_num_views}, got {completeness[key]}")

    spread_stats = _center_spread_stats(metadata)
    if spread_stats["views_per_ts_min"] != expected_num_views:
        failures.append(
            f"metadata views_per_ts_min must be {expected_num_views}, "
            f"got {spread_stats['views_per_ts_min']}"
        )
    if spread_stats["views_per_ts_max"] != expected_num_views:
        failures.append(
            f"metadata views_per_ts_max must be {expected_num_views}, "
            f"got {spread_stats['views_per_ts_max']}"
        )
    if spread_stats["same_ts_spread_max_m"] >= max_center_spread:
        failures.append(
            f"same_ts_spread_max_m must be < {max_center_spread:g}, "
            f"got {spread_stats['same_ts_spread_max_m']:.6g}"
        )

    if not (eval_dir / "metrics_claim_test.json").exists():
        failures.append(f"missing test metric artifact: {eval_dir / 'metrics_claim_test.json'}")
    if not (eval_dir / "metrics_diagnostic_all.json").exists():
        failures.append(f"missing diagnostic metric: {eval_dir / 'metrics_diagnostic_all.json'}")

    summary = {
        "run": str(run_dir),
        "expected_num_views": expected_num_views,
        "leakage_audit": leakage,
        "completeness": completeness,
        "rig_incremental_refinement": refinement,
        **spread_stats,
        "metrics_claim_test_exists": (eval_dir / "metrics_claim_test.json").exists(),
        "metrics_diagnostic_all_exists": (eval_dir / "metrics_diagnostic_all.json").exists(),
    }
    return failures, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="OTF-Rig run directory.")
    parser.add_argument("--expected-num-views", type=int, required=True)
    parser.add_argument("--max-center-spread", type=float, default=1e-6)
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument("--output", default="", help="Optional JSON summary path.")
    args = parser.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    failures, summary = _protocol_failures(
        run_dir=run_dir,
        expected_num_views=args.expected_num_views,
        max_center_spread=args.max_center_spread,
        fail_on_missing=args.fail_on_missing,
    )
    if args.output:
        Path(args.output).expanduser().resolve().write_text(
            json.dumps({"ok": not failures, "failures": failures, **summary}, indent=2)
        )

    if failures:
        print("Rig protocol artifact check FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("Rig protocol artifact check OK")
    print(
        "  "
        f"views/timestep={summary['views_per_ts_min']}..{summary['views_per_ts_max']}, "
        f"same_ts_spread_max_m={summary['same_ts_spread_max_m']:.3e}"
    )


if __name__ == "__main__":
    main()
