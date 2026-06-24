#!/usr/bin/env python3
"""Preflight checks for an OB3D virtual-rig scene.

This is intentionally CPU/file-system only. Run it before launching an OTF rig
experiment to catch invalid ref/holdout view names, missing view folders,
missing shared frames, and missing GT center files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_rig_views(rig_config: Path) -> list[str]:
    data = json.loads(rig_config.read_text())
    views: list[str] = []
    for ring in data:
        for cam in ring.get("cameras", []):
            name = cam.get("name")
            if not name:
                raise ValueError(f"Camera without name in {rig_config}")
            views.append(str(name))
    if not views:
        raise ValueError(f"No cameras found in rig config: {rig_config}")
    return views


def image_names(view_dir: Path) -> set[str]:
    if not view_dir.is_dir():
        raise FileNotFoundError(f"Missing view directory: {view_dir}")
    return {p.name for p in view_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}


def load_gt_count(gt_path: Path) -> int:
    raw: Any = json.loads(gt_path.read_text())
    if isinstance(raw, dict):
        return len(raw)
    if isinstance(raw, list):
        return len(raw)
    raise ValueError(f"Unsupported gt_centers.json shape: {type(raw)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="Prepared rig scene directory.")
    parser.add_argument("--rig-config", required=True, help="Rig JSON used by train.py.")
    parser.add_argument("--ref-view", required=True, help="Reference view name.")
    parser.add_argument("--holdout-view", default="", help="Optional holdout view name.")
    parser.add_argument("--images-dir", default="images", help="Image root under scene.")
    args = parser.parse_args()

    scene = Path(args.scene).expanduser().resolve()
    rig_config = Path(args.rig_config).expanduser().resolve()
    views = load_rig_views(rig_config)
    if args.ref_view not in views:
        raise ValueError(f"ref view {args.ref_view!r} not in rig views: {views}")
    if args.holdout_view:
        if args.holdout_view not in views:
            raise ValueError(f"holdout view {args.holdout_view!r} not in rig views: {views}")
        if args.holdout_view == args.ref_view:
            raise ValueError("holdout view cannot equal ref view")

    images_root = scene / args.images_dir
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing images root: {images_root}")
    per_view = {view: image_names(images_root / view) for view in views}
    common = set.intersection(*per_view.values())
    gt_path = scene / "gt_centers.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT centers: {gt_path}")
    gt_count = load_gt_count(gt_path)

    missing_counts = {view: len(common - names) for view, names in per_view.items()}
    extra_counts = {view: len(names - common) for view, names in per_view.items()}
    if not common:
        raise RuntimeError(f"No shared frames across all {len(views)} views in {images_root}")
    if gt_count < len(common):
        raise RuntimeError(
            f"GT centers has {gt_count} entries but shared frames has {len(common)}"
        )

    print(f"scene: {scene}")
    print(f"views: {len(views)}")
    print(f"ref_view: {args.ref_view}")
    print(f"holdout_view: {args.holdout_view or '<none>'}")
    print(f"shared_frames: {len(common)}")
    print(f"gt_centers: {gt_count}")
    print(f"per_view_min_frames: {min(len(v) for v in per_view.values())}")
    print(f"per_view_max_frames: {max(len(v) for v in per_view.values())}")
    print(f"missing_from_common: {missing_counts}")
    print(f"extra_vs_common: {extra_counts}")
    print("status: OK")


if __name__ == "__main__":
    main()
