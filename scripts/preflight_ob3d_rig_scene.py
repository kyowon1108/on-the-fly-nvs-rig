#!/usr/bin/env python3
"""Preflight checks for an OB3D virtual-rig scene.

This is intentionally CPU/file-system only. Run it before launching an OTF rig
experiment to catch invalid ref/holdout view names, missing view folders,
missing shared frames, missing GT center files, and invalid OB3D train/test
timestep split files.
"""

from __future__ import annotations

import argparse
import json
import re
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


def parse_timestep_token(token: str) -> int:
    token = token.strip()
    if token.isdigit():
        return int(token)
    match = re.search(r"(\d+)", Path(token).name)
    if match is None:
        raise ValueError(f"Cannot parse timestep token: {token!r}")
    return int(match.group(1))


def load_timesteps(path: Path) -> set[int]:
    timesteps: set[int] = set()
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                timesteps.add(parse_timestep_token(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not timesteps:
        raise ValueError(f"No test timesteps found in {path}")
    return timesteps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="Prepared rig scene directory.")
    parser.add_argument("--rig-config", required=True, help="Rig JSON used by train.py.")
    parser.add_argument("--ref-view", required=True, help="Reference view name.")
    parser.add_argument("--holdout-view", default="", help="Optional holdout view name.")
    parser.add_argument(
        "--test-timesteps-file",
        default="",
        help="Optional rig timestep/EQR holdout file, e.g. OB3D test.txt.",
    )
    parser.add_argument(
        "--train-timesteps-file",
        default="",
        help="Optional rig timestep/EQR train file, e.g. OB3D train.txt.",
    )
    parser.add_argument("--images-dir", default="images", help="Image root under scene.")
    args = parser.parse_args()

    scene = Path(args.scene).expanduser().resolve()
    rig_config = Path(args.rig_config).expanduser().resolve()
    views = load_rig_views(rig_config)
    if args.ref_view not in views:
        raise ValueError(f"ref view {args.ref_view!r} not in rig views: {views}")
    if args.holdout_view:
        if args.test_timesteps_file or args.train_timesteps_file:
            raise ValueError("--holdout-view and timestep split files are mutually exclusive")
        if args.holdout_view not in views:
            raise ValueError(f"holdout view {args.holdout_view!r} not in rig views: {views}")
        if args.holdout_view == args.ref_view:
            raise ValueError("holdout view cannot equal ref view")
    if args.train_timesteps_file and not args.test_timesteps_file:
        raise ValueError("--train-timesteps-file requires --test-timesteps-file")

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
    train_timesteps: set[int] = set()
    test_timesteps: set[int] = set()
    common_ts = {parse_timestep_token(name) for name in common}
    if args.train_timesteps_file:
        train_path = Path(args.train_timesteps_file).expanduser().resolve()
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train timestep file: {train_path}")
        train_timesteps = load_timesteps(train_path)
        missing_ts = sorted(train_timesteps - common_ts)
        if missing_ts:
            raise RuntimeError(
                f"{train_path} contains timesteps missing from prepared rig scene: "
                f"{missing_ts[:10]}"
            )
    if args.test_timesteps_file:
        split_path = Path(args.test_timesteps_file).expanduser().resolve()
        if not split_path.exists():
            raise FileNotFoundError(f"Missing test timestep file: {split_path}")
        test_timesteps = load_timesteps(split_path)
        missing_ts = sorted(test_timesteps - common_ts)
        if missing_ts:
            raise RuntimeError(
                f"{split_path} contains timesteps missing from prepared rig scene: "
                f"{missing_ts[:10]}"
            )
    overlap = sorted(train_timesteps & test_timesteps)
    if overlap:
        raise RuntimeError(f"train/test timesteps overlap: {overlap[:10]}")
    tracking_timesteps = common_ts - train_timesteps - test_timesteps if train_timesteps else set()

    print(f"scene: {scene}")
    print(f"views: {len(views)}")
    print(f"ref_view: {args.ref_view}")
    print(f"holdout_view: {args.holdout_view or '<none>'}")
    print(f"train_timesteps_file: {args.train_timesteps_file or '<none>'}")
    print(f"test_timesteps_file: {args.test_timesteps_file or '<none>'}")
    print(f"train_timesteps: {len(train_timesteps)}")
    print(f"test_timesteps: {len(test_timesteps)}")
    print(f"tracking_only_timesteps: {len(tracking_timesteps)}")
    if train_timesteps:
        print(f"train_frames_expanded_to_views: {len(train_timesteps) * len(views)}")
    if test_timesteps:
        print(f"test_frames_expanded_to_views: {len(test_timesteps) * len(views)}")
    if tracking_timesteps:
        print(f"tracking_frames_expanded_to_views: {len(tracking_timesteps) * len(views)}")
    print(f"shared_frames: {len(common)}")
    print(f"gt_centers: {gt_count}")
    print(f"per_view_min_frames: {min(len(v) for v in per_view.values())}")
    print(f"per_view_max_frames: {max(len(v) for v in per_view.values())}")
    print(f"missing_from_common: {missing_counts}")
    print(f"extra_vs_common: {extra_counts}")
    print("status: OK")


if __name__ == "__main__":
    main()
