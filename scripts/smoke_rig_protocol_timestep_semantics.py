#!/usr/bin/env python3
"""Smoke-test rig source timestep and stream-order semantics without CUDA images."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloaders.rig_image_dataset import (
    _build_timestep_records,
    _load_timestep_split,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def main() -> None:
    names = {"frame_000010.png", "frame_000002.png"}
    records = _build_timestep_records(names)

    assert [record["source_ts"] for record in records] == [2, 10]
    assert [record["stream_idx"] for record in records] == [0, 1]
    assert records[1]["source_ts"] == 10
    assert records[1]["stream_idx"] == 1

    records_after_start = _build_timestep_records(names, start_at=1)
    assert [record["source_ts"] for record in records_after_start] == [10]
    assert [record["stream_idx"] for record in records_after_start] == [0]

    with tempfile.TemporaryDirectory() as tmp:
        split_path = _write(Path(tmp) / "test.txt", "10\n")
        test_timesteps = _load_timestep_split(str(split_path), {2, 10})
    assert test_timesteps == {10}
    assert 1 not in test_timesteps

    try:
        _build_timestep_records({"frame_000002.png", "rgb_000002.png"})
    except ValueError as exc:
        assert "duplicate source timestep 2" in str(exc)
    else:
        raise AssertionError("duplicate source timesteps must be rejected")

    print("rig timestep-semantics protocol smoke passed")


if __name__ == "__main__":
    main()
