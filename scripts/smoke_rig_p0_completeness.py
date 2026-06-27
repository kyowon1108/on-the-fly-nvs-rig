#!/usr/bin/env python3
"""Smoke-test P0-3 rig completeness accounting."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import _build_rig_completeness


def main() -> None:
    expected = {
        "all": [2, 10, 18],
        "train": [2],
        "test": [10],
        "tracking": [18],
    }
    per_frame = [
        {"rig_ts": 2, "rig_view": "A", "rig_eval_split": "train"},
        {"rig_ts": 2, "rig_view": "B", "rig_eval_split": "train"},
        {"rig_ts": 18, "rig_view": "A", "rig_eval_split": "tracking"},
        {"rig_ts": 18, "rig_view": "B", "rig_eval_split": "tracking"},
    ]
    failures = [{
        "source_ts": 10,
        "stream_idx": 1,
        "split": "test",
        "reason": "forced smoke failure",
    }]

    completeness = _build_rig_completeness(expected, per_frame, failures, n_views=2)

    assert completeness["views_per_timestep_expected"] == 2
    assert completeness["registered_timesteps_all"] == [2, 18]
    assert completeness["failed_timesteps_all"] == [10]
    assert completeness["missing_timesteps_all"] == [10]
    assert completeness["registered_timesteps_test"] == []
    assert completeness["failed_timesteps_test"] == [10]
    assert completeness["missing_timesteps_test"] == [10]
    assert completeness["registration_recall_test"] == 0.0
    assert completeness["views_per_timestep_min"] == 2
    assert completeness["views_per_timestep_max"] == 2

    print("P0-3 rig completeness smoke passed")


if __name__ == "__main__":
    main()
