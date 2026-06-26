#!/usr/bin/env python3
"""CPU-only smoke checks for OTF-rig P0-1 triangulation filtering."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poses.triangulator import Triangulator
from rig.triangulation_policy import (
    collect_allowed_rig_triangulation_ids,
    make_rig_triangulation_audit,
    record_used_rig_triangulation_ids,
)


class FakeMatches:
    def __init__(self, n_matches: int):
        self.idx = torch.arange(n_matches, dtype=torch.long)
        self.kpts_other = torch.zeros(n_matches, 2)


class FakeDesc:
    def __init__(self, n_keypoints: int = 8):
        self.kpts = torch.zeros(n_keypoints, 2)
        self.matches = {}
        self.has_pt3d = torch.zeros(n_keypoints, dtype=torch.bool)
        self.pts3d = torch.zeros(n_keypoints, 3)


def _scene_keyframes():
    return [
        SimpleNamespace(info={"rig_ts": 9, "rig_eval_split": "train", "is_test": False}),
        SimpleNamespace(info={"rig_ts": 10, "rig_eval_split": "train", "is_test": False}),
        SimpleNamespace(info={"rig_ts": 8, "rig_eval_split": "test", "is_test": True}),
        SimpleNamespace(info={"rig_ts": 7, "rig_eval_split": "tracking", "is_test": True}),
    ]


def _triangulator_for_prepare(n_cams: int = 2):
    tri = Triangulator.__new__(Triangulator)
    tri.n_cams = n_cams
    return tri


def test_prepare_matches_allowed_filter() -> None:
    tri = _triangulator_for_prepare()
    desc = FakeDesc()
    desc.matches["1"] = FakeMatches(2)
    desc.matches[2] = FakeMatches(7)
    desc.matches[999] = FakeMatches(8)
    desc.matches["tmp_pnp"] = FakeMatches(8)

    _, uvs_others, chosen = tri.prepare_matches(desc, allowed_kf_ids=[1])
    assert chosen == [1], chosen
    assert uvs_others.shape == (2, 8, 2)

    _, uvs_others, chosen = tri.prepare_matches(desc, allowed_kf_ids=[])
    assert chosen == [], chosen
    assert uvs_others.device.type == "cpu"
    assert uvs_others.dtype == desc.kpts.dtype


def test_rig_partner_policy_and_actual_use_assert() -> None:
    keyframes = _scene_keyframes()
    target_info = {"rig_ts": 10, "rig_eval_split": "train", "is_test": False}
    desc = FakeDesc()
    for partner_id in (0, 1, 2, 3, "tmp_pnp"):
        desc.matches[partner_id] = FakeMatches(8 if partner_id in (3, "tmp_pnp") else 2)
    audit = make_rig_triangulation_audit()

    allowed = collect_allowed_rig_triangulation_ids(
        keyframes, target_info, desc.matches.keys(), audit
    )
    assert allowed == [0], allowed
    assert audit["triangulation_candidate_train_cross_ts"] == 1
    assert audit["triangulation_candidate_same_ts"] == 1
    assert audit["triangulation_candidate_test"] == 1
    assert audit["triangulation_candidate_tracking"] == 1
    assert audit["triangulation_candidate_invalid_id"] == 1

    tri = _triangulator_for_prepare()
    _, _, chosen = tri.prepare_matches(desc, allowed_kf_ids=allowed)
    assert chosen == [0], chosen

    record_used_rig_triangulation_ids(keyframes, target_info, chosen, audit)
    assert audit["triangulation_partner_count_train_cross_ts"] == 1
    try:
        record_used_rig_triangulation_ids(keyframes, target_info, [1], audit)
    except AssertionError:
        pass
    else:
        raise AssertionError("same-ts partner was not rejected")


def test_no_valid_train_cross_ts_partner_is_empty_and_non_mutating() -> None:
    keyframes = _scene_keyframes()
    target_info = {"rig_ts": 10, "rig_eval_split": "train", "is_test": False}
    desc = FakeDesc()
    for partner_id in (1, 2, 3, "tmp_pnp"):
        desc.matches[partner_id] = FakeMatches(8)
    has_pt3d_before = desc.has_pt3d.clone()
    pts3d_before = desc.pts3d.clone()
    audit = make_rig_triangulation_audit()

    allowed = collect_allowed_rig_triangulation_ids(
        keyframes, target_info, desc.matches.keys(), audit
    )
    assert allowed == [], allowed

    tri = _triangulator_for_prepare()
    _, _, chosen = tri.prepare_matches(desc, allowed_kf_ids=allowed)
    assert chosen == [], chosen
    assert torch.equal(desc.has_pt3d, has_pt3d_before)
    assert torch.equal(desc.pts3d, pts3d_before)


def main() -> None:
    test_prepare_matches_allowed_filter()
    test_rig_partner_policy_and_actual_use_assert()
    test_no_valid_train_cross_ts_partner_is_empty_and_non_mutating()
    print("P0-1 rig triangulation smoke passed")


if __name__ == "__main__":
    main()
