#!/usr/bin/env python3
"""Smoke checks for non-train Gaussian spawn state isolation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene.scene_model import SceneModel, _make_rig_protocol_audit


class TrainPathReached(Exception):
    pass


class FakeDesc:
    def __init__(self):
        self.has_pt3d = torch.zeros(4, dtype=torch.bool)
        self.pts3d = torch.zeros(4, 3)


class FakeKeyframe:
    def __init__(self, split: str):
        self.info = {
            "is_test": split != "train",
            "rig_eval_split": split,
            "source_ts": 3,
            "rig_view": "E+0_A000",
        }
        self.desc_kpts = FakeDesc()
        self.depth_alignment_state = torch.tensor([7.0])
        self.update_3dpts_calls = 0
        self.align_depth_calls = 0

    def update_3dpts(self, _all_keyframes):
        self.update_3dpts_calls += 1
        if self.info["is_test"]:
            self.desc_kpts.has_pt3d[:] = True
            self.desc_kpts.pts3d += 1
            raise AssertionError("non-train keyframe reached update_3dpts")
        raise TrainPathReached

    def align_depth(self):
        self.align_depth_calls += 1
        if self.info["is_test"]:
            self.depth_alignment_state += 1
            raise AssertionError("non-train keyframe reached align_depth")


def _fake_scene(keyframe):
    scene = SceneModel.__new__(SceneModel)
    scene.use_rig = True
    scene.keyframes = [keyframe]
    scene.rig_leakage_audit = _make_rig_protocol_audit()
    scene.gaussian_params = {
        "xyz": {"val": torch.zeros(5, 3)},
    }
    return scene


def _call_spawn_planner(scene):
    if hasattr(SceneModel, "_plan_new_gaussians"):
        return SceneModel._plan_new_gaussians(scene, 0)
    return SceneModel.add_new_gaussians(scene, 0)


def _assert_non_train_unchanged(split: str, expected_counter: str):
    keyframe = FakeKeyframe(split)
    scene = _fake_scene(keyframe)
    has_pt3d_before = keyframe.desc_kpts.has_pt3d.clone()
    pts3d_before = keyframe.desc_kpts.pts3d.clone()
    depth_state_before = keyframe.depth_alignment_state.clone()
    gaussian_count_before = scene.xyz.shape[0]

    result = _call_spawn_planner(scene)

    assert result is None
    assert keyframe.update_3dpts_calls == 0
    assert keyframe.align_depth_calls == 0
    assert torch.equal(keyframe.desc_kpts.has_pt3d, has_pt3d_before)
    assert torch.equal(keyframe.desc_kpts.pts3d, pts3d_before)
    assert torch.equal(keyframe.depth_alignment_state, depth_state_before)
    assert scene.xyz.shape[0] == gaussian_count_before
    assert scene.rig_leakage_audit[expected_counter] == 1


def test_non_train_spawn_state_is_unchanged() -> None:
    _assert_non_train_unchanged("test", "spawn_skip_count_test")
    _assert_non_train_unchanged("tracking", "spawn_skip_count_tracking")


def test_train_keyframe_still_enters_existing_path() -> None:
    keyframe = FakeKeyframe("train")
    scene = _fake_scene(keyframe)
    try:
        _call_spawn_planner(scene)
    except TrainPathReached:
        pass
    else:
        raise AssertionError("train keyframe was skipped before update_3dpts")
    assert keyframe.update_3dpts_calls == 1
    assert scene.rig_leakage_audit["spawn_skip_count_test"] == 0
    assert scene.rig_leakage_audit["spawn_skip_count_tracking"] == 0


def main() -> None:
    test_non_train_spawn_state_is_unchanged()
    test_train_keyframe_still_enters_existing_path()
    print("rig spawn-state protocol smoke passed")


if __name__ == "__main__":
    main()
