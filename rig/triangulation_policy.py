"""Lightweight rig triangulation partner policy.

This module deliberately avoids importing scene/keyframe/feature code so the
rig protocol checks can run on CPU without loading CUDA/CuPy-backed matchers.
"""

RIG_TRIANGULATION_AUDIT_KEYS = (
    "triangulation_candidate_train_cross_ts",
    "triangulation_candidate_same_ts",
    "triangulation_candidate_test",
    "triangulation_candidate_tracking",
    "triangulation_candidate_invalid_id",
    "triangulation_partner_count_train_cross_ts",
    "triangulation_partner_count_same_ts",
    "triangulation_partner_count_test",
    "triangulation_partner_count_tracking",
    "triangulation_partner_count_invalid_id",
)


def make_rig_triangulation_audit() -> dict[str, int]:
    return {key: 0 for key in RIG_TRIANGULATION_AUDIT_KEYS}


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def classify_rig_triangulation_partner(keyframes, target_info: dict, partner_id) -> str:
    partner_id = _as_int(partner_id)
    if partner_id is None or partner_id < 0 or partner_id >= len(keyframes):
        return "invalid_id"
    partner = keyframes[partner_id]
    partner_info = getattr(partner, "info", {})
    target_ts = target_info.get("source_ts")
    partner_ts = partner_info.get("source_ts")
    if target_ts is None or partner_ts is None:
        return "invalid_id"
    if int(partner_ts) == int(target_ts):
        return "same_ts"
    split = partner_info.get("rig_eval_split")
    if partner_info.get("is_test", False) or split != "train":
        if split == "tracking":
            return "tracking"
        return "test"
    return "train_cross_ts"


def collect_allowed_rig_triangulation_ids(
    keyframes,
    target_info: dict,
    match_keys,
    audit: dict[str, int] | None = None,
) -> list[int]:
    allowed = []
    for raw_id in list(match_keys):
        partner_id = _as_int(raw_id)
        cls = classify_rig_triangulation_partner(keyframes, target_info, partner_id)
        if audit is not None:
            audit[f"triangulation_candidate_{cls}"] += 1
        if cls == "train_cross_ts":
            allowed.append(partner_id)
    return allowed


def record_used_rig_triangulation_ids(
    keyframes,
    target_info: dict,
    partner_ids,
    audit: dict[str, int] | None = None,
) -> None:
    bad = []
    for raw_id in partner_ids:
        partner_id = _as_int(raw_id)
        cls = classify_rig_triangulation_partner(keyframes, target_info, partner_id)
        if audit is not None:
            audit[f"triangulation_partner_count_{cls}"] += 1
        if cls != "train_cross_ts":
            bad.append((raw_id, cls))
    if bad:
        raise AssertionError(
            "rig triangulation used non-train/cross-ts partners: "
            f"{bad[:10]}"
        )
