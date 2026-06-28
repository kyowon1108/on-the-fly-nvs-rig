"""blender_rig.json을 COLMAP-style relative world-to-camera transform으로 변환한다.

`eqr_to_pinhole.py`의 view 방향과 같은 convention을 써서 pinhole crop과 pose가
일치하게 만든다. Translation은 의도적으로 0으로 둔다. 즉 이 rig는 rotation-only
zero-baseline rig이며, 모든 view가 하나의 optical center를 공유한다.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import Tensor


_AXIS_FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0]))


def _quat_to_R(q: Tensor) -> Tensor:
    # Blender quaternion order: (w, x, y, z)
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = torch.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]),
    ])
    return R


def _blender_quat_to_world2cam(q: Tensor) -> Tensor:
    R_c2w = _quat_to_R(q)
    return _AXIS_FLIP.to(q) @ R_c2w.T


@dataclass
class RigConfig:
    """Reference view 기준 fixed rig geometry.

    `relative_Rt[view]`는 다음 관계를 만족하는 4x4 matrix다.
        view_world2cam = relative_Rt[view] @ rig_world2cam
    여기서 `rig_world2cam`은 reference view의 world-to-camera transform이며,
    timestep의 shared optical-center pose로 해석된다.
    """
    view_names: List[str]
    ref_view: str
    relative_Rt: Dict[str, Tensor] = field(default_factory=dict)

    def ref_index(self) -> int:
        return self.view_names.index(self.ref_view)


def load_rig_config(config_path: str, ref_view: str = "High_Cam07",
                    device: str = "cpu") -> RigConfig:
    """blender_rig.json -> RigConfig.

    ring->camera 순서를 `view_names`로 flatten하고, ref_view 기준
    `relative_Rt`를 만든다. 모든 `rel_t`는 0이며,
    `view_w2c = relative_Rt[view] @ rig_w2c`가 성립한다.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        rings = json.load(f)

    quats: Dict[str, Tensor] = {}
    order: List[str] = []
    for ring in rings:
        for cam in ring["cameras"]:
            order.append(cam["name"])
            quats[cam["name"]] = torch.tensor(cam["rotation"], dtype=torch.float32,
                                              device=device)

    if ref_view not in quats:
        raise ValueError(
            f"--ref_view {ref_view!r} not found in rig config {config_path!r}. "
            f"Available views: {order}. "
            f"(The default --ref_view is 'High_Cam07' for the Insta360 rig; for "
            f"OB3D-rig configs like rig12_panosfm.json pass e.g. "
            f"--ref_view E+0_A000.)"
        )

    ref_R_w2c = _blender_quat_to_world2cam(quats[ref_view])

    relative: Dict[str, Tensor] = {}
    for name in order:
        R_w2c = _blender_quat_to_world2cam(quats[name])
        rel_R = R_w2c @ ref_R_w2c.T
        Rt = torch.eye(4, dtype=torch.float32, device=device)
        Rt[:3, :3] = rel_R
        # Translation은 의도적으로 0: same-timestep stereo baseline을 만들지 않는다.
        relative[name] = Rt

    return RigConfig(view_names=order, ref_view=ref_view, relative_Rt=relative)
