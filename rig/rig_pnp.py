"""Rig-aware PnP.

각 view에서 PnP를 독립적으로 풀고, 고정 relative transform을 이용해 view pose를
shared rig pose 후보로 lift한다. 여러 후보는 SE(3) 위의 robust weighted mean으로
합친다.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from rig.rig_loader import RigConfig
from rig.se3_utils import se3_robust_mean


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x)


def rig_pnp_per_view(
    correspondences: Dict[str, Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]],
    rig_config: RigConfig,
    K: Tensor,
    min_correspondences: int = 6,
    reproj_error_px: float = 4.0,
    scene_scale: float = 0.1,
    huber_trans: float = 0.05,
    min_success_views: int = 2,
) -> Tuple[Optional[Tensor], Dict[str, dict]]:
    """
    correspondences: {view_name: (pts_2d[N,2], pts_3d[N,3], optional_conf[N])}
    K: (3, 3) camera intrinsic matrix (assumed identical across views).
    scene_scale: 인접 rig pose 사이의 예상 translation 크기. Robust mean의
        translation Huber threshold를 dataset scale에 맞추는 데 쓴다.
    huber_trans: `scene_scale` 기준 normalized Huber threshold.
    Returns: (rig_world2cam 4x4 or None, per_view_stats dict).
    """
    K_np = _to_numpy(K).astype(np.float64)
    rig_candidates = []
    weights = []
    stats: Dict[str, dict] = {}

    for view_name, data in correspondences.items():
        pts2d, pts3d = data[:2]
        conf = data[2] if len(data) > 2 else None
        n_raw = pts2d.shape[0] if pts2d is not None else 0
        n_reject_nonfinite = 0
        n_reject_conf = 0
        if pts2d is not None and pts3d is not None and n_raw > 0:
            finite_mask = torch.isfinite(pts2d).all(dim=-1) & torch.isfinite(pts3d).all(dim=-1)
            if conf is not None:
                conf = conf.reshape(-1)
                conf_valid = torch.isfinite(conf) & (conf > 0)
            else:
                conf_valid = torch.ones_like(finite_mask, dtype=torch.bool)
            valid_mask = finite_mask & conf_valid
            n_reject_nonfinite = int((~finite_mask).sum().item())
            n_reject_conf = int((finite_mask & ~conf_valid).sum().item())
            pts2d = pts2d[valid_mask]
            pts3d = pts3d[valid_mask]
        n = pts2d.shape[0] if pts2d is not None else 0
        if n < min_correspondences:
            stats[view_name] = {
                "success": False,
                "n_raw": int(n_raw),
                "n_points": int(n),
                "n_reject_nonfinite": n_reject_nonfinite,
                "n_reject_conf": n_reject_conf,
                "n_inliers": 0,
            }
            continue
        pts2d_np = _to_numpy(pts2d).astype(np.float64).reshape(-1, 1, 2)
        pts3d_np = _to_numpy(pts3d).astype(np.float64).reshape(-1, 1, 3)
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d_np, pts2d_np, K_np, None,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=reproj_error_px,
                iterationsCount=2000,
                confidence=0.999,
            )
        except cv2.error:
            ok, inliers = False, None

        if not ok or inliers is None or len(inliers) < min_correspondences:
            stats[view_name] = {
                "success": False,
                "n_raw": int(n_raw),
                "n_points": int(n),
                "n_reject_nonfinite": n_reject_nonfinite,
                "n_reject_conf": n_reject_conf,
                "n_inliers": 0 if inliers is None else int(len(inliers)),
            }
            continue

        R_view, _ = cv2.Rodrigues(rvec)
        view_w2c = np.eye(4, dtype=np.float64)
        view_w2c[:3, :3] = R_view
        view_w2c[:3, 3] = tvec.reshape(3)
        view_w2c_t = torch.from_numpy(view_w2c).to(dtype=torch.float32, device=K.device)

        rel = rig_config.relative_Rt[view_name].to(view_w2c_t)
        # View PnP 결과를 shared rig pose 후보로 lift한다.
        #   view_w2c = rel @ rig_w2c  =>  rig_w2c = rel^{-1} @ view_w2c
        rig_w2c = torch.linalg.inv(rel) @ view_w2c_t

        rig_candidates.append(rig_w2c)
        weights.append(float(len(inliers)))
        stats[view_name] = {
            "success": True,
            "n_raw": int(n_raw),
            "n_points": int(n),
            "n_reject_nonfinite": n_reject_nonfinite,
            "n_reject_conf": n_reject_conf,
            "n_inliers": int(len(inliers)),
        }

    successful_views = [v for v, s in stats.items() if s.get("success")]
    stats["_summary"] = {
        "successful_views": successful_views,
        "num_successful_views": len(successful_views),
        "min_success_views": int(min_success_views),
        "single_view_fallback": bool(len(successful_views) == 1 and min_success_views <= 1),
        "accepted": False,
    }
    if len(rig_candidates) < min_success_views:
        return None, stats

    Ts = torch.stack(rig_candidates, dim=0)
    w = torch.tensor(weights, dtype=torch.float32, device=Ts.device)
    rig_mean, eff_w = se3_robust_mean(Ts, w, scene_scale=scene_scale, huber_trans=huber_trans)
    # Post-Huber effective weight를 기록해 어떤 view 후보가 실제로 신뢰됐는지 남긴다.
    for v, ew in zip(successful_views, eff_w.tolist()):
        stats[v]["effective_weight"] = float(ew)
    stats["_summary"]["accepted"] = True
    return rig_mean, stats
