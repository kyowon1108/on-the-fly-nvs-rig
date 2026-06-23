"""Minimal SE(3) log/exp and weighted Fréchet mean for pose averaging.

Only what rig_pnp needs. Shapes are 4x4 homogeneous matrices.
"""

import torch
from torch import Tensor


_EPS = 1e-7


def _so3_log(R: Tensor) -> Tensor:
    # Rodrigues inverse. Returns axis-angle vector in R^3.
    cos = torch.clamp((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5, -1.0, 1.0)
    theta = torch.acos(cos)
    small = theta.abs() < 1e-4
    # Near identity: use 1st-order expansion.
    k = torch.where(small, torch.ones_like(theta) * 0.5,
                    theta / (2.0 * torch.sin(theta).clamp(min=_EPS)))
    w = torch.stack([
        k * (R[..., 2, 1] - R[..., 1, 2]),
        k * (R[..., 0, 2] - R[..., 2, 0]),
        k * (R[..., 1, 0] - R[..., 0, 1]),
    ], dim=-1)
    # theta -> pi: the antisymmetric formula above suffers catastrophic cancellation
    # (the off-diagonal differences are 2 sin(theta)*a, a near-zero difference of two
    # O(1) entries, then amplified by k ~ theta/(2 sin theta)). Recover the axis from
    # the *symmetric* part instead, which is exact for any theta and cancellation-free:
    #   S = (R + R^T)/2 = cos(theta) I + (1-cos theta) a a^T
    #   => a a^T = (S - cos(theta) I) / (1 - cos theta)
    # take a_i = sqrt(diag), fix relative signs from column i_max, then fix the global
    # sign from the (still-informative for theta<pi) antisymmetric part. Switch over at
    # theta>2 rad where both formulas agree, so the where() blend is continuous. Rare in
    # rig PnP (candidate deltas are tiny) but a 180deg-off outlier would otherwise yield
    # a garbage rotation distance the Huber kernel can't reject.
    near_pi = theta > 2.0
    if near_pi.any():
        eye = torch.eye(3, device=R.device, dtype=R.dtype)
        S = (R + R.transpose(-1, -2)) * 0.5
        cos_e = cos.unsqueeze(-1).unsqueeze(-1)
        aaT = (S - cos_e * eye) / (1.0 - cos_e).clamp_min(_EPS)   # a a^T, exact
        diag = aaT.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)      # a_i^2
        i_max = diag.argmax(dim=-1, keepdim=True)
        col = torch.gather(aaT, -1, i_max.unsqueeze(-2).expand(*aaT.shape[:-1], 1)).squeeze(-1)
        rel = torch.sign(col)
        rel = torch.where(rel == 0, torch.ones_like(rel), rel)   # signs rel. to a_{i_max}>0
        axis = diag.sqrt() * rel
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        antisym = torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ], dim=-1)                                               # = 2 sin(theta) a
        gsign = torch.where(
            (antisym * axis).sum(-1, keepdim=True) < 0,
            -torch.ones_like(theta).unsqueeze(-1),
            torch.ones_like(theta).unsqueeze(-1),
        )
        w_pi = axis * gsign * theta.unsqueeze(-1)
        w = torch.where(near_pi.unsqueeze(-1), w_pi, w)
    return w


def _so3_exp(w: Tensor) -> Tensor:
    theta = w.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    k = w / theta
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] =  k[..., 1]
    K[..., 1, 0] =  k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] =  k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    return I + s * K + (1 - c) * (K @ K)


def se3_log(T: Tensor) -> Tensor:
    """4x4 -> 6-vector (rotation 3, translation 3).

    Uses the standard closed-form V^{-1} t so that `exp(log(T)) == T`.
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    w = _so3_log(R)
    theta = w.norm(dim=-1, keepdim=True)
    # V = I + (1-cos)/theta^2 * K + (theta-sin)/theta^3 * K^2
    K = torch.zeros_like(R)
    K[..., 0, 1] = -w[..., 2]; K[..., 0, 2] =  w[..., 1]
    K[..., 1, 0] =  w[..., 2]; K[..., 1, 2] = -w[..., 0]
    K[..., 2, 0] = -w[..., 1]; K[..., 2, 1] =  w[..., 0]
    th2 = (theta * theta).clamp(min=_EPS)
    th3 = (theta * th2).clamp(min=_EPS)
    A = (1 - torch.cos(theta)) / th2
    B = (theta - torch.sin(theta)) / th3
    # Handle tiny theta with Taylor expansion.
    small = theta.squeeze(-1) < 1e-4
    A = torch.where(small.unsqueeze(-1), torch.full_like(A, 0.5), A)
    B = torch.where(small.unsqueeze(-1), torch.full_like(B, 1.0 / 6.0), B)
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand_as(R)
    V = I + A.unsqueeze(-1) * K + B.unsqueeze(-1) * (K @ K)
    u = torch.linalg.solve(V, t.unsqueeze(-1)).squeeze(-1)
    return torch.cat([w, u], dim=-1)


def se3_exp(xi: Tensor) -> Tensor:
    w = xi[..., :3]
    u = xi[..., 3:]
    R = _so3_exp(w)
    theta = w.norm(dim=-1, keepdim=True)
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    K[..., 0, 1] = -w[..., 2]; K[..., 0, 2] =  w[..., 1]
    K[..., 1, 0] =  w[..., 2]; K[..., 1, 2] = -w[..., 0]
    K[..., 2, 0] = -w[..., 1]; K[..., 2, 1] =  w[..., 0]
    th2 = (theta * theta).clamp(min=_EPS)
    th3 = (theta * th2).clamp(min=_EPS)
    A = (1 - torch.cos(theta)) / th2
    B = (theta - torch.sin(theta)) / th3
    small = theta.squeeze(-1) < 1e-4
    A = torch.where(small.unsqueeze(-1), torch.full_like(A, 0.5), A)
    B = torch.where(small.unsqueeze(-1), torch.full_like(B, 1.0 / 6.0), B)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(R)
    V = I + A.unsqueeze(-1) * K + B.unsqueeze(-1) * (K @ K)
    t = (V @ u.unsqueeze(-1)).squeeze(-1)
    T = torch.eye(4, device=w.device, dtype=w.dtype).repeat(*w.shape[:-1], 1, 1)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


def se3_weighted_mean(Ts: Tensor, weights: Tensor, iters: int = 3) -> Tensor:
    """Weighted Fréchet mean on SE(3) via iterated tangent-space averaging.

    Ts: (N, 4, 4), weights: (N,) non-negative, not all zero.
    Returns: (4, 4).
    """
    assert Ts.dim() == 3 and Ts.shape[-2:] == (4, 4)
    w = weights.to(Ts) / weights.sum().clamp(min=_EPS)
    # Initialize at the max-weight pose.
    ref_idx = int(torch.argmax(weights).item())
    T_mean = Ts[ref_idx].clone()
    for _ in range(iters):
        T_mean_inv = torch.linalg.inv(T_mean)
        delta = T_mean_inv.unsqueeze(0) @ Ts        # (N, 4, 4), in mean's frame
        xi = se3_log(delta)                          # (N, 6)
        xi_mean = (w.unsqueeze(-1) * xi).sum(dim=0)  # (6,)
        if xi_mean.abs().max() < 1e-8:
            break
        T_mean = T_mean @ se3_exp(xi_mean)
    return T_mean


def se3_robust_mean(
    Ts: Tensor,
    weights: Tensor,
    iters: int = 8,
    huber_rot_rad: float = 0.2,
    huber_trans: float = 0.05,
    scene_scale: float = 1.0,
    # Backwards-compat shim: old callers passed `huber_c=0.3`. If provided it
    # becomes the rotation threshold and translation is scaled to match.
    huber_c: float = None,
) -> Tensor:
    """IRLS Fréchet mean on SE(3) with a *scale-independent* Huber kernel.

    Rotation and translation use separate thresholds (a single ||xi|| over
    [ω(rad), u(units)] would mix radians with scene units and be meaningless):
      - `rot_dist    = ||ω||`                 (radians)
      - `trans_dist  = ||u|| / scene_scale`   (unit-less)
    and combines them via the tighter Huber weight
      `w = min(huber(rot_dist, huber_rot_rad),
               huber(trans_dist, huber_trans))`
    so either component going rogue triggers down-weighting.

    Args:
        Ts: (N, 4, 4) candidate transforms.
        weights: (N,) non-negative prior weights (e.g. PnP inlier counts).
        iters: IRLS outer iterations.
        huber_rot_rad: rotation threshold in radians (default ≈11.5°).
        huber_trans: translation threshold in `scene_scale`-normalized units.
        scene_scale: expected translation magnitude between neighboring
            frames (for this repo: bootstrap normalizes to 0.1).
        huber_c: deprecated single-scalar threshold; if set, used as
            `huber_rot_rad` and `huber_trans = huber_c * scene_scale`.

    Returns: (T_mean, effective_weights)
    """
    assert Ts.dim() == 3 and Ts.shape[-2:] == (4, 4)
    if huber_c is not None:
        huber_rot_rad = float(huber_c)
        huber_trans = float(huber_c) * float(scene_scale)
    base_w = weights.to(Ts)
    base_w = base_w / base_w.sum().clamp(min=_EPS)
    ref_idx = int(torch.argmax(weights).item())
    T_mean = Ts[ref_idx].clone()
    eff_w = base_w.clone()
    for _ in range(iters):
        T_mean_inv = torch.linalg.inv(T_mean)
        delta = T_mean_inv.unsqueeze(0) @ Ts
        xi = se3_log(delta)                      # (N, 6); [rot(3), trans(3)]
        rot_dist = xi[..., :3].norm(dim=-1)      # radians
        trans_dist = xi[..., 3:].norm(dim=-1) / max(float(scene_scale), _EPS)
        rot_huber = torch.where(
            rot_dist <= huber_rot_rad,
            torch.ones_like(rot_dist),
            huber_rot_rad / rot_dist.clamp(min=_EPS),
        )
        trans_huber = torch.where(
            trans_dist <= huber_trans,
            torch.ones_like(trans_dist),
            huber_trans / trans_dist.clamp(min=_EPS),
        )
        huber = torch.minimum(rot_huber, trans_huber)
        eff_w = base_w * huber
        eff_w = eff_w / eff_w.sum().clamp(min=_EPS)
        xi_mean = (eff_w.unsqueeze(-1) * xi).sum(dim=0)
        if xi_mean.abs().max() < 1e-8:
            break
        T_mean = T_mean @ se3_exp(xi_mean)
    return T_mean, eff_w
