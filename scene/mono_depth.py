#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
import sys
import math
import urllib.request
import torch.nn.functional as F

from poses.feature_detector import DescribedKeypoints
from utils import sample

sys.path.append("submodules/Depth-Anything-V2")
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
from depth_anything_v2.dpt import DepthAnythingV2

size = 518
encoder = "vitl"


class MonoDepthInternal(torch.nn.Module):
    def __init__(self):
        super(MonoDepthInternal, self).__init__()
        model_path = f"models/depth_anything_v2_{encoder}.pth"
        if not os.path.exists(model_path):
            print(f"Downloading Depth-Anything-V2 model for {encoder}, may take a few minutes...")
            model_sizes = {
                "vits": "Small",
                "vitb": "Base",
                "vitl": "Large",
                "vitg": "Giant",
            }
            url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_sizes[encoder]}/resolve/main/depth_anything_v2_{encoder}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(url, model_path)
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model = model.to("cuda").half().eval()
        self.sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )
        self.sobel_y = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, image: torch.Tensor):
        img = torch.nn.functional.interpolate(
            image[None].half(), (size, size), mode="bilinear", align_corners=True
        )
        depth = self.model(img)[None]
        t, s = get_t_s(depth)
        depth = (depth - t) / s

        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        edges = torch.cat((grad_x, grad_y), dim=0)

        edges_sq_norm = (edges**2).sum(0, keepdim=True)
        var = 0.2
        confidence = torch.exp(-edges_sq_norm / var)
        return depth.float(), confidence.float()


def get_t_s(d):
    t = d.median()
    s = (d - t).abs().median()
    return t, s


def relative_idepth_to_depth(idepth: torch.Tensor, fallback_depth: float = 1.0) -> torch.Tensor:
    """DA-V2 returns median/MAD-normalised inverse depth (can be negative; 1/idepth blows
    up near 0). Shift so the finite min becomes +1.0, then invert -> strictly positive,
    order-preserving depth. Non-finite entries fall back to fallback_depth. Absolute
    per-view scale is re-anchored later by bootstrap BA's 0.1 normalisation."""
    finite_mask = torch.isfinite(idepth)
    if not finite_mask.any():
        return torch.full_like(idepth, fallback_depth)
    idepth_min = idepth[finite_mask].min()
    depth = 1.0 / (idepth - idepth_min + 1.0)
    return torch.where(finite_mask, depth, torch.full_like(depth, fallback_depth))


def align_rig_views(idepth_dict, rel_R_dict, view_names, ref_view,
                    f, cx, cy, height, width,
                    n_lat=64, n_lon=128, min_overlap_bins=15):
    """Reconcile per-view monocular depth SCALES using the shared-optical-centre
    overlap between rig views.

    The N rig views share one optical centre (zero baseline, rel_t=0), so the SAME
    3D point has the SAME radial depth in every view that sees it; two pixels whose
    rig-frame ray directions coincide are looking at the same point. We bin each
    view's per-pixel radial depth by ray direction on the sphere, and for every pair
    of views that share bins we read off the median log-scale offset. A weighted
    least-squares then solves one multiplicative scale per view (ref_view fixed to 1)
    so all overlapping rays agree. This removes the per-view scale incoherence of
    independently-normalised DA-V2 maps (the cause of the wide-view collapse, #5)
    WITHOUT triangulating (zero baseline gives no new depth, only scale consistency;
    the common scale stays gauge-free and is re-anchored by the bootstrap's 0.1 norm).

    Returns {view: depth_map (1,1,H,W)} (z-depth) on a common scale. Falls back to the
    independent per-view depth for any view left unconstrained by the overlap graph.

    Refs: 360MonoDepth (CVPR'22), OmniFusion (CVPR'22) — same tangent-image scale
    reconciliation; MiDaS scale-shift alignment; H&Z infinite homography K R K^-1."""
    device = idepth_dict[view_names[0]].device
    vv, uu = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ray = torch.stack([(uu - cx) / f, (vv - cy) / f, torch.ones_like(uu)], dim=-1)  # (H,W,3)
    ray_norm = ray.norm(dim=-1)                      # z-depth -> radial-depth factor
    rayn = (ray / ray_norm[..., None]).reshape(-1, 3)  # unit camera rays (HW,3)
    nb = n_lat * n_lon

    depth_z, bin_logr, bin_has = {}, {}, {}
    for v in view_names:
        dz = relative_idepth_to_depth(idepth_dict[v]).float().reshape(height, width).clamp_min(1e-6)
        depth_z[v] = dz
        r = (dz * ray_norm).reshape(-1)              # radial depth (view-invariant up to scale)
        dirs = (rel_R_dict[v].T @ rayn.T).T          # rig-frame ray dirs (HW,3)
        lon = torch.atan2(dirs[:, 1], dirs[:, 0])                 # [-pi, pi]
        lat = torch.atan2(dirs[:, 2], dirs[:, :2].norm(dim=-1))   # [-pi/2, pi/2]
        bl = ((lon + math.pi) / (2 * math.pi) * n_lon).long().clamp(0, n_lon - 1)
        bt = ((lat + math.pi / 2) / math.pi * n_lat).long().clamp(0, n_lat - 1)
        bidx = bt * n_lon + bl                       # (HW,)
        s = torch.zeros(nb, device=device).scatter_add_(0, bidx, r.log())
        c = torch.zeros(nb, device=device).scatter_add_(0, bidx, torch.ones_like(r))
        bin_logr[v] = s / c.clamp_min(1)
        bin_has[v] = c > 0

    n = len(view_names)
    rows, targets, weights = [], [], []
    for a in range(n):
        for b in range(a + 1, n):
            common = bin_has[view_names[a]] & bin_has[view_names[b]]
            k = int(common.sum())
            if k < min_overlap_bins:
                continue
            # align log r_a + x_a == log r_b + x_b  ->  x_a - x_b = median(log r_b - log r_a)
            delta = torch.median(bin_logr[view_names[b]][common] - bin_logr[view_names[a]][common])
            row = torch.zeros(n, device=device)
            row[a], row[b] = 1.0, -1.0
            rows.append(row); targets.append(delta); weights.append(float(k) ** 0.5)

    ref_i = view_names.index(ref_view) if ref_view in view_names else 0
    grow = torch.zeros(n, device=device); grow[ref_i] = 1.0           # gauge: x_ref = 0
    rows.append(grow); targets.append(torch.zeros((), device=device)); weights.append(1e3)

    w = torch.tensor(weights, device=device)
    A = torch.stack(rows) * w[:, None]
    d = torch.stack(targets) * w
    x = torch.linalg.lstsq(A, d).solution
    if not torch.isfinite(x).all():
        x = torch.zeros(n, device=device)            # degenerate graph -> no rescale
    scales = torch.exp((x - x[ref_i]).clamp(-5, 5))   # ref scale = 1; guard runaway

    if os.environ.get("OTF_DEBUG_BOOT"):
        print(f"[align_rig_views] per-view scales = "
              f"{ {view_names[i]: round(float(scales[i]), 3) for i in range(n)} }")

    return {v: (depth_z[v] * scales[i]).reshape(1, 1, height, width)
            for i, v in enumerate(view_names)}


def align_samples(tri_idepth: torch.Tensor, mono_idepth: torch.Tensor):
    t_tri, s_tri = get_t_s(tri_idepth)
    t_mono, s_mono = get_t_s(mono_idepth)
    scale = s_tri / s_mono
    offset = t_tri - t_mono * scale
    return mono_idepth * scale + offset, scale, offset


def align_depth(
    mono_depth_map: torch.Tensor, desc_kpts: DescribedKeypoints, width: int, height: int
):
    """Aligns the mono depth map with the triangulated depth from keypoints by finding the best scale and offset."""
    mono_idepth = sample(
        mono_depth_map,
        desc_kpts.kpts[desc_kpts.has_pt3d].view(1, 1, -1, 2),
        width,
        height,
    )[0, 0, 0]
    tri_idepth = 1 / desc_kpts.depth[desc_kpts.has_pt3d]

    mono_idepth_aligned, scale, offset = align_samples(tri_idepth, mono_idepth)
    err = (mono_idepth_aligned - tri_idepth).abs()
    valid = err < 5 * err.median()
    mono_idepth_aligned, scale, offset = align_samples(
        tri_idepth[valid], mono_idepth[valid]
    )
    mono_depth_map_aligned = mono_depth_map * scale + offset

    return mono_depth_map_aligned


class MonoDepthEstimator:
    @torch.no_grad()
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        model = MonoDepthInternal()

        dummy = torch.zeros(3, height, width).cuda()
        self.model = torch.cuda.make_graphed_callables(model, [dummy])

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth, conf = self.model(image)
        return depth.clone(), conf.clone()
