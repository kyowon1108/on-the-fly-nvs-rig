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

from argparse import Namespace
from functools import wraps
import gc
import os
import json
import math
import threading
import time
import warnings
import cv2
import torch
import torch.nn.functional as F
import numpy as np

import lpips
from fused_ssim import fused_ssim
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distIndex2
from poses.feature_detector import DescribedKeypoints
from poses.matcher import Matcher
from poses.guided_mvs import GuidedMVS
from rig.triangulation_policy import (
    classify_rig_triangulation_partner,
    collect_allowed_rig_triangulation_ids,
    make_rig_triangulation_audit,
    record_used_rig_triangulation_ids,
)
from scene.optimizers import SparseGaussianAdam
from scene.keyframe import Keyframe
from scene.anchor import Anchor
from utils import (
    RGB2SH,
    depth2points,
    focal2fov,
    get_lapla_norm,
    getProjectionMatrix,
    inverse_sigmoid,
    align_poses,
    make_torch_sampler,
    psnr,
    rotation_distance,
    sixD2mtx,
)
from dataloaders.read_write_model import write_model


def _locked_scene_method(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self.lock:
            return fn(self, *args, **kwargs)

    return wrapped


def _make_rig_protocol_audit() -> dict[str, int]:
    audit = make_rig_triangulation_audit()
    audit.update({
        "mvs_partner_count_same_ts": 0,
        "spawn_count_train": 0,
        "spawn_count_test": 0,
        "spawn_count_tracking": 0,
        "spawn_skip_count_test": 0,
        "spawn_skip_count_tracking": 0,
    })
    return audit


class SceneModel:
    """
    Scene Model class that contains the scene's Gaussians, anchors, keyframes, and methods for rendering and optimization.
    """
    def __init__(
        self,
        width: int,
        height: int,
        args: Namespace,
        matcher: Matcher = None,
        inference_mode: bool = False,
    ):
        """
        Args:
            width: Width of the image.
            height: Height of the image.
            args: Arguments for the scene model. Should always have anchor_overlap, and training parameters if inference_mode is False.
            matcher: Matcher for the scene model. Defaults to None (if inference_mode is True).
            inference_mode: Whether we load the scene for visualization. Defaults to False.
        """
        self.width = width
        self.height = height
        self.matcher = matcher
        self.centre = torch.tensor([width / 2, height / 2], device="cuda")  # cx=cy=W/2 to match eqr_to_pinhole
        self.anchor_overlap = args.anchor_overlap
        self.use_rig = getattr(args, "use_rig", False)
        self.freeze_rig_poses = getattr(args, "freeze_rig_poses", False)
        # Rig mode owns one shared SE(3) pose per timestep (6D rotation parameter
        # plus 3D translation). Each virtual view derives pose as rel @ rig with
        # rel_t=0, so all views from the same source panorama remain co-centered.
        # When rig_optimizer is active, get_Rts() bypasses the Rt cache because
        # the shared poses can change every iteration.
        self.rig_R6D = torch.nn.ParameterList()
        self.rig_t = torch.nn.ParameterList()
        self.rig_optimizer = None
        self.optimization_thread = None

        try:
            import sys

            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            warnings.filterwarnings("ignore")
            self.lpips = lpips.LPIPS(net="vgg").cuda()
            sys.stdout = original_stdout
        except:
            self.lpips = None

        if not inference_mode:
            self.num_prev_keyframes_check = args.num_prev_keyframes_check
            self.active_sh_degree = args.sh_degree
            self.max_sh_degree = args.sh_degree
            self.lambda_dssim = args.lambda_dssim
            self.init_proba_scaler = args.init_proba_scaler
            self.max_active_keyframes = args.max_active_keyframes
            # (rig) recent-context window in TIMESTEPS; n_rig_views (=N) is set by
            # train.py from the rig config. n_kept_frames is then N-view-aware
            # (= n_kept_timesteps * N) instead of a fixed keyframe count, so a 6/12/
            # 15-view rig keeps the same number of recent TIMESTEPS live, not frames.
            self.n_rig_views = 1
            self.n_kept_timesteps = getattr(args, "n_kept_timesteps", 2)
            self.n_kept_frames = 20  # default until place_anchor_if_needed sets it
            self.use_last_frame_proba = args.use_last_frame_proba
            self.active_frames_cpu = []
            self.active_frames_gpu = []
            self.guided_mvs = GuidedMVS(args)
            self.lr_dict = {
                "xyz": {
                    "lr_init": args.position_lr_init,
                    "lr_decay": args.position_lr_decay,
                }
            }

            ## Initialize Gaussian parameters
            self.gaussian_params = {
                "xyz": {
                    "val": torch.empty(0, 3, device="cuda"),
                    "lr": args.position_lr_init,
                },
                "f_dc": {
                    "val": torch.empty(0, 1, 3, device="cuda"),
                    "lr": args.feature_lr,
                },
                "f_rest": {
                    "val": torch.empty(
                        0,
                        (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
                        3,
                        device="cuda",
                    ),
                    "lr": args.feature_lr / 20.0,
                },
                "scaling": {
                    "val": torch.empty(0, 3, device="cuda"),
                    "lr": args.scaling_lr,
                },
                "rotation": {
                    "val": torch.empty(0, 4, device="cuda"),
                    "lr": args.rotation_lr,
                },
                "opacity": {
                    "val": torch.empty(0, 1, device="cuda"),
                    "lr": args.opacity_lr,
                },
            }
            self.active_anchor = Anchor(self.gaussian_params)
            self.anchors = [self.active_anchor]
            ## Initialize optimizer
            self.reset_optimizer()

        self.keyframes = []
        self.anchor_weights = [1.0]
        self.f = 0.7 * width
        self.init_intrinsics()

        self.approx_cam_centres = None
        self.gt_Rts = torch.empty(0, 4, 4, device="cuda")
        self.gt_Rts_mask = torch.empty(0, device="cuda", dtype=bool)
        self.gt_f = self.f
        self.cached_Rts = torch.empty(0, 4, 4, device="cuda")
        self.valid_Rt_cache = torch.empty(0, device="cuda", dtype=torch.bool)
        self.sorted_frame_indices = None
        self.last_trained_id = 0
        self.valid_keyframes = torch.empty(0, dtype=torch.bool)
        # Serializes live viewer renders with Gaussian spawn/prune/optimization.
        # The lock is re-entrant because training renders call SceneModel.render()
        # from inside optimization/add_new_gaussians paths that also need the lock.
        self.lock = threading.RLock()
        self.inference_mode = inference_mode
        self.extra_metadata = {}
        self.rig_expected_timesteps = {
            "all": [],
            "train": [],
            "test": [],
            "tracking": [],
        }
        self.rig_failed_timesteps = []
        self.rig_completeness = {}
        self.rig_leakage_audit = _make_rig_protocol_audit()

        ## Initialize helpers for Gaussian initialization
        radius = 3
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1)
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.disc_kernel[0, 0, torch.sqrt(x**2 + y**2) <= radius + 0.5] = 1
        self.disc_kernel = self.disc_kernel.cuda() / self.disc_kernel.sum()

        self.uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, width), torch.arange(0, height), indexing="xy"
                ),
                dim=-1,
            )
            .float()
            .cuda()
        )

    def reset_optimizer(self):
        for key in self.gaussian_params:
            if not self.gaussian_params[key]["val"].requires_grad:
                self.gaussian_params[key]["val"].requires_grad = True
        self.optimizer = SparseGaussianAdam(
            self.gaussian_params, (0.5, 0.99), lr_dict=self.lr_dict
        )

    @property
    def xyz(self):
        return self.gaussian_params["xyz"]["val"]

    @property
    def f_dc(self):
        return self.gaussian_params["f_dc"]["val"]

    @property
    def f_rest(self):
        return self.gaussian_params["f_rest"]["val"]

    @property
    def scaling(self):
        return torch.exp(self.gaussian_params["scaling"]["val"])

    @property
    def rotation(self):
        return F.normalize(self.gaussian_params["rotation"]["val"])

    @property
    def opacity(self):
        return torch.sigmoid(self.gaussian_params["opacity"]["val"])

    @property
    def n_active_gaussians(self):
        return self.xyz.shape[0]

    @classmethod
    def from_scene(cls, scene_dir: str, args):
        with open(os.path.join(scene_dir, "metadata.json")) as f:
            metadata = json.load(f)

        width = metadata["config"]["width"]
        height = metadata["config"]["height"]
        scene_model = cls(width, height, args, inference_mode=True)
        scene_model.active_sh_degree = metadata["config"]["sh_degree"]
        scene_model.max_sh_degree = metadata["config"]["sh_degree"]

        # Load anchors
        scene_model.anchors = []
        for i in range(len(metadata["anchors"])):
            scene_model.anchors.append(
                Anchor.from_ply(
                    os.path.join(scene_dir, "point_clouds", f"anchor_{i}.ply"),
                    torch.tensor(metadata["anchors"][i]["position"]),
                    metadata["config"]["sh_degree"],
                )
            )

        scene_model.active_anchor = scene_model.anchors[0]

        # Load keyframes
        for i in range(len(metadata["keyframes"])):
            keyframe = Keyframe.from_json(metadata["keyframes"][i], i, width, height)
            scene_model.add_keyframe(keyframe)

        return scene_model

    @property
    def first_active_frame(self):
        return self.active_anchor.keyframe_ids[0]

    @property
    def last_active_frame(self):
        return self.active_anchor.keyframe_ids[-1]

    @property
    def n_active_keyframes(self):
        return self.last_active_frame - self.first_active_frame + 1

    def _active_optimization_frames(self):
        """Return active frame ids that may drive scene/rig optimization.

        In rig evaluation splits, `is_test=True` covers both metric test frames
        and tracking-only frames. They stay in the online stream for pose
        registration, but their RGB must not consume optimization iterations or
        update Gaussian/rig state. Non-rig keeps the upstream behavior.
        """
        if not self.use_rig:
            return self.active_frames_gpu
        train_frames = [
            i for i in self.active_frames_gpu
            if not self.keyframes[i].info.get("is_test", False)
        ]
        return train_frames

    def set_rig_view_count(self, n_views: int):
        """Set N for the rig and keep frame-count windows timestep-aligned."""
        self.n_rig_views = int(n_views)
        if self.use_rig:
            self.n_kept_frames = self.n_kept_timesteps * self.n_rig_views

    def set_rig_expected_timesteps(self, expected: dict[str, list[int]]):
        """Set the source-timestep universe used by completeness reports."""
        self.rig_expected_timesteps = {
            split: sorted({int(ts) for ts in expected.get(split, [])})
            for split in ("all", "train", "test", "tracking")
        }

    def record_rig_timestep_failure(
        self,
        source_ts: int,
        stream_idx: int,
        split: str,
        reason: str,
    ):
        """Record a timestep that failed online registration and was skipped."""
        self.rig_failed_timesteps.append({
            "source_ts": int(source_ts),
            "stream_idx": int(stream_idx),
            "split": str(split),
            "reason": str(reason),
        })

    def _classify_rig_partner(self, target_keyframe: Keyframe, partner_id: int) -> str:
        return classify_rig_triangulation_partner(
            self.keyframes, target_keyframe.info, partner_id
        )

    def rig_triangulation_allowed_ids(self, target_keyframe: Keyframe) -> list[int]:
        """Return split-safe, cross-timestep triangulation partners for a rig keyframe.

        `desc_kpts.matches` is persistent and may contain same-timestep,
        test/tracking, or temporary ids from online PnP. Count those candidates
        for diagnostics, but only return valid train partners from a different
        source timestep.
        """
        return collect_allowed_rig_triangulation_ids(
            self.keyframes,
            target_keyframe.info,
            target_keyframe.desc_kpts.matches.keys(),
            self.rig_leakage_audit,
        )

    def record_rig_triangulation_use(self, target_keyframe: Keyframe, partner_ids):
        """Record and assert the actual partner ids used by rig triangulation."""
        record_used_rig_triangulation_ids(
            self.keyframes,
            target_keyframe.info,
            partner_ids,
            self.rig_leakage_audit,
        )

    def record_rig_spawn_skip(self, keyframe: Keyframe):
        """Count non-train keyframes skipped before Gaussian state mutation."""
        split = keyframe.info.get("rig_eval_split", "test")
        key = (
            "spawn_skip_count_tracking"
            if split == "tracking"
            else "spawn_skip_count_test"
        )
        self.rig_leakage_audit[key] = self.rig_leakage_audit.get(key, 0) + 1

    def _latest_train_packet_frames(self):
        """Return train keyframes from the latest active rig timestep.

        Upstream OTF's `keyframe_id=-1` fast path means "train the latest
        image". In rig mode the latest image is only one arbitrary view of the
        latest timestamp packet, so using -1 gives that view extra gradient
        budget. The rig unit is the whole timestamp packet; sample uniformly
        from its train views instead.
        """
        if not self.use_rig:
            return []
        active_train = [
            i for i in self.active_frames_gpu
            if not self.keyframes[i].info.get("is_test", False)
            and self.keyframes[i].ts_idx is not None
        ]
        if not active_train:
            return []
        latest_ts = max(int(self.keyframes[i].ts_idx) for i in active_train)
        return [i for i in active_train if int(self.keyframes[i].ts_idx) == latest_ts]

    def _live_centres_for_keyframe_ids(self, frame_ids):
        """Return current camera centres for keyframe ids.

        `approx_cam_centres` stores initial centres and does not move when
        photometric BA changes rig poses. Anchor placement/render blending must
        use live centres or long loop sequences can associate frames with the
        wrong submap.
        """
        if len(frame_ids) == 0:
            return torch.empty(0, 3, device="cuda")
        if self.use_rig and len(self.rig_R6D) > 0:
            centres_ts = self.get_live_rig_centres()
            centres = []
            for frame_id in frame_ids:
                kf = self.keyframes[int(frame_id)]
                if kf.ts_idx is not None:
                    centres.append(centres_ts[int(kf.ts_idx)])
                else:
                    centres.append(kf.get_centre().detach())
            return torch.stack(centres, dim=0)
        return torch.stack(
            [self.keyframes[int(frame_id)].get_centre().detach() for frame_id in frame_ids],
            dim=0,
        )

    @_locked_scene_method
    def optimization_step(self, finetuning=False):
        if len(self.xyz) == 0:
            return
        # Select which keyframe to train on
        # We train on the latest keyframe with self.use_last_frame_proba probability or a random keyframe otherwise
        active_optim_frames = self._active_optimization_frames()
        if len(active_optim_frames) == 0:
            return
        if (
            np.random.rand() > self.use_last_frame_proba
            or self.last_trained_id == -1
            or finetuning
        ):
            keyframe_id = np.random.choice(active_optim_frames)
        else:
            if self.use_rig:
                packet = self._latest_train_packet_frames()
                keyframe_id = (
                    int(np.random.choice(packet))
                    if packet else int(np.random.choice(active_optim_frames))
                )
            else:
                keyframe_id = -1
        keyframe = self.keyframes[keyframe_id]
        lvl = keyframe.pyr_lvl

        # Zero gradients
        keyframe.zero_grad()
        self.optimizer.zero_grad()
        if self.rig_optimizer is not None:  # (rig) shared per-ts pose
            self.rig_optimizer.zero_grad()

        # Render image and depth
        render_pkg = self.render_from_id(
            keyframe_id, pyr_lvl=lvl, bg=torch.rand(3, device="cuda")
        )
        image = render_pkg["render"]
        invdepth = render_pkg["invdepth"]

        gt_image = keyframe.image_pyr[lvl]
        mono_idepth = keyframe.get_mono_idepth(lvl)

        # Mask image and depth if necessary
        if keyframe.mask_pyr is not None:
            image = image * keyframe.mask_pyr[lvl]
            gt_image = gt_image * keyframe.mask_pyr[lvl]
            invdepth = invdepth * keyframe.mask_pyr[lvl]
            mono_idepth = mono_idepth * keyframe.mask_pyr[lvl]

        # Loss
        l1_loss = (image - gt_image).abs().mean()
        ssim_loss = 1 - fused_ssim(image[None], gt_image[None])
        depth_loss = (invdepth - mono_idepth).abs().mean()
        # Mono depth has the wrong absolute scale, but still acts as an ordinal
        # prior for Gaussian depth ordering. Its causal role is measured by the
        # documented `--depth_loss_weight_init 0` ablation, not assumed here.
        loss = (
            self.lambda_dssim * ssim_loss
            + (1 - self.lambda_dssim) * l1_loss
            + keyframe.depth_loss_weight * depth_loss
        )
        loss.backward()

        # Optimizers
        with torch.no_grad():
            # Pose optimization
            keyframe.step()

            # Skip the scene optimization if the current keyframe is a test keyframe
            if not keyframe.info["is_test"]:
                # Scene Gaussian optimization
                self.optimizer.step(
                    render_pkg["visibility_filter"], render_pkg["radii"].shape[0]
                )

                # (rig) ONE shared rig-pose step moves all N views of this
                # timestep together → rel_t stays exactly 0. Gated on not-is_test
                # (holdout never perturbs the rig pose) and the freeze flag.
                if (getattr(keyframe, "is_rig_mode", False)
                        and self.rig_optimizer is not None
                        and not self.freeze_rig_poses):
                    self.rig_optimizer.step()

                # NOTE: raw_scaling.clamp was here but REMOVED.
                # Clamping scaling created Adam optimizer state mismatch that
                # paradoxically *caused* the cudaErrorIllegalAddress crash at
                # iter>=20. Removing it resolved the crash.
                raw_xyz = self.gaussian_params["xyz"]["val"]
                if raw_xyz.numel() > 0:
                    # Scene is at ~0.1-unit scale; 100 is a generous envelope
                    # that still prevents means2D from overflowing the
                    # rasterizer's tile arithmetic.
                    raw_xyz.data.clamp_(min=-100.0, max=100.0)
                # Guard against stray NaNs from numerically unstable updates.
                for key in ("xyz", "scaling", "rotation", "opacity", "f_dc"):
                    if key in self.gaussian_params:
                        v = self.gaussian_params[key]["val"]
                        if v.numel() > 0:
                            torch.nan_to_num_(v.data, nan=0.0, posinf=0.0, neginf=0.0)

            keyframe.latest_invdepth = render_pkg["invdepth"].detach()

        self.valid_Rt_cache[keyframe_id] = False
        self.last_trained_id = keyframe_id

    def optimization_loop(self, n_iters: int, run_until_interupt: bool = False):
        """
        Runs at least n_iters optimization steps.
        If run_until_interupt, also runs until join_optimization_thread is called (Useful to run the optimization until the next keyframe is added in streaming mode).
        """
        self.interupt_optimization = False
        i = 0
        while i < n_iters or (run_until_interupt and not self.interupt_optimization): 
            self.optimization_step()
            i += 1
        
    def join_optimization_thread(self):
        """
        Interupts the optimization loop and waits for the thread to finish.
        """
        if self.optimization_thread is not None:
            self.interupt_optimization = True
            self.optimization_thread.join()
            self.optimization_thread = None
    
    def optimize_async(self, n_iters: int):
        """
        Starts an optimization thread that runs at least n_iters optimization steps.
        """
        self.join_optimization_thread()
        self.optimization_thread = threading.Thread(
            target=self.optimization_loop, args=(n_iters, True)
        )
        self.optimization_thread.start()

    @torch.no_grad()
    def _is_metric_test_keyframe(self, keyframe: Keyframe) -> bool:
        """True only for frames that should enter held-out image metrics."""
        if self.use_rig:
            return keyframe.info.get("rig_eval_split") == "test"
        return bool(keyframe.info.get("is_test", False))

    @torch.no_grad()
    def harmonize_test_exposure(self):
        """Harmonizes the exposure matrices of test keyframes by averaging the exposure of the previous and next keyframes."""
        for index, keyframe in enumerate(self.keyframes):
            if self._is_metric_test_keyframe(keyframe):
                idxm = index - 1 if index != 0 else 1
                idxp = (
                    index + 1
                    if index != len(self.keyframes) - 1
                    else len(self.keyframes) - 2
                )
                keyframe.exposure = (
                    self.keyframes[idxm].exposure + self.keyframes[idxp].exposure
                ) / 2

    @torch.no_grad()
    def evaluate(self, eval_poses=False, with_LPIPS=False, all=False):
        # Make sure test keyframes have similar exposure matrices compared to their neighbors
        self.harmonize_test_exposure()

        # Compute image quality metrics
        metrics = {"PSNR": 0, "SSIM": 0}
        if with_LPIPS:
            metrics["LPIPS"] = 0
        n_test_frames = 0
        start_index = 0 if all else self.active_anchor.keyframe_ids[0]
        for index, keyframe in enumerate(self.keyframes[start_index:]):
            if self._is_metric_test_keyframe(keyframe):
                gt_image = keyframe.get_eval_image().cuda()
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = render_pkg["render"]
                mask = (
                    keyframe.mask_pyr[0].cuda()
                    if keyframe.mask_pyr is not None
                    else torch.ones_like(image[:1] > 0)
                )
                mask = mask.expand_as(image)
                image = image * mask
                gt_image = gt_image * mask
                metrics["PSNR"] += psnr(image[mask], gt_image[mask])
                metrics["SSIM"] += fused_ssim(
                    image[None], gt_image[None], train=False
                ).item()
                if with_LPIPS and self.lpips is not None:
                    metrics["LPIPS"] += self.lpips(
                        image[None] * 2 - 1,
                        gt_image[None] * 2 - 1,
                    ).item()
                n_test_frames += 1

        if n_test_frames > 0:
            for metric in metrics:
                metrics[metric] /= n_test_frames
        else:
            metrics = {}

        # Compute pose errors
        if eval_poses:
            Rts = self.get_Rts()
            gt_Rts = self.get_gt_Rts(align=False)
            if len(Rts) == len(gt_Rts):
                Rts_aligned = torch.linalg.inv(align_poses(Rts, gt_Rts))
                gt_Rts = torch.linalg.inv(gt_Rts)
                R_error = rotation_distance(Rts_aligned[:, :3, :3], gt_Rts[:, :3, :3])
                t_error = (Rts_aligned[:, :3, 3] - gt_Rts[:, :3, 3]).norm(dim=-1)

                metrics["R°"] = R_error.mean().item() * 180 / math.pi
                metrics["t"] = t_error.mean().item()

        return metrics

    @torch.no_grad()
    def save_test_frames(self, out_dir):
        self.harmonize_test_exposure()
        os.makedirs(out_dir, exist_ok=True)
        for keyframe in self.keyframes:
            if self._is_metric_test_keyframe(keyframe):
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = torch.clamp(render_pkg["render"], 0, 1) * 255
                image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # rig mode names are extension-less keys (e.g. "View__ts00001");
                # fall back to .png so cv2.imwrite has a writer.
                name = keyframe.info["name"]
                if not os.path.splitext(name)[-1]:
                    name = name + ".png"
                is_jpeg = os.path.splitext(name)[-1].lower() in [".jpg", ".jpeg"]
                write_flag = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if is_jpeg else []
                cv2.imwrite(os.path.join(out_dir, name), image, write_flag)

    def render_from_id(
        self,
        keyframe_id,
        pyr_lvl=0,
        scaling_modifier=1,
        bg=torch.zeros(3, device="cuda"),
    ):
        """
        Render the scene from a given keyframe id at a specified resolution level (pyr_lvl).
        Applies the exposure matrix of the keyframe to the rendered image.
        """
        keyframe = self.keyframes[keyframe_id]
        view_matrix = keyframe.get_Rt().transpose(0, 1)
        scale = 2**pyr_lvl
        width, height = self.width // scale, self.height // scale
        render_pkg = self.render(width, height, view_matrix, scaling_modifier, bg)
        render_pkg["render"] = (
            keyframe.exposure[:3, :3] @ render_pkg["render"].view(3, -1)
        ) + keyframe.exposure[:3, 3, None]
        render_pkg["render"] = render_pkg["render"].clamp(0, 1).view(3, height, width)
        return render_pkg

    @_locked_scene_method
    def render(
        self,
        width: int,
        height: int,
        view_matrix: torch.Tensor,
        scaling_modifier: float,
        bg: torch.Tensor = torch.zeros(3, device="cuda"),
        top_view: bool = False,
        fov_x: float = None,
        fov_y: float = None,
    ):
        cam_centre = view_matrix.detach().inverse()[3, :3]

        # Use the scene's intrinsic parameters if not provided
        if fov_x is None and fov_y is None:
            tanfovx, tanfovy = self.tanfovx, self.tanfovy
            projection_matrix = self.projection_matrix
        # Use the provided FOV values
        elif fov_x is not None and fov_y is not None:
            tanfovx = math.tan(fov_x * 0.5)
            tanfovy = math.tan(fov_y * 0.5)
            projection_matrix = (
                getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y)
                .transpose(0, 1)
                .cuda()
            )
        else:
            raise ValueError("Both fov_x and fov_y should be provided or neither.")

        raster_settings = GaussianRasterizationSettings(
            height,
            width,
            tanfovx,
            tanfovy,
            bg,
            1 if top_view else scaling_modifier,
            projection_matrix,
            self.active_sh_degree,
            cam_centre,
            False,
            False,
        )
        rasterizer = GaussianRasterizer(raster_settings)
        with self.lock:
            # Load and blend anchors if in inference mode 
            if self.inference_mode and not top_view:
                self.gaussian_params, self.anchor_weights = Anchor.blend(cam_centre, self.anchors, self.anchor_overlap)
            screenspace_points = torch.zeros_like(self.xyz, requires_grad=True)
            if self.xyz.shape[0] > 0:
                # Set constant scaling and opacity to visualize the Gaussians' positions in the top view
                if top_view:
                    scaling = torch.ones_like(self.scaling) * scaling_modifier
                    opacity = torch.ones_like(self.opacity)
                else:
                    scaling = self.scaling
                    opacity = self.opacity
                color, invdepth, mainGaussID, radii = rasterizer(
                    self.xyz,
                    screenspace_points,
                    opacity,
                    self.f_dc,
                    self.f_rest,
                    scaling,
                    self.rotation,
                    view_matrix,
                )
            else:
                # If no Gaussians are present, return empty tensors
                color = torch.zeros(3, height, width, device="cuda")
                invdepth = torch.zeros(1, height, width, device="cuda")
                mainGaussID = torch.zeros(
                    1, height, width, device="cuda", dtype=torch.int32
                )
                radii = torch.zeros(1, height, width, device="cuda")
        return {
            "render": color,
            "invdepth": invdepth,
            "mainGaussID": mainGaussID,
            "radii": radii,
            "visibility_filter": radii > 0,
            "screenspace_points": screenspace_points,
        }

    def get_closest_by_cam(self, cam_centre, k=3):
        closest_anchors = []
        closest_anchors_ids = []
        offset = 0
        approx_cam_centres = self.approx_cam_centres.clone()
        for l in range(min(k, len(self.anchors))):
            if approx_cam_centres.shape[0] == 0:
                break
            dists = torch.linalg.norm(approx_cam_centres - cam_centre[None], dim=-1)
            min_dist, min_id = torch.min(dists, dim=0)

            if min_dist < 1e9:
                for anchor_id, anchor in enumerate(self.anchors):
                    if min_id in anchor.keyframe_ids:
                        closest_anchors.append(anchor)
                        closest_anchors_ids.append(anchor_id)
                        approx_cam_centres[
                            anchor.keyframe_ids[0] : anchor.keyframe_ids[-1] + 1
                        ] = 1e9
                        break

        return closest_anchors, closest_anchors_ids

    @torch.no_grad()
    def get_live_rig_centres(self):
        """Return one live camera centre per rig timestep.

        In the rotation-only rig, all views in a timestep share the same optical
        centre:
            C_view = -(R_rel R_rig)^T (R_rel t_rig) = -R_rig^T t_rig.
        Computing this once per timestep avoids calling Keyframe.get_centre() for
        every view/keyframe candidate during neighbour selection.
        """
        if len(self.rig_R6D) == 0:
            return torch.empty(0, 3, device="cuda")
        rig_R6D = torch.stack([p.detach() for p in self.rig_R6D], dim=0)
        rig_t = torch.stack([p.detach() for p in self.rig_t], dim=0)
        rig_R = sixD2mtx(rig_R6D)
        return -(rig_R.transpose(1, 2) @ rig_t[..., None]).squeeze(-1)

    @torch.no_grad()
    def get_prev_keyframes(self, n: int, update_3dpts: bool, desc_kpts: DescribedKeypoints = None, exclude_ts: int = None, target_centre: torch.Tensor = None, require_dense: bool = False):
        """
        Get the n previous keyframes that are the closest to the last
        If desc_kpts is not None, we find the previous keyframes that have the most matches with desc_kpts. The search window is given by self.num_prev_keyframes_check
        If exclude_ts is not None (rig mode), keyframes sharing that rig timestamp are
        dropped from the candidate pool *before* selection: the N views of one timestep
        share the rig's optical center (zero baseline), so using them as triangulation/
        MVS partners yields degenerate depth (conventions §7). They are the nearest
        neighbours by camera-center distance, so a post-hoc filter would leave none —
        the exclusion must happen on the candidate pool.
        If target_centre is given, neighbours are sorted by distance to THAT live
        camera centre instead of the cached global sorted_frame_indices (which is
        keyed to the LAST-added keyframe's *initial* pose). For spawning Gaussians
        from a non-last keyframe — or any keyframe whose pose moved during
        photometric optimization — the global order picks the wrong neighbours,
        causing bootstrap/reboot/non-last-spawn order dependence.
        """
        # Make sure the optimization thread is not running
        self.join_optimization_thread()

        if target_centre is not None:
            # Candidate pool = ALL keyframes; we re-sort by live-centre distance
            # AFTER the holdout / same-ts filters below, recomputing each
            # candidate's centre from its CURRENT pose (not approx_cam_centres,
            # which is keyed to initial poses). Sorting the cached centres would
            # leave live-target vs stale-candidates — only a half fix.
            candidate_idx = torch.arange(len(self.keyframes), dtype=torch.long)
        else:
            candidate_idx = self.sorted_frame_indices
        if self.use_rig:
            # Holdout/test keyframes must never be triangulation / MVS / PnP
            # partners: their geometry would leak into the poses & depths of the
            # training views, inflating the held-out metric. Drop them from the
            # candidate pool up-front (covers both the incremental PnP selection
            # and the guided-MVS neighbour selection).
            keep_train = torch.tensor(
                [
                    (
                        not self.keyframes[int(i)].info.get("is_test", False)
                        and self.keyframes[int(i)].info.get("rig_eval_split") == "train"
                    )
                    for i in candidate_idx
                ],
                device=candidate_idx.device,
            )
            candidate_idx = candidate_idx[keep_train]
        if exclude_ts is not None:
            keep = torch.tensor(
                [self.keyframes[int(i)].info.get("source_ts") != exclude_ts for i in candidate_idx],
                device=candidate_idx.device,
            )
            candidate_idx = candidate_idx[keep]
        if require_dense:
            keep_dense = torch.tensor(
                [self.keyframes[int(i)].has_dense_cache for i in candidate_idx],
                device=candidate_idx.device,
            )
            candidate_idx = candidate_idx[keep_dense]

        # (rig) Now that holdout/same-ts candidates are removed, sort the survivors
        # by distance to the TARGET keyframe's LIVE centre. For rig keyframes the
        # live centre is view-independent, so compute it in one batched pass per
        # timestep and index by each candidate's ts_idx instead of calling
        # Keyframe.get_centre() once per candidate.
        if target_centre is not None and len(candidate_idx) > 0:
            if self.use_rig and len(self.rig_R6D) > 0:
                centres_ts = self.get_live_rig_centres()
                candidate_ts_idx = torch.tensor(
                    [self.keyframes[int(i)].ts_idx for i in candidate_idx],
                    dtype=torch.long,
                    device=centres_ts.device,
                )
                centres = centres_ts[candidate_ts_idx]
            else:
                centres = torch.stack(
                    [self.keyframes[int(i)].get_centre().detach() for i in candidate_idx],
                    dim=0,
                )
            target = target_centre.detach().to(centres)
            order = torch.argsort(
                torch.linalg.vector_norm(centres - target[None], dim=-1)
            )
            candidate_idx = candidate_idx[order.cpu()]

        # Look for the previous keyframes with the most matches with desc_kpts (if provided)
        if desc_kpts is not None and len(candidate_idx) >= n:
            # N-view scaling: the candidate pool is the closest-by-center keyframes, but
            # for an N-view rig each timestep contributes N keyframes at the SAME center,
            # so a fixed window only spans ~window/N timesteps and yields ~window/N
            # same-view candidates per view — starving the per-view match selection and
            # causing incremental pose drift. Scale the pool by N so each view sees a
            # mono-equivalent number of same-view keyframes.
            check_window = self.num_prev_keyframes_check * (self.n_rig_views if self.use_rig else 1)
            n_ckecks = min(check_window, len(candidate_idx))
            keyframes_indices_to_check = candidate_idx[:n_ckecks]
            n_matches = torch.zeros(len(keyframes_indices_to_check), device="cuda")
            for i, index in enumerate(keyframes_indices_to_check):
                n_matches[i] = self.matcher.evaluate_match(
                    self.keyframes[index].desc_kpts, desc_kpts
                )
            _, top_indices = torch.topk(n_matches, n)
            prev_keyframes_indices = keyframes_indices_to_check[top_indices.cpu()]
        # If desc_kpts is not provided, we take the n closest keyframes
        else:
            prev_keyframes_indices = candidate_idx[:n]
        prev_keyframes = [self.keyframes[i] for i in prev_keyframes_indices]

        # Re-run triangulation if necessary
        if update_3dpts:
            for keyframe in prev_keyframes:
                keyframe.update_3dpts(self.keyframes)
        return prev_keyframes

    def get_Rts(self):
        # (rig) rig_R6D/rig_t are stepped every photometric iteration, so a cached
        # Rt would carry a stale autograd graph and be numerically out of date.
        # Recompute from the live shared rig pose each call.
        if self.rig_optimizer is not None:
            return torch.stack([kf.get_Rt() for kf in self.keyframes])
        invalid_ids = torch.where(~self.valid_Rt_cache)[0]
        if len(invalid_ids) > 0:
            for keyframe_id in invalid_ids:
                self.cached_Rts[keyframe_id] = self.keyframes[keyframe_id].get_Rt()
            self.valid_Rt_cache[invalid_ids] = True
        return self.cached_Rts

    def get_gt_Rts(self, align):
        n_poses = min(self.gt_Rts_mask.shape[0], self.cached_Rts.shape[0])
        if align and n_poses > 0:
            Rts = self.get_Rts()[:n_poses][self.gt_Rts_mask[:n_poses]]
            return align_poses(self.gt_Rts[: len(Rts)], Rts)
        else:
            return self.gt_Rts

    def make_dummy_ext_tensor(self):
        return {
            "xyz": self.xyz[:0].detach(),
            "f_dc": self.f_dc[:0].detach(),
            "f_rest": self.f_rest[:0].detach(),
            "opacity": self.opacity[:0].detach(),
            "scaling": self.scaling[:0].detach(),
            "rotation": self.rotation[:0].detach(),
        }

    def reset(self, keyframe_id: int = -1):
        """Remove the Gaussians that are visible in the given keyframe."""
        valid_mask = self.opacity[:, 0] > 0.05
        render_pkg = self.render_from_id(keyframe_id)
        valid_mask[render_pkg["visibility_filter"]] = False
        self.optimizer.add_and_prune(self.make_dummy_ext_tensor(), valid_mask)

    def add_new_gaussians(self, keyframe_id: int = -1):
        """Use one keyframe to add new Gaussians to the scene model."""
        self.add_new_gaussians_for_keyframes([keyframe_id])

    @torch.no_grad()
    @_locked_scene_method
    def add_new_gaussians_for_keyframes(self, keyframe_ids):
        """Plan all requested keyframes first, then commit once.

        For rig mode this is the timestamp-packet contract: all N views of a
        timestep see the same pre-spawn Gaussian state, and the optimizer
        mutation is committed once after planning. This is atomic scene-state
        mutation, not a bitwise view-order-invariance guarantee for stochastic
        per-view proposals.
        """
        plans = []
        for keyframe_id in keyframe_ids:
            plan = self._plan_new_gaussians(keyframe_id)
            if plan is not None:
                plans.append(plan)
        self._commit_new_gaussian_plans(plans)

    def _plan_new_gaussians(self, keyframe_id: int = -1):
        """Build a spawn plan without mutating the Gaussian optimizer."""
        keyframe = self.keyframes[keyframe_id]
        if keyframe.info.get("is_test", False):
            if self.use_rig:
                self.record_rig_spawn_skip(keyframe)
            return None

        ## align the keyframe's depth
        if keyframe.desc_kpts.has_pt3d.sum() == 0:
            keyframe.update_3dpts(self.keyframes)
        keyframe.align_depth()

        # Live camera centre of the TARGET keyframe (from its current pose, not the
        # cached approx_centre keyed to the initial pose). Used for MVS-neighbour
        # selection, Gaussian scale, and the huge-Gaussian prune below.
        cam_centre = keyframe.get_centre().detach()

        ## Get the pixel-wise probability to add a Gaussian
        img = keyframe.image_pyr[0]
        img = F.avg_pool2d(img, 2)
        img = F.interpolate(
            img[None], (self.height, self.width), mode="bilinear", align_corners=True
        )[0]
        init_proba = get_lapla_norm(img, self.disc_kernel) # eq. 1

        if keyframe.mask_pyr is not None:
            dilated_mask = (
                F.conv2d(
                    keyframe.mask_pyr[0][None].float(), self.disc_kernel, padding="same"
                )[0, 0]
                >= 0.99
            )
            init_proba *= dilated_mask

        ## Compute the penalty based on the rendering from the new keyframe's point of view
        penalty = 0
        rendered_depth = None
        if self.xyz.shape[0] > 0:
            render_pkg = self.render_from_id(keyframe_id)
            render = render_pkg["render"]
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)
            penalty = get_lapla_norm(render, self.disc_kernel)

        ## Define which pixels should become Gaussians
        init_proba *= self.init_proba_scaler
        penalty *= self.init_proba_scaler
        sample_mask = torch.rand_like(init_proba) < init_proba - penalty # eq. 3

        sampled_uv = self.uv[sample_mask]
        # The guided-MVS branch needs selected pixels AND enough cross-ts
        # neighbours. If either is missing we skip ONLY the MVS branch (a CUDA-
        # launch crash guard: grid_size = ceil(0) = 0) — the triangulated match
        # Gaussians below are still spawned. mvs_pts/depth/accurate_mask stay
        # empty and the downstream concatenations remain correct.
        mvs_pts = torch.empty(0, 3, device="cuda")
        depth = torch.empty(0, device="cuda")
        accurate_mask = torch.empty(0, dtype=torch.bool, device="cuda")
        run_mvs = sampled_uv.numel() > 0
        if run_mvs:
            ## Initialize positions
            # Get the samples' depth with guided stereo matching.
            # Neighbour selection uses the TARGET keyframe's live centre (not the
            # global sorted_frame_indices keyed to the last-added keyframe's initial
            # pose) so spawning from a non-last keyframe — bootstrap, reboot, or any
            # rig view — picks the geometrically correct MVS partners.
            prev_KFs = self.get_prev_keyframes(
                self.guided_mvs.n_cams + 1, update_3dpts=False,
                exclude_ts=keyframe.info.get("source_ts"),
                target_centre=cam_centre,
                require_dense=True,
            )
            for i, prev_keyframe in enumerate(prev_KFs):
                if keyframe.index == prev_keyframe.index:
                    prev_KFs.pop(i)
                    break
            # (rig) same-ts exclusion safety net: prev_KFs must never share this
            # keyframe's rig timestamp (zero baseline -> degenerate depth).
            _ex_ts = keyframe.info.get("source_ts")
            if _ex_ts is not None:
                same_ts_mvs = sum(p.info.get("source_ts") == _ex_ts for p in prev_KFs)
                self.rig_leakage_audit["mvs_partner_count_same_ts"] += same_ts_mvs
                assert same_ts_mvs == 0, \
                    "zero-baseline same-ts keyframe leaked into guided_mvs partners"
            # guided_mvs' CUDA kernel is compiled for a fixed NUM_CAMS = n_cams and
            # indexes exactly that many neighbours: passing FEWER reads out of bounds.
            # If the cross-ts pool is too small, skip only the MVS branch (keep match
            # Gaussians) rather than discarding the whole keyframe.
            if len(prev_KFs) < self.guided_mvs.n_cams:
                run_mvs = False
        if run_mvs:
            depth, accurate_mask = self.guided_mvs(sampled_uv, keyframe, prev_KFs)
            valid_mask = (keyframe.sample_conf(sampled_uv) > 0.5) * (depth > 1e-6)
            sample_mask[sample_mask.clone()] = valid_mask
            depth = depth[valid_mask]
            sampled_uv = sampled_uv[valid_mask]
            accurate_mask = accurate_mask[valid_mask]
        else:
            # No MVS samples: zero the pixel mask so all the sample_mask-indexed
            # ops below become empty (match Gaussians are unaffected).
            sample_mask.zero_()
            sampled_uv = sampled_uv[:0]

        # Remove Gaussians that are coarser than the newpoints. In the packet
        # path this is only a keep-mask plan: committing here would make later
        # views in the same timestamp see a different scene than earlier views.
        coarse_valid_gs_mask = None
        if len(self.xyz) > 0:
            main_gaussians_map = render_pkg["mainGaussID"]
            accurate_sample_mask = sample_mask.clone()
            accurate_sample_mask[accurate_sample_mask.clone()] = accurate_mask
            selected_main_gaussians = main_gaussians_map[:, accurate_sample_mask]
            ids, counts = torch.unique(
                selected_main_gaussians[selected_main_gaussians >= 0],
                return_counts=True,
            )
            coarse_valid_gs_mask = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
            coarse_valid_gs_mask[ids] = counts < 10

        # Check for occlusions
        if rendered_depth is not None:
            valid_mask = depth < rendered_depth[sample_mask]
            sample_mask[sample_mask.clone()] = valid_mask
            depth = depth[valid_mask]
            sampled_uv = sampled_uv[valid_mask]
            accurate_mask = accurate_mask[valid_mask]

        # Get the samples' 3D positions (MVS branch only if we have valid samples)
        if sampled_uv.shape[0] > 0:
            mvs_pts = depth2points(sampled_uv, depth.unsqueeze(-1), self.f, self.centre)
            mvs_pts = (mvs_pts - keyframe.get_t()) @ keyframe.get_R()
        # Add points from matching (these survive even when MVS was skipped)
        match_pts = keyframe.desc_kpts.pts3d[keyframe.desc_kpts.has_pt3d]
        if mvs_pts.shape[0] == 0 and match_pts.shape[0] == 0:
            return None
        new_pts = torch.cat([mvs_pts, match_pts], dim=0)

        ## Initialize Colour
        f_dc = img[:, sample_mask]
        match_sampler = keyframe.desc_kpts.kpts[keyframe.desc_kpts.has_pt3d]
        match_sampler = make_torch_sampler(match_sampler, self.width, self.height)
        match_colors = F.grid_sample(
            img[None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(3, -1)
        f_dc = torch.cat([f_dc, match_colors], dim=1)
        f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1))

        ## Initialize Scales
        sampled_init_proba = init_proba[sample_mask]
        match_init_proba = F.grid_sample(
            init_proba[None, None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(-1)
        sampled_init_proba = torch.cat([sampled_init_proba, match_init_proba], dim=0)
        # Expected distance to the nearest neighbour (eq. 4)
        scales = 1 / (torch.sqrt(sampled_init_proba))
        scales.clamp_(1, self.width / 10)
        # Scale by the distance to the camera centre
        scales.mul_(1 / self.f)
        scales *= torch.linalg.vector_norm(
            new_pts - cam_centre[None], dim=-1
        )
        scales = torch.log(scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)

        ## Initialize opacities
        opacities = torch.ones(f_dc.shape[0], 1, device="cuda")
        # Lower inital opacity depending for innacurate points
        opacities[: sampled_uv.shape[0]] *= (
            0.07 * accurate_mask[..., None] + 0.02 * ~accurate_mask[..., None]
        )
        # High opacity for triangulated Gaussians
        opacities[sampled_uv.shape[0] :] *= 0.2
        opacities = inverse_sigmoid(opacities)

        ## Initialize SH, rotations as identity
        f_rest = torch.zeros(
            f_dc.shape[0],
            (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
            3,
            device="cuda",
        )
        rots = torch.zeros(f_dc.shape[0], 4, device="cuda")
        rots[:, 0] = 1

        ## Get which Gaussians should be pruned
        if self.xyz.shape[0] > 0:
            # Only keep Gaussians with non neglectible opacity
            valid_gs_mask = self.opacity[:, 0] > 0.05

            # Discard huge Gaussians
            dist = torch.linalg.vector_norm(
                self.xyz - cam_centre[None], dim=-1
            )
            screen_size = self.f * self.scaling.max(dim=-1)[0] / dist
            valid_gs_mask *= screen_size < 0.5 * self.width
            if coarse_valid_gs_mask is not None:
                valid_gs_mask &= coarse_valid_gs_mask
        else:
            valid_gs_mask = torch.ones(0, device="cuda", dtype=torch.bool)

        ## Append the new Gaussians
        extension_tensors = {
            "xyz": new_pts,
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opacities,
            "scaling": scales,
            "rotation": rots,
        }
        if self.use_rig:
            split = keyframe.info.get("rig_eval_split", "test")
            key = (
                f"spawn_count_{split}"
                if split in ("train", "test", "tracking")
                else "spawn_count_test"
            )
            self.rig_leakage_audit[key] = self.rig_leakage_audit.get(key, 0) + 1
        return {
            "extension_tensors": extension_tensors,
            "valid_gs_mask": valid_gs_mask,
        }

    def _commit_new_gaussian_plans(self, plans):
        """Apply one or more spawn plans with a single optimizer mutation."""
        plans = [plan for plan in plans if plan is not None]
        if not plans:
            return
        keys = list(plans[0]["extension_tensors"].keys())
        extension_tensors = {
            key: torch.cat([plan["extension_tensors"][key] for plan in plans], dim=0)
            for key in keys
        }
        valid_gs_mask = plans[0]["valid_gs_mask"]
        for plan in plans[1:]:
            if plan["valid_gs_mask"].shape != valid_gs_mask.shape:
                raise RuntimeError(
                    "packet spawn plan saw a different Gaussian count before commit; "
                    "the plan/commit invariant was violated"
                )
            valid_gs_mask = valid_gs_mask & plan["valid_gs_mask"]
        old_count = self.xyz.shape[0]
        if valid_gs_mask.shape[0] != old_count:
            raise RuntimeError(
                f"packet spawn keep mask has {valid_gs_mask.shape[0]} entries, "
                f"but scene has {old_count} Gaussians"
            )
        self.optimizer.add_and_prune(extension_tensors, valid_gs_mask)

    def init_intrinsics(self):
        self.FoVx = focal2fov(self.f, self.width)
        self.FoVy = focal2fov(self.f, self.height)
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)
        self.projection_matrix = (
            getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1)
            .cuda()
        )

    def _move_keyframe_to_cpu(self, frame_id: int):
        self.keyframes[frame_id].to("cpu")
        if self.use_rig:
            self.keyframes[frame_id].release_dense_cache()
        self.active_frames_cpu.append(frame_id)
        self.active_frames_gpu.remove(frame_id)

    def move_rand_keyframe_to_cpu(self):
        """Move old active keyframes to CPU memory.

        Non-rig preserves upstream's single-frame eviction. Rig mode evicts an
        entire old timestamp packet so the dense active set does not contain
        partial 12-view packets, which would bias guided-MVS and optimization
        toward whichever views happened to remain resident.
        """
        protected_tail = set(self.active_frames_gpu[-self.n_kept_frames:])
        if self.use_rig:
            eligible_by_ts = {}
            for frame_id in self.active_frames_gpu:
                if frame_id in protected_tail:
                    continue
                ts = self.keyframes[frame_id].info.get("source_ts")
                if ts is None:
                    continue
                eligible_by_ts.setdefault(ts, []).append(frame_id)
            if not eligible_by_ts:
                return
            chosen_ts = np.random.choice(list(eligible_by_ts.keys()))
            for frame_id in list(eligible_by_ts[chosen_ts]):
                if frame_id in self.active_frames_gpu:
                    self._move_keyframe_to_cpu(frame_id)
            return

        frame_id = np.random.choice(self.active_frames_gpu[:-self.n_kept_frames])
        self._move_keyframe_to_cpu(frame_id)

    def move_rand_keyframe_to_gpu(self):
        """Move a random keyframe to GPU memory"""
        if len(self.active_frames_cpu) > 0:
            candidates = [
                frame_id for frame_id in self.active_frames_cpu
                if self.keyframes[frame_id].has_dense_cache
            ]
            if len(candidates) == 0:
                return
            frame_id = np.random.choice(candidates)
            self.keyframes[frame_id].to("cuda")
            self.active_frames_gpu.insert(0, frame_id)
            self.active_frames_cpu.remove(frame_id) 

    def register_rig_poses(self, rig_R6D_init, rig_t_init, lr):
        """(rig) Bootstrap setup. Wrap each per-ts (3,2) rotation + (3,) translation
        as an nn.Parameter owned by a dedicated rig_optimizer. View poses derive
        from these as rel @ rig, so the N views of a timestep stay rigidly
        co-centered (rel_t=0) through photometric optimization."""
        from scene.optimizers import BaseAdam
        assert len(self.rig_R6D) == 0, "rig poses already registered"
        params = {}
        for i in range(rig_R6D_init.shape[0]):
            p_R = torch.nn.Parameter(rig_R6D_init[i].clone())
            p_t = torch.nn.Parameter(rig_t_init[i].clone())
            self.rig_R6D.append(p_R)
            self.rig_t.append(p_t)
            params[f"rig_R6D_{i}"] = {"val": p_R, "lr": lr}
            params[f"rig_t_{i}"] = {"val": p_t, "lr": lr}
        self.rig_optimizer = BaseAdam(params, betas=(0.8, 0.99))

    def append_rig_pose(self, rig_R6D_new, rig_t_new, lr=None):
        """(rig) Incremental grow. Add one ts slot, registering its params with
        rig_optimizer while preserving the moments of already-registered ts."""
        assert self.rig_optimizer is not None, \
            "register_rig_poses must be called before append_rig_pose"
        p_R = torch.nn.Parameter(rig_R6D_new.clone())
        p_t = torch.nn.Parameter(rig_t_new.clone())
        self.rig_R6D.append(p_R)
        self.rig_t.append(p_t)
        new_idx = len(self.rig_R6D) - 1
        if lr is None:
            lr = next(iter(self.rig_optimizer.params.values()))["lr"]
        self.rig_optimizer.add_param(f"rig_R6D_{new_idx}", p_R, lr=lr)
        self.rig_optimizer.add_param(f"rig_t_{new_idx}", p_t, lr=lr)

    def add_keyframe(self, keyframe: Keyframe, f=None):
        """Add a keyframe to the scene, add and prune Gaussians"""

        # Make sure training is not running
        self.join_optimization_thread()

        # (rig) attach scene_model back-ref so get_R/get_t use the shared rig pose
        if getattr(keyframe, "is_rig_mode", False):
            keyframe.scene_model = self

        ## Add the keyframe and update the indices (sorted by distance to last keyframe)
        self.keyframes.append(keyframe)
        if self.approx_cam_centres is None:
            self.approx_cam_centres = keyframe.approx_centre[None]
        else:
            self.approx_cam_centres = torch.cat(
                [self.approx_cam_centres, keyframe.approx_centre[None]], dim=0
            )
        dist_to_last = torch.linalg.vector_norm(
            self.approx_cam_centres - keyframe.approx_centre[None], dim=-1
        )
        self.sorted_frame_indices = torch.argsort(dist_to_last).cpu()

        ## Update intrinsics
        if f is not None:
            self.f = f.item()
            self.init_intrinsics()

        ## Update cached Rts for the viewer
        self.cached_Rts = torch.cat(
            [self.cached_Rts, keyframe.get_Rt().unsqueeze(0)], dim=0
        )
        self.valid_Rt_cache = torch.cat(
            [self.valid_Rt_cache, torch.ones(1, device="cuda", dtype=torch.bool)], dim=0
        )
        gt_pose = keyframe.info.get("Rt", None)
        if gt_pose is not None:
            self.gt_Rts = torch.cat([self.gt_Rts, gt_pose.unsqueeze(0)], dim=0)
        self.gt_Rts_mask = torch.cat(
            [
                self.gt_Rts_mask,
                torch.Tensor([gt_pose is not None]).to(self.gt_Rts_mask),
            ],
            dim=0,
        )
        self.gt_f = keyframe.info.get("focal", self.f)

        if not self.inference_mode:
            ## Add keyframe to the active anchor
            self.active_anchor.add_keyframe(keyframe)
            self.active_frames_gpu.append(keyframe.index)

            ## Clear memory if there are many keyframes. N-view scaling: a rig adds N
            ## keyframes per timestep, so a fixed cap offloads after only cap/N timesteps
            ## (excessive CPU<->GPU churn at high N / long sequences). Scale by N so the
            ## GPU-residency window is measured in timesteps, like n_kept_frames.
            max_active = self.max_active_keyframes * (self.n_rig_views if self.use_rig else 1)
            if len(self.active_frames_gpu) > max_active:
                self.move_rand_keyframe_to_cpu()
                if self.use_rig:
                    return
                # Reshuffle the active keyframes and clear cache
                if len(self.active_frames_cpu) % 5 == 0:
                    self.move_rand_keyframe_to_cpu()
                    self.move_rand_keyframe_to_gpu()

                    gc.collect()
                    torch.cuda.empty_cache()

    def enable_inference_mode(self):
        """Enable inference mode and sets the anchor position to the mean of the active keyframes."""
        self.inference_mode = True
        self.update_anchor()

    def update_anchor(self, n_left_frames: int = 0):
        """Update the active anchor from the live poses that optimized it."""
        anchor_ids = list(self.active_anchor.keyframe_ids)
        if n_left_frames > 0:
            anchor_ids = anchor_ids[:-n_left_frames]
        if len(anchor_ids) == 0:
            return
        anchor_position = self._live_centres_for_keyframe_ids(anchor_ids).mean(dim=0)
        self.active_anchor.position = anchor_position
        if n_left_frames > 0:
            self.active_anchor.keyframes = self.active_anchor.keyframes[:-n_left_frames]
            self.active_anchor.keyframe_ids = self.active_anchor.keyframe_ids[
                :-n_left_frames
            ]

    def place_anchor_if_needed(self):
        """Check if many Gaussians appear small on the screen. If so, place a new anchor. and merge the Gaussians."""
        small_prop_thresh = 0.4
        k = 3
        # N-view-aware recent-context window: keep the last n_kept_timesteps full
        # timesteps (= n_kept_timesteps * N_views keyframes) live, so a 6/9/12/15-
        # view rig keeps the same number of recent TIMESTEPS, not a fixed frame
        # count. Non-rig keeps the original 20.
        self.n_kept_frames = (self.n_kept_timesteps * self.n_rig_views) if self.use_rig else 20
        if (
            self.xyz.shape[0] > 0
            and self.first_active_frame < len(self.keyframes) - 2 * self.n_kept_frames
        ):
            with torch.no_grad():
                latest_centre = self._live_centres_for_keyframe_ids([len(self.keyframes) - 1])[0]
                dist = torch.linalg.vector_norm(
                    self.xyz - latest_centre[None], dim=-1
                )
                screen_size = self.f * self.scaling.mean(dim=-1) / dist
                small_mask = screen_size < 1
                small_prop = small_mask.float().mean()

            if small_prop > small_prop_thresh:
                with torch.no_grad():
                    small_mask = screen_size < 1.5
                    # Update anchor positions using the camera poses used to optimize it
                    self.update_anchor(self.n_kept_frames)

                    ## Merge fine Gaussians for the current active set
                    # Select a subset and get their nearest neighbours for merging
                    small_gaussians = {
                        name: self.gaussian_params[name]["val"][small_mask]
                        for name in self.gaussian_params
                    }
                    xyz = small_gaussians["xyz"].contiguous()
                    _, nn_idx = distIndex2(xyz, k)
                    nn_idx = nn_idx.view(-1, k)
                    perm = torch.randperm(xyz.shape[0], device=xyz.device)
                    idx = perm[: (xyz.shape[0] // (k + 1))]
                    selected_nn_idx = torch.cat([idx[..., None], nn_idx[idx]], dim=-1)

                    # Compute merging weights based on contribution to the rendering
                    weights = self.gaussian_params["opacity"]["val"][
                        selected_nn_idx, 0
                    ].sigmoid() * (screen_size[selected_nn_idx] ** 2)
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                    weights.unsqueeze_(-1)

                    # Merge the Gaussians by averaging their parameters
                    merged_gaussians = {
                        "xyz": (self.gaussian_params["xyz"]['val'][selected_nn_idx, :] * weights).sum(dim=1),
                        "f_dc": (self.gaussian_params["f_dc"]['val'][selected_nn_idx, :] * weights.unsqueeze(-1)).sum(dim=1),
                        "f_rest": (self.gaussian_params["f_rest"]['val'][selected_nn_idx, :] * weights.unsqueeze(-1)).sum(dim=1),
                        "opacity": inverse_sigmoid(self.gaussian_params["opacity"]['val'][selected_nn_idx, :].sigmoid() * weights).sum(dim=1),
                        "scaling": torch.log((torch.exp(self.gaussian_params["scaling"]['val'][selected_nn_idx, :]) * weights * (k+1)).sum(dim=1)),
                        "rotation": (self.gaussian_params["rotation"]['val'][selected_nn_idx, :] * weights).sum(dim=1),
                    }

                    # Offload the previous Gaussians to the CPU
                    self.active_anchor.duplicate_param_dict()
                    self.active_anchor.to("cpu", with_keyframes=True)

                    ## Add the merged Gaussians to the set of Gaussians and reset the optimizer
                    with self.lock:
                        self.optimizer.add_and_prune(merged_gaussians, ~small_mask)

                    # Create a new active anchor with the merged Gaussians
                    new_anchor_centre = self._live_centres_for_keyframe_ids(
                        [kf.index for kf in self.keyframes[-self.n_kept_frames :]]
                    ).mean(dim=0)
                    self.active_anchor = Anchor(
                        self.gaussian_params,
                        new_anchor_centre,
                        self.keyframes[-self.n_kept_frames :],
                    )
                    self.anchors.append(self.active_anchor)
                    self.active_frames_gpu = [kf.index for kf in self.active_anchor.keyframes]
                    self.active_frames_cpu = []

                    # Visualization
                    self.anchor_weights = np.zeros(len(self.anchors))
                    self.anchor_weights[-1] = 1.0

                gc.collect()
                torch.cuda.empty_cache()

    def save(self, path: str, reconstruction_time: float = 0, n_frames: int = 0, n_timesteps: int = 0):
        # Get metrics
        metrics = {
            "num anchors": len(self.anchors),
            "num keyframes": len(self.keyframes),
        }
        if reconstruction_time > 0:
            metrics["time"] = reconstruction_time
            if n_frames > 0:
                if self.use_rig and n_timesteps > 0:
                    metrics["FPS"] = n_timesteps / reconstruction_time
                    metrics["timesteps/sec"] = n_timesteps / reconstruction_time
                    metrics["images/sec"] = n_frames / reconstruction_time
                    metrics["FPS_unit"] = "rig_timesteps_per_second"
                else:
                    metrics["FPS"] = n_frames / reconstruction_time
                    metrics["FPS_unit"] = "frames_per_second"
        metrics.update(self.evaluate(True, True, True))

        if path == "":
            print("No path provided, skipping save")
            return metrics
        os.makedirs(path, exist_ok=True)

        # Save anchors
        pcd_path = os.path.join(path, "point_clouds")
        os.makedirs(pcd_path, exist_ok=True)
        for index, anchor in enumerate(self.anchors):
            anchor.save_ply(os.path.join(pcd_path, f"anchor_{index}.ply"))

        # Save metadata
        metadata = {
            "config": {
                "width": self.width,
                "height": self.height,
                "sh_degree": self.max_sh_degree,
                "f": self.f,
            },
            "anchors": [
                {
                    "position": anchor.position.cpu().numpy().tolist(),
                }
                for anchor in self.anchors
            ],
            "keyframes": [keyframe.to_json() for keyframe in self.keyframes],
        }
        if self.use_rig:
            rig_leakage_audit = dict(self.rig_leakage_audit)
            metadata["rig"] = {
                "num_timesteps": len(self.rig_R6D),
                "num_views": self.n_rig_views,
                "metric_policy": "rig_eval_split == 'test'",
                "completeness": getattr(self, "rig_completeness", {}),
                "split_counts": {
                    split: sum(
                        1 for keyframe in self.keyframes
                        if keyframe.info.get("rig_eval_split") == split
                    )
                    for split in ("train", "test", "tracking")
                },
                "leakage_audit": rig_leakage_audit,
            }
            self.extra_metadata["rig_leakage_audit"] = rig_leakage_audit
            if getattr(self, "rig_completeness", None):
                self.extra_metadata["rig_completeness"] = self.rig_completeness
        if self.extra_metadata:
            metadata["extra"] = self.extra_metadata
        metadata = {**metrics, **metadata}

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # Save renders of test views
        self.save_test_frames(os.path.join(path, "test_images"))

        # Saving cameras with COLMAP format
        images = {}
        cameras = {}
        colmap_save_path = os.path.join(path, "colmap")
        os.makedirs(colmap_save_path, exist_ok=True)
        for index, keyframe in enumerate(self.keyframes):
            camera, image = keyframe.to_colmap(index)
            cameras[index] = camera
            images[index] = image
        write_model(cameras, images, {}, colmap_save_path, ext=".bin")

        return metrics

    def get_closest_keyframe(
        self, position: torch.Tensor, count: int = 1
    ) -> list[Keyframe]:
        dists = torch.linalg.vector_norm(
            self.approx_cam_centres - position[None], dim=-1
        )
        closest_ids = dists.argsort()[:count]
        return [self.keyframes[closest_id] for closest_id in closest_ids]

    def finetune_epoch(self):
        """
        Go through all anchors and optimize them one by one.
        This is used for finetuning after the initial training.
        """
        self.anchor_weights = np.zeros(len(self.anchors))
        for anchor_id, anchor in enumerate(self.anchors):
            self.active_anchor = anchor
            # Load the anchor and make its parameters optimizable
            anchor.to("cuda", with_keyframes=True)
            self.gaussian_params = anchor.gaussian_params
            self.anchor_weights[anchor_id] = 1
            self.reset_optimizer()

            # Optimize the anchor by going through its keyframes
            for _ in range(len(anchor.keyframes)):
                self.optimization_step(finetuning=True)

            # Update the anchor and store it on cpu
            anchor.gaussian_params = self.gaussian_params
            self.anchor_weights[anchor_id] = 0
