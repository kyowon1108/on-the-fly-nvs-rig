#
# Copyright (C) 2025
# EQR to pinhole conversion example for the virtual panoramic rig.
#
# This file is used by both offline extraction and single-image append mode.
# Main responsibilities:
# - Decode EQR input (video or single image)
# - Project to rig-defined pinhole views
# - Save images in COLMAP-friendly layout: images/<view>/frame_xxxxxx.ext

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class EQRToPinholeConverter:
    """
    Converts equirectangular (EQR) images to multiple pinhole camera images.
    Uses precomputed sampling grids for efficient conversion.
    """

    def __init__(
        self,
        rig_config_path: str,
        output_width: int = 960,
        output_height: int = 960,
        fov_deg: float = 90.0,
        phi_offset_deg: float = -90.0,
        device: str = "cuda",
    ):
        self.output_width = output_width
        self.output_height = output_height
        self.fov_deg = fov_deg
        self.phi_offset = math.radians(phi_offset_deg)
        self.device = device

        self.cameras = self._load_rig_config(rig_config_path)
        # Precompute per-view sampling grids once for fast per-frame conversion.
        self.sampling_grids = self._precompute_grids()
        self.camera_names = [cam["name"] for cam in self.cameras]
        self.camera_name_to_index = {
            name: idx for idx, name in enumerate(self.camera_names)
        }

    def _load_rig_config(self, config_path: str) -> List[dict]:
        # blender_rig.json is ring-based; flatten it into a simple camera list.
        with open(config_path, "r", encoding="utf-8") as f:
            rig_config = json.load(f)

        cameras = []
        for ring in rig_config:
            for cam in ring["cameras"]:
                cameras.append(
                    {
                        "name": cam["name"],
                        "location": torch.tensor(
                            cam["location"], device=self.device, dtype=torch.float32
                        ),
                        "rotation": torch.tensor(
                            cam["rotation"], device=self.device, dtype=torch.float32
                        ),  # quaternion (w, x, y, z)
                        "ring": ring["name"],
                        "tilt": ring["tilt"],
                    }
                )
        return cameras

    def _quaternion_to_rotation_matrix(self, q: Tensor) -> Tensor:
        # Quaternion order is (w, x, y, z).
        w, x, y, z = q[0], q[1], q[2], q[3]

        norm = torch.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        R = torch.zeros(3, 3, device=self.device, dtype=torch.float32)
        R[0, 0] = 1 - 2 * y * y - 2 * z * z
        R[0, 1] = 2 * x * y - 2 * z * w
        R[0, 2] = 2 * x * z + 2 * y * w
        R[1, 0] = 2 * x * y + 2 * z * w
        R[1, 1] = 1 - 2 * x * x - 2 * z * z
        R[1, 2] = 2 * y * z - 2 * x * w
        R[2, 0] = 2 * x * z - 2 * y * w
        R[2, 1] = 2 * y * z + 2 * x * w
        R[2, 2] = 1 - 2 * x * x - 2 * y * y
        return R

    def _blender_quat_to_world2cam(self, q: Tensor) -> Tensor:
        """
        Convert Blender camera quaternion to COLMAP-style world-to-camera rotation.
        """
        R_c2w = self._quaternion_to_rotation_matrix(q)
        axis_flip = torch.diag(
            torch.tensor([1.0, -1.0, -1.0], device=self.device, dtype=torch.float32)
        )
        return axis_flip @ R_c2w.T

    def _compute_sampling_grid(self, rotation: Tensor) -> Tensor:
        # Build grid_sample UV map that sends pinhole rays into EQR coordinates.
        rays_world = self._compute_world_rays(rotation)
        x_w, y_w, z_w = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]

        phi = torch.atan2(x_w, y_w)
        theta = torch.asin(torch.clamp(z_w, -1, 1))

        phi = phi + self.phi_offset
        phi = torch.remainder(phi + math.pi, 2 * math.pi) - math.pi

        u_eqr = phi / math.pi
        v_eqr = -theta / (math.pi / 2)
        return torch.stack([u_eqr, v_eqr], dim=-1).unsqueeze(0)

    def _compute_world_rays(self, rotation: Tensor) -> Tensor:
        H, W = self.output_height, self.output_width
        fov_rad = self.fov_deg * math.pi / 180.0
        focal = W / (2 * math.tan(fov_rad / 2))

        u = torch.linspace(0, W - 1, W, device=self.device)
        v = torch.linspace(0, H - 1, H, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing="xy")

        cx, cy = W / 2, H / 2
        x_cam = (uu - cx) / focal
        y_cam = (vv - cy) / focal
        z_cam = torch.ones_like(x_cam)

        # OpenCV -> Blender camera local conversion.
        rays_blender_local = torch.stack(
            [
                x_cam,
                -y_cam,
                -z_cam,
            ],
            dim=-1,
        )
        rays_blender_local = rays_blender_local / rays_blender_local.norm(
            dim=-1, keepdim=True
        )

        R = self._quaternion_to_rotation_matrix(rotation)
        return torch.einsum("ij,hwj->hwi", R, rays_blender_local)

    def _compute_forward_direction(self, rotation: Tensor) -> Tensor:
        R = self._quaternion_to_rotation_matrix(rotation)
        forward = R @ torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=torch.float32
        )
        return forward / forward.norm()

    def _precompute_grids(self) -> List[Tensor]:
        # One static grid per camera because rig rotations are fixed.
        grids = []
        for cam in self.cameras:
            grids.append(self._compute_sampling_grid(cam["rotation"]))
        return grids

    @torch.no_grad()
    def convert(
        self, eqr_frame: Tensor, camera_indices: Optional[List[int]] = None
    ) -> List[Tensor]:
        # Single-frame conversion used in the extraction loop.
        if eqr_frame.dim() == 3:
            eqr_frame = eqr_frame.unsqueeze(0)

        if eqr_frame.device != torch.device(self.device):
            eqr_frame = eqr_frame.to(self.device)

        if camera_indices is None:
            camera_indices = list(range(len(self.cameras)))

        pinhole_images = []
        for idx in camera_indices:
            pinhole = F.grid_sample(
                eqr_frame,
                self.sampling_grids[idx],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            pinhole_images.append(pinhole.squeeze(0))
        return pinhole_images

    @torch.no_grad()
    def convert_batch(
        self, eqr_frames: Tensor, camera_indices: Optional[List[int]] = None
    ) -> Tensor:
        # Batch conversion path retained for high-throughput/experiments.
        if eqr_frames.dim() == 3:
            eqr_frames = eqr_frames.unsqueeze(0)
        if eqr_frames.device != torch.device(self.device):
            eqr_frames = eqr_frames.to(self.device)

        B, C = eqr_frames.shape[0], eqr_frames.shape[1]
        if camera_indices is None:
            camera_indices = list(range(len(self.cameras)))

        num_cams = len(camera_indices)
        grids = torch.cat([self.sampling_grids[i] for i in camera_indices], dim=0)
        eqr_expanded = eqr_frames.unsqueeze(1).expand(-1, num_cams, -1, -1, -1)
        eqr_expanded = eqr_expanded.reshape(B * num_cams, C, eqr_frames.shape[2], eqr_frames.shape[3])

        grids_expanded = grids.unsqueeze(0).expand(B, -1, -1, -1, -1)
        grids_expanded = grids_expanded.reshape(
            B * num_cams, self.output_height, self.output_width, 2
        )

        pinhole_flat = F.grid_sample(
            eqr_expanded,
            grids_expanded,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return pinhole_flat.reshape(B, num_cams, C, self.output_height, self.output_width)

    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        fov_rad = self.fov_deg * math.pi / 180.0
        focal = self.output_width / (2 * math.tan(fov_rad / 2))
        cx = self.output_width / 2
        cy = self.output_height / 2
        return focal, focal, cx, cy

    def get_relative_poses(self, ref_view_name: str) -> Tensor:
        if ref_view_name not in self.camera_name_to_index:
            raise ValueError(
                f"Unknown ref_view_name={ref_view_name}. "
                f"Available views: {self.camera_names}"
            )
        num_cameras = len(self.cameras)
        ref_idx = self.camera_name_to_index[ref_view_name]
        ref_R_world2cam = self._blender_quat_to_world2cam(
            self.cameras[ref_idx]["rotation"]
        )

        relative_Rts = []
        for cam_idx in range(num_cameras):
            cam = self.cameras[cam_idx]
            R_world2cam = self._blender_quat_to_world2cam(cam["rotation"])
            rel_R = R_world2cam @ ref_R_world2cam.T
            Rt = torch.eye(4, device=self.device, dtype=torch.float32)
            Rt[:3, :3] = rel_R
            Rt[:3, 3] = 0.0
            relative_Rts.append(Rt)

        return torch.stack(relative_Rts, dim=0)

    def build_feature_masks(self, camera_indices: Sequence[int]) -> Dict[str, np.ndarray]:
        selected_forwards = []
        for idx in camera_indices:
            selected_forwards.append(
                self._compute_forward_direction(self.cameras[idx]["rotation"])
            )
        selected_forwards = torch.stack(selected_forwards, dim=0)

        masks: Dict[str, np.ndarray] = {}
        for local_idx, cam_idx in enumerate(camera_indices):
            rays_world = self._compute_world_rays(self.cameras[cam_idx]["rotation"])
            closest_idx = torch.argmax(
                torch.einsum("hwc,nc->hwn", rays_world, selected_forwards), dim=-1
            )
            mask = (closest_idx == local_idx).to(torch.uint8).mul(255).cpu().numpy()
            masks[self.camera_names[cam_idx]] = mask
        return masks

    def __len__(self) -> int:
        return len(self.cameras)

    def __repr__(self) -> str:
        return (
            "EQRToPinholeConverter("
            f"num_cameras={len(self.cameras)}, "
            f"output_size=({self.output_width}, {self.output_height}), "
            f"fov={self.fov_deg}deg)"
        )


def _resolve_views(
    rig_config_path: Path, views_arg: str
) -> List[str]:
    # Validate requested view subset against blender_rig.json.
    with open(rig_config_path, "r", encoding="utf-8") as f:
        rig = json.load(f)

    all_views = []
    for ring in rig:
        for cam in ring["cameras"]:
            all_views.append(cam["name"])

    if not views_arg:
        return all_views

    views = [item.strip() for item in views_arg.split(",") if item.strip()]
    missing = [view for view in views if view not in all_views]
    if missing:
        raise ValueError(
            f"Unknown views in --views: {missing}. Available views: {all_views}"
        )
    return views


def _build_parser() -> argparse.ArgumentParser:
    # CLI is shared by offline video extraction and single-image append mode.
    parser = argparse.ArgumentParser(
        description="Extract multi-view pinhole images from EQR video."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Input EQR video path. Set exactly one of --video / --eqr_image.",
    )
    parser.add_argument(
        "--eqr_image",
        type=str,
        default="",
        help="Input single EQR image path. Set exactly one of --video / --eqr_image.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="datasets/saebit",
        help="Output dataset root. Images are written to <out_dir>/images/<view>/frame_xxxxxx.png",
    )
    parser.add_argument(
        "--rig_config",
        type=str,
        default="blender_rig.json",
        help="Path to blender_rig.json",
    )
    parser.add_argument("--stride", type=int, default=20, help="Frame sampling stride.")
    parser.add_argument(
        "--views",
        type=str,
        default="",
        help="Comma-separated views. Default: all cameras from rig_config.",
    )
    parser.add_argument(
        "--ref_view",
        type=str,
        default="High_Cam07",
        help="Reference view name for downstream rig processing.",
    )
    parser.add_argument("--fov", type=float, default=90.0, help="Pinhole FoV in degrees.")
    parser.add_argument(
        "--size",
        type=int,
        default=960,
        choices=[960, 1920],
        help="Output size (square): 960 or 1920.",
    )
    parser.add_argument(
        "--also_make_960",
        action="store_true",
        help="If set with --size 1920, also writes a downscaled 960 dataset.",
    )
    parser.add_argument(
        "--also_make_960_out_dir",
        type=str,
        default="",
        help="Optional output root for downscaled dataset. Default: <out_dir>_960",
    )
    parser.add_argument(
        "--frame_limit",
        type=int,
        default=-1,
        help="Optional cap on sampled frames (for quick tests).",
    )
    parser.add_argument(
        "--append_mode",
        action="store_true",
        help="Append new frames after existing frame_xxxxxx files in out_dir/images/<view>.",
    )
    parser.add_argument(
        "--phi_offset_deg",
        type=float,
        default=-90.0,
        help="Longitude offset used in EQR projection mapping.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Computation device.",
    )
    parser.add_argument(
        "--write_masks",
        action="store_true",
        help="Write static per-view masks to <out_dir>/masks/<view>/frame_xxxxxx.png.",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default="frame_manifest.jsonl",
        help="Per-timestamp manifest filename written under out_dir.",
    )
    return parser


def _save_tensor_as_png(image_tensor: Tensor, out_path: Path) -> np.ndarray:
    return _save_tensor_image(
        image_tensor=image_tensor,
        out_path=out_path,
        image_format="png",
    )


def _save_tensor_image(
    image_tensor: Tensor,
    out_path: Path,
    image_format: str = "png",
    jpeg_quality: int = 92,
    webp_quality: int = 90,
) -> np.ndarray:
    # Convert torch RGB [0,1] to OpenCV BGR uint8 and encode with requested format.
    fmt = image_format.strip().lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in {"png", "jpg", "webp"}:
        raise ValueError(f"Unsupported image_format: {image_format}")

    rgb = image_tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    params: List[int] = []
    if fmt == "jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif fmt == "webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(webp_quality)]

    ok = cv2.imwrite(str(out_path), bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")
    return bgr


def _save_mask(mask: np.ndarray, out_path: Path) -> None:
    ok = cv2.imwrite(str(out_path), mask)
    if not ok:
        raise RuntimeError(f"Failed to write mask: {out_path}")


def _compute_next_frame_index(out_images_dir: Path, views: Sequence[str]) -> int:
    # In append mode, continue numbering after the latest existing frame index.
    if not views:
        return 0
    ref_dir = out_images_dir / views[0]
    if not ref_dir.exists():
        return 0
    pattern = re.compile(r"^frame_(\d{6})\.(png|jpg|jpeg|webp)$", re.IGNORECASE)
    max_idx = -1
    for path in ref_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _convert_and_save_single_eqr(
    frame_rgb: np.ndarray,
    converter: EQRToPinholeConverter,
    selected_indices: Sequence[int],
    views: Sequence[str],
    out_images_dir: Path,
    downscale_out_dir: Optional[Path],
    frame_number: int,
) -> str:
    # Core per-frame routine: EQR RGB -> selected views -> disk (and optional 960 downscale).
    eqr_tensor = (
        torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous().float() / 255.0
    )
    pinholes = converter.convert(eqr_tensor, list(selected_indices))
    save_name = f"frame_{frame_number:06d}.png"
    for view_name, pinhole in zip(views, pinholes):
        out_path = out_images_dir / view_name / save_name
        image_bgr = _save_tensor_as_png(pinhole, out_path)
        if downscale_out_dir is not None:
            small = cv2.resize(
                image_bgr,
                (960, 960),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imwrite(
                str(downscale_out_dir / "images" / view_name / save_name), small
            )
    return save_name


def _write_masks_for_frame(
    masks: Dict[str, np.ndarray],
    views: Sequence[str],
    out_dir: Path,
    save_name: str,
    downscale_out_dir: Optional[Path],
) -> None:
    for view_name in views:
        mask = masks[view_name]
        _save_mask(mask, out_dir / "masks" / view_name / save_name)
        if downscale_out_dir is not None:
            small = cv2.resize(mask, (960, 960), interpolation=cv2.INTER_NEAREST)
            _save_mask(small, downscale_out_dir / "masks" / view_name / save_name)


def extract_dataset(args: argparse.Namespace) -> None:
    # Orchestrates one extraction run and writes extraction_meta.json for reproducibility.
    rig_config_path = Path(args.rig_config)
    out_dir = Path(args.out_dir)
    out_images_dir = out_dir / "images"

    views = _resolve_views(rig_config_path, args.views)
    if args.ref_view not in views:
        raise ValueError(
            f"--ref_view {args.ref_view} is not in selected --views: {views}"
        )
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")
    if args.frame_limit == 0:
        raise ValueError("--frame_limit must be -1 (no limit) or >= 1")
    if args.also_make_960 and args.size != 1920:
        raise ValueError("--also_make_960 requires --size 1920")
    has_video = bool(args.video)
    has_eqr_image = bool(args.eqr_image)
    if has_video == has_eqr_image:
        raise ValueError("Set exactly one input mode: --video or --eqr_image")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"

    converter = EQRToPinholeConverter(
        rig_config_path=str(rig_config_path),
        output_width=args.size,
        output_height=args.size,
        fov_deg=args.fov,
        phi_offset_deg=args.phi_offset_deg,
        device=device,
    )

    selected_indices = [converter.camera_name_to_index[name] for name in views]
    static_masks = (
        converter.build_feature_masks(selected_indices) if args.write_masks else None
    )

    for view in views:
        (out_images_dir / view).mkdir(parents=True, exist_ok=True)
        if static_masks is not None:
            (out_dir / "masks" / view).mkdir(parents=True, exist_ok=True)

    downscale_out_dir: Optional[Path] = None
    if args.also_make_960:
        downscale_out_dir = (
            Path(args.also_make_960_out_dir)
            if args.also_make_960_out_dir
            else out_dir.parent / f"{out_dir.name}_960"
        )
        for view in views:
            (downscale_out_dir / "images" / view).mkdir(parents=True, exist_ok=True)
            if static_masks is not None:
                (downscale_out_dir / "masks" / view).mkdir(
                    parents=True, exist_ok=True
                )

    next_frame_index = (
        _compute_next_frame_index(out_images_dir, views) if args.append_mode else 0
    )
    sampled_count = 0
    frame_idx = 0
    fps = 0.0
    manifest_lines: List[dict] = []
    if has_video:
        # Offline mode: sample every `stride` frames from video stream.
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(
            "Video opened: %s (frames=%s, fps=%.3f)",
            args.video,
            total_video_frames if total_video_frames > 0 else "unknown",
            fps if fps > 0 else -1.0,
        )

        stop = False
        while not stop:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % args.stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                save_name = _convert_and_save_single_eqr(
                    frame_rgb=frame_rgb,
                    converter=converter,
                    selected_indices=selected_indices,
                    views=views,
                    out_images_dir=out_images_dir,
                    downscale_out_dir=downscale_out_dir,
                    frame_number=next_frame_index,
                )
                if static_masks is not None:
                    _write_masks_for_frame(
                        masks=static_masks,
                        views=views,
                        out_dir=out_dir,
                        save_name=save_name,
                        downscale_out_dir=downscale_out_dir,
                    )
                manifest_lines.append(
                    {
                        "frame_index": next_frame_index,
                        "source_video_frame": frame_idx,
                        "source_time_sec": (frame_idx / fps) if fps > 0 else None,
                        "ref_view": args.ref_view,
                        "views": {
                            view_name: f"images/{view_name}/{save_name}"
                            for view_name in views
                        },
                        "masks": (
                            {
                                view_name: f"masks/{view_name}/{save_name}"
                                for view_name in views
                            }
                            if static_masks is not None
                            else {}
                        ),
                    }
                )
                next_frame_index += 1
                sampled_count += 1
                if sampled_count % 10 == 0:
                    logging.info(
                        "Extracted %d frames (%d views each, size=%dx%d)",
                        sampled_count,
                        len(views),
                        args.size,
                        args.size,
                    )
                if args.frame_limit > 0 and sampled_count >= args.frame_limit:
                    stop = True

            frame_idx += 1

        cap.release()
    else:
        # Single-image mode: consume one EQR image and append one frame.
        image_path = Path(args.eqr_image)
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Cannot read EQR image: {image_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        save_name = _convert_and_save_single_eqr(
            frame_rgb=frame_rgb,
            converter=converter,
            selected_indices=selected_indices,
            views=views,
            out_images_dir=out_images_dir,
            downscale_out_dir=downscale_out_dir,
            frame_number=next_frame_index,
        )
        if static_masks is not None:
            _write_masks_for_frame(
                masks=static_masks,
                views=views,
                out_dir=out_dir,
                save_name=save_name,
                downscale_out_dir=downscale_out_dir,
            )
        manifest_lines.append(
            {
                "frame_index": next_frame_index,
                "source_video_frame": None,
                "source_time_sec": None,
                "ref_view": args.ref_view,
                "views": {
                    view_name: f"images/{view_name}/{save_name}" for view_name in views
                },
                "masks": (
                    {
                        view_name: f"masks/{view_name}/{save_name}"
                        for view_name in views
                    }
                    if static_masks is not None
                    else {}
                ),
            }
        )
        sampled_count = 1

    fx, fy, cx, cy = converter.get_intrinsics()
    # Persist run metadata so COLMAP/training runs are traceable.
    meta = {
        "input_mode": "video" if has_video else "eqr_image",
        "video": str(Path(args.video).resolve()) if has_video else "",
        "eqr_image": str(Path(args.eqr_image).resolve()) if has_eqr_image else "",
        "rig_config": str(rig_config_path.resolve()),
        "stride": args.stride,
        "sampled_frames": sampled_count,
        "processed_video_frames": frame_idx,
        "append_mode": bool(args.append_mode),
        "views": views,
        "ref_view": args.ref_view,
        "fov_deg": args.fov,
        "size": args.size,
        "device": device,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "also_make_960": bool(args.also_make_960),
        "also_make_960_out_dir": str(downscale_out_dir) if downscale_out_dir else "",
        "write_masks": bool(args.write_masks),
        "manifest_name": args.manifest_name,
    }
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "extraction_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / args.manifest_name, "w", encoding="utf-8") as f:
        for item in manifest_lines:
            f.write(json.dumps(item) + "\n")
    if downscale_out_dir is not None:
        with open(
            downscale_out_dir / "extraction_meta.json", "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=2)
        with open(downscale_out_dir / args.manifest_name, "w", encoding="utf-8") as f:
            for item in manifest_lines:
                f.write(json.dumps(item) + "\n")

    logging.info("Extraction complete.")
    logging.info(
        "Summary: total_extracted_frames=%d, views=%d, resolution=%dx%d",
        sampled_count,
        len(views),
        args.size,
        args.size,
    )
    logging.info("Output: %s", out_dir / "images")
    if downscale_out_dir is not None:
        logging.info("Additional downscaled output: %s", downscale_out_dir / "images")


def main() -> None:
    # Thin CLI wrapper.
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    parser = _build_parser()
    args = parser.parse_args()
    extract_dataset(args)


if __name__ == "__main__":
    main()
