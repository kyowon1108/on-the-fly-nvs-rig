"""Rig-aware image dataset for an N-view zero-baseline virtual rig.

Each timestep emits one interleaved N-view batch: the reference view first, then
the remaining non-ref views in the order they appear in `rig_config.view_names`.
The training loop pulls a whole timestep at once and estimates a single shared
rig pose for it; each frame carries its fixed `rig_relative_Rt` in `info`, and
per-view poses are derived as rel @ rig (no independent per-view pose, rel_t=0).
`start_at` is in timestep units (not per-image).
"""

import json
import logging
import os
import re
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, List

import cv2
import torch

from rig.rig_loader import load_rig_config
from utils import get_image_names


def _frame_key(view: str, ts: int) -> str:
    # Unique key used both as `info["name"]` and as the dict key in `self.infos`.
    return f"{view}__ts{ts:05d}"


def _parse_timestep_token(token: str) -> int:
    """Parse an OB3D split token into the original integer frame index.

    OB3D split files use bare integers (e.g. `2`), while local tooling often
    names prepared pinholes as `frame_000002.png`. Accept both so the split
    manifest can stay close to the source dataset or to the prepared rig tree.
    """
    token = token.strip()
    if not token:
        raise ValueError("empty timestep token")
    if token.isdigit():
        return int(token)
    base = os.path.basename(token)
    match = re.search(r"(\d+)", base)
    if match is None:
        raise ValueError(f"Cannot parse timestep index from split token: {token!r}")
    return int(match.group(1))


def _load_timestep_split(path: str, valid_timesteps: set[int]) -> set[int]:
    test_timesteps: set[int] = set()
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            ts = _parse_timestep_token(line)
            if ts not in valid_timesteps:
                raise ValueError(
                    f"{path}:{line_no}: timestep {ts} is not present in the "
                    f"prepared rig scene. Valid range/sample: "
                    f"{min(valid_timesteps)}..{max(valid_timesteps)}"
                )
            test_timesteps.add(ts)
    if not test_timesteps:
        raise ValueError(f"No test timesteps found in {path}")
    return test_timesteps


def _build_timestep_records(names, start_at: int = 0) -> list[dict]:
    """Return source timestep semantics for a sorted online stream.

    `source_ts` is the original EQR frame id parsed from the filename and is
    used for splits, ATE, and reporting. `stream_idx` is only the rank in the
    online stream after `start_at` trimming; it is never a pose-optimizer slot.
    """
    timestep_names = sorted(names)
    if not timestep_names:
        raise RuntimeError("No timesteps are shared by all rig views.")
    if start_at < 0:
        raise ValueError(f"--start_at must be non-negative, got {start_at}")
    if start_at >= len(timestep_names):
        raise ValueError(
            f"--start_at ({start_at}) >= num_timesteps "
            f"({len(timestep_names)}); nothing to iterate."
        )

    records = []
    seen_source_ts = {}
    for stream_idx, fname in enumerate(timestep_names[start_at:]):
        source_ts = _parse_timestep_token(fname)
        if source_ts in seen_source_ts:
            raise ValueError(
                "Prepared rig scene has duplicate source timestep "
                f"{source_ts}: {seen_source_ts[source_ts]!r} and {fname!r}"
            )
        seen_source_ts[source_ts] = fname
        records.append({
            "filename": fname,
            "source_ts": source_ts,
            "stream_idx": stream_idx,
        })
    return records


class RigImageDataset:
    """Two-pass iterator over an N-view rig directory."""

    def __init__(self, args: Namespace):
        self.source_path = args.source_path
        self.images_root = os.path.join(args.source_path, args.images_dir)

        # Optional per-view mask tree (mirrors images/<view>/<frame>). Only
        # activated if --masks_dir is set AND the directory actually exists, so
        # current rig runs (no masks/) are unaffected. Masks are applied to the
        # loss/eval/spawn exactly like the single-camera ImageDataset path.
        # Fail-closed: if the user asked for masks, a missing dir is an error, not
        # a silent mask-OFF (which would leave them thinking masks are active).
        masks_dir = getattr(args, "masks_dir", "") or ""
        self.masks_root = ""
        if masks_dir:
            masks_root = os.path.join(args.source_path, masks_dir)
            if not os.path.isdir(masks_root):
                raise FileNotFoundError(
                    f"--masks_dir was set ({masks_dir!r}) but not found: {masks_root}"
                )
            self.masks_root = masks_root

        # Auto-load focal from the extraction step's metadata so a forgotten
        # --init_fov can't silently produce a wrong reconstruction. The
        # extraction script (eqr_to_pinhole.py) is the source of truth.
        meta_path = os.path.join(args.source_path, "extraction_meta.json")
        if os.path.exists(meta_path) and args.init_focal < 0 and args.init_fov < 0:
            with open(meta_path) as f:
                meta = json.load(f)
            args.init_focal = float(meta["intrinsics"]["fx"])
            print(
                f"[rig] auto-loaded init_focal={args.init_focal:.1f} "
                f"from {meta_path}"
            )
        elif not os.path.exists(meta_path) and args.init_focal < 0 and args.init_fov < 0:
            # Without extraction_meta.json the focal can only fall back to
            # 0.7*width, which for the OB3D-rig pinhole crop is wildly wrong and
            # silently corrupts the reconstruction. Force the user to pass the
            # known intrinsic explicitly rather than guessing.
            raise FileNotFoundError(
                f"[rig] no extraction_meta.json at {meta_path} and neither "
                "--init_focal nor --init_fov was given. Pass the pinhole focal "
                "explicitly (e.g. --init_focal <fx_pixels> or --init_fov <deg>); "
                "the 0.7*width fallback is not valid for the rig crop."
            )

        # Load rig geometry (relative Rt per view, in COLMAP world-to-camera convention).
        self.rig = load_rig_config(args.rig_config, ref_view=args.ref_view, device="cpu")
        self.ref_view = self.rig.ref_view
        self.non_ref_views = [v for v in self.rig.view_names if v != self.ref_view]

        # Scan per-view frame files and derive the common timestep list.
        self.frames_per_view: Dict[str, List[str]] = {}
        for view in self.rig.view_names:
            view_dir = os.path.join(self.images_root, view)
            if not os.path.isdir(view_dir):
                raise FileNotFoundError(f"Rig view directory missing: {view_dir}")
            names = sorted(get_image_names(view_dir))
            self.frames_per_view[view] = names
        common = set(self.frames_per_view[self.ref_view])
        for view in self.rig.view_names:
            common &= set(self.frames_per_view[view])
        self.timestep_records = _build_timestep_records(common, int(args.start_at))
        self.timestep_names = [record["filename"] for record in self.timestep_records]
        self.source_ts_by_name = {
            record["filename"]: record["source_ts"]
            for record in self.timestep_records
        }
        self.stream_index = {
            record["filename"]: record["stream_idx"]
            for record in self.timestep_records
        }
        # Backward-compatible name for code that only needs online ordering.
        self.ts_index = self.stream_index
        self.num_timesteps = len(self.timestep_names)

        # Iteration order: every timestep emits a full N-view batch, ref first,
        # non-ref in fixed rig order. train.py consumes the whole batch together.
        self.items: List[dict] = []
        for record in self.timestep_records:
            fname = record["filename"]
            source_ts = record["source_ts"]
            stream_idx = record["stream_idx"]
            self.items.append({"view": self.ref_view, "ts": source_ts,
                               "source_ts": source_ts, "stream_idx": stream_idx,
                               "path": os.path.join(self.images_root, self.ref_view, fname),
                               "filename": fname})
            for view in self.non_ref_views:
                self.items.append({"view": view, "ts": source_ts,
                                   "source_ts": source_ts, "stream_idx": stream_idx,
                                   "path": os.path.join(self.images_root, view, fname),
                                   "filename": fname})

        # Per-item metadata consumed by the training loop.
        # is_test marks held-out frames so add_new_gaussians and the scene loss
        # skip them, while the online stream still estimates their poses. Two
        # holdout modes exist:
        #   1) rig_holdout_view: one direction across all timesteps (diagnostic)
        #   2) rig_test_timesteps_file: all N views for held-out EQR timesteps
        #   3) rig_train_timesteps_file + rig_test_timesteps_file: OB3D-style
        #      train/test split; remaining timesteps are tracking-only.
        holdout_view = (getattr(args, "rig_holdout_view", "") or "").strip()
        if holdout_view and holdout_view not in self.rig.view_names:
            raise ValueError(
                f"--rig_holdout_view={holdout_view!r} is not in the rig views "
                f"{self.rig.view_names}"
            )
        if holdout_view and holdout_view == self.ref_view:
            raise ValueError(
                f"--rig_holdout_view cannot be the ref_view ({self.ref_view}); "
                "ref view is needed for bootstrap and incremental pose tracking."
            )
        self.holdout_view = holdout_view
        test_split_file = (getattr(args, "rig_test_timesteps_file", "") or "").strip()
        train_split_file = (getattr(args, "rig_train_timesteps_file", "") or "").strip()
        self.test_timesteps: set[int] = set()
        self.train_timesteps: set[int] = set()
        valid_timesteps = {item["source_ts"] for item in self.items}
        if test_split_file:
            if holdout_view:
                raise ValueError(
                    "--rig_holdout_view and --rig_test_timesteps_file are mutually exclusive"
                )
            if not os.path.exists(test_split_file):
                raise FileNotFoundError(
                    f"--rig_test_timesteps_file not found: {test_split_file}"
                )
            self.test_timesteps = _load_timestep_split(test_split_file, valid_timesteps)
        if train_split_file:
            if not test_split_file:
                raise ValueError(
                    "--rig_train_timesteps_file requires --rig_test_timesteps_file"
                )
            if not os.path.exists(train_split_file):
                raise FileNotFoundError(
                    f"--rig_train_timesteps_file not found: {train_split_file}"
                )
            self.train_timesteps = _load_timestep_split(train_split_file, valid_timesteps)
            overlap = sorted(self.train_timesteps & self.test_timesteps)
            if overlap:
                raise ValueError(
                    "Train/test timestep splits overlap; first overlaps: "
                    f"{overlap[:10]}"
                )
        self.holdout_mode = (
            "ob3d"
            if self.train_timesteps and self.test_timesteps
            else ("timestep" if self.test_timesteps else ("view" if holdout_view else "none"))
        )
        self.infos: Dict[str, dict] = {}
        for item in self.items:
            key = _frame_key(item["view"], item["source_ts"])
            if item["source_ts"] in self.test_timesteps:
                eval_split = "test"
            elif self.train_timesteps:
                eval_split = (
                    "train" if item["source_ts"] in self.train_timesteps
                    else "tracking"
                )
            elif holdout_view and item["view"] == holdout_view:
                eval_split = "test"
            else:
                eval_split = "train"
            self.infos[key] = {
                # Historical bool consumed by SceneModel as an optimization gate.
                # In OB3D train/test mode, tracking-only frames are also excluded
                # from Gaussian/loss updates, so they are True here but are NOT
                # counted as test metrics (`rig_eval_split == "tracking"`).
                "is_test": (eval_split != "train"),
                "name": key,
                "rig_view": item["view"],
                "rig_ts": item["source_ts"],
                "source_ts": item["source_ts"],
                "stream_idx": item["stream_idx"],
                "rig_eval_split": eval_split,
                "rig_filename": item["filename"],
                "rig_relative_Rt": self.rig.relative_Rt[item["view"]].clone(),
            }

        # Loading config
        self.downsampling = args.downsampling
        self.num_threads = min(args.num_loader_threads, max(len(self.items), 1))
        self.current_index = 0
        self.preload_queue: Queue = Queue(maxsize=self.num_threads)
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)

        # Probe the first image for resolution (and trigger auto-downsampling if huge).
        first_image = self._load_image(self.items[0]["path"])
        self.width, self.height = first_image.shape[2], first_image.shape[1]
        res = self.width * self.height
        max_res = 1_500_000
        if self.downsampling <= 0.0 and res > max_res:
            logging.warning(
                "Large rig images, downsampling to 1.5 Mpx. "
                "Pass --downsampling=1 to disable."
            )
            self.downsampling = (res / max_res) ** 0.5
            first_image = self._load_image(self.items[0]["path"])
            self.width, self.height = first_image.shape[2], first_image.shape[1]

        self.start_preloading()

    # ---- dataset API matching ImageDataset ----

    def __len__(self) -> int:
        return len(self.items)

    def get_image_size(self):
        return self.height, self.width

    @torch.no_grad()
    def __getitem__(self, index: int):
        item = self.items[index]
        image = self._load_image(item["path"], cv2.IMREAD_UNCHANGED)
        key = _frame_key(item["view"], item["source_ts"])
        # Return a shallow copy so per-frame mutations (mask, ts_idx set later by
        # train.py) don't accumulate on the shared self.infos[key] dict.
        info = dict(self.infos[key])
        if image.shape[0] == 4:
            info["mask"] = image[-1][None].cpu()
            image = image[:3]
        if self.masks_root:
            info["mask"] = self._load_mask(item["view"], item["filename"])
        return image.cuda(), info

    def _load_mask(self, view: str, filename: str):
        """Load the per-view mask matching images/<view>/<filename>, downsampled
        the same way as the image. Returns a [1, H, W] float tensor in {0,1} on
        CPU. Raises if no mask file is found (fail-closed: masks were requested).
        Tries a few naming conventions (the OB3D-rig export uses
        masks/<view>/<filename>.png, i.e. a doubled extension like
        frame_000000.png.png)."""
        view_dir = os.path.join(self.masks_root, view)
        stem = os.path.splitext(filename)[0]
        candidates = [
            os.path.join(view_dir, stem + ".png"),
            os.path.join(view_dir, filename),
            os.path.join(view_dir, filename + ".png"),
        ]
        mask_path = next((p for p in candidates if os.path.exists(p)), None)
        if mask_path is None:
            raise FileNotFoundError(
                f"No rig mask found for view={view}, filename={filename}; "
                f"tried: {candidates}"
            )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Rig mask at {mask_path} could not be loaded.")
        if self.downsampling > 0.0 and self.downsampling != 1.0:
            mask = cv2.resize(
                mask, (0, 0),
                fx=1 / self.downsampling, fy=1 / self.downsampling,
                interpolation=cv2.INTER_AREA,
            )
        mask = torch.from_numpy(mask).float()[None] / 255.0
        return mask

    def _load_image(self, image_path: str, mode: int = cv2.IMREAD_COLOR):
        image = cv2.imread(image_path, mode)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        if self.downsampling > 0.0 and self.downsampling != 1.0:
            image = cv2.resize(
                image, (0, 0),
                fx=1 / self.downsampling, fy=1 / self.downsampling,
                interpolation=cv2.INTER_AREA,
            )
        image = cv2.cvtColor(
            image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB
        )
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image

    def _submit(self):
        if self.current_index < len(self):
            self.preload_queue.put(
                self.executor.submit(self.__getitem__, self.current_index)
            )

    def start_preloading(self):
        for self.current_index in range(self.num_threads):
            self._submit()

    def getnext(self):
        item = self.preload_queue.get().result()
        self.current_index += 1
        self._submit()
        return item
