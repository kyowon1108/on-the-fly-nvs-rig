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

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from socketserver import TCPServer
from http.server import SimpleHTTPRequestHandler
from args import get_args
from threading import Thread
from dataloaders.image_dataset import ImageDataset
from dataloaders.rig_image_dataset import RigImageDataset
from dataloaders.stream_dataset import StreamDataset
from poses.feature_detector import Detector
from poses.matcher import Matcher
from poses.pose_initializer import PoseInitializer
from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from scene.keyframe import Keyframe
from scene.mono_depth import MonoDepthEstimator
from scene.scene_model import SceneModel
from gaussianviewer import GaussianViewer
from webviewer.webviewer import WebViewer
from graphdecoviewer.types import ViewerMode
from utils import align_mean_up_fwd, increment_runtime, mtx2sixD


def _build_rig_completeness(expected, per_frame, failures, n_views):
    """Build split-wise source-timestep completeness for rig reporting."""
    expected = {
        split: sorted({int(ts) for ts in expected.get(split, [])})
        for split in ("all", "train", "test", "tracking")
    }
    registered_by_split = {split: set() for split in ("train", "test", "tracking")}
    views_by_ts = {}
    for row in per_frame:
        ts = int(row["source_ts"])
        split = row.get("rig_eval_split", "train")
        if split in registered_by_split:
            registered_by_split[split].add(ts)
        views_by_ts.setdefault(ts, set()).add(row.get("rig_view", ""))

    registered_all = set().union(*registered_by_split.values())
    failed_by_split = {split: set() for split in ("train", "test", "tracking")}
    failed_all = set()
    for failure in failures:
        ts = int(failure["source_ts"])
        split = failure.get("split", "tracking")
        failed_all.add(ts)
        if split in failed_by_split:
            failed_by_split[split].add(ts)

    def _recall(split, registered):
        exp = set(expected[split])
        return float(len(exp & registered) / len(exp)) if exp else 1.0

    view_counts = [len(v) for v in views_by_ts.values()]
    completeness = {
        "views_per_timestep_expected": int(n_views),
        "expected_timesteps_all": expected["all"],
        "registered_timesteps_all": sorted(registered_all),
        "failed_timesteps_all": sorted(failed_all),
        "missing_timesteps_all": sorted(set(expected["all"]) - registered_all),
        "failed_timestep_records": list(failures),
        "views_per_timestep_min": int(min(view_counts)) if view_counts else 0,
        "views_per_timestep_max": int(max(view_counts)) if view_counts else 0,
    }
    for split in ("train", "test", "tracking"):
        registered = registered_by_split[split]
        exp = set(expected[split])
        completeness[f"expected_timesteps_{split}"] = expected[split]
        completeness[f"registered_timesteps_{split}"] = sorted(registered)
        completeness[f"failed_timesteps_{split}"] = sorted(failed_by_split[split])
        completeness[f"missing_timesteps_{split}"] = sorted(exp - registered)
        completeness[f"registration_recall_{split}"] = _recall(split, registered)
    completeness["registration_recall_all"] = _recall("all", registered_all)
    return completeness


def _registered_rig_rows_from_keyframes(keyframes):
    rows = []
    for keyframe in keyframes:
        info = keyframe.info
        if "rig_view" not in info:
            continue
        if "source_ts" not in info:
            raise ValueError(f"Rig keyframe is missing source_ts: {info!r}")
        if "rig_eval_split" not in info:
            raise ValueError(f"Rig keyframe is missing rig_eval_split: {info!r}")
        rows.append({
            "source_ts": int(info["source_ts"]),
            "rig_view": info.get("rig_view"),
            "rig_eval_split": info["rig_eval_split"],
        })
    return rows


if __name__ == "__main__":
    args = get_args()
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize dataloader
    if "://" in args.source_path:
        dataset = StreamDataset(args.source_path, args.downsampling)
        is_stream = True
    elif args.use_rig:
        dataset = RigImageDataset(args)
        is_stream = False
    else:
        dataset = ImageDataset(args)
        is_stream = False
    height, width = dataset.get_image_size()

    # Initialize other modules
    print("Initializing modules and running just in time compilation, may take a while...")
    max_error = max(args.match_max_error * width, 1.5)
    min_displacement = max(args.min_displacement * width, 30)
    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(
        args.num_kpts, args.num_prev_keyframes_miniba_incr, max_error
    )
    pose_initializer = PoseInitializer(
        width, height, triangulator, matcher, 2 * max_error, args,
        rig_config=(dataset.rig if args.use_rig else None),
    )
    focal = pose_initializer.f_init
    dense_extractor = DenseExtractor(width, height)
    depth_estimator = MonoDepthEstimator(width, height)
    scene_model = SceneModel(width, height, args, matcher)
    if args.use_rig:
        # Rig 단위는 image가 아니라 timestep packet이다. SceneModel에는 N을
        # 알려서 active window가 `n_kept_timesteps * N` frame으로 잡히게 한다.
        scene_model.set_rig_view_count(len(dataset.rig.view_names))
        scene_model.set_rig_expected_timesteps(dataset.get_expected_timestep_splits())
        split_policy = (
            "ob3d_timestep_packet_split"
            if getattr(args, "rig_train_timesteps_file", "")
            else "timestep_packet_holdout"
            if getattr(args, "rig_test_timesteps_file", "")
            else "diagnostic_view_holdout"
            if getattr(args, "rig_holdout_view", "")
            else "no_holdout"
        )
        scene_model.rig_policy.update({
            "split_policy": split_policy,
            "pose_policy": "pose_assisted_online_all_registered_frames",
            "radiance_policy": "Gaussian spawn and photometric optimization use rig_eval_split == 'train' only",
            "metric_policy": "claim NVS metrics use rig_eval_split == 'test' only",
        })
    detector = Detector(args.num_kpts, width, height)

    # Initialize the viewer
    if args.viewer_mode in ["server", "local"]:
        viewer_mode = ViewerMode.SERVER if args.viewer_mode == "server" else ViewerMode.LOCAL
        viewer = GaussianViewer.from_scene_model(scene_model, viewer_mode)
        viewer_thd = Thread(target=viewer.run, args=(args.ip, args.port), daemon=True)
        viewer_thd.start()
        viewer.throttling = True # Enable throttling when training
    elif args.viewer_mode == "web":
        ip = "0.0.0.0"
        server = TCPServer((ip, 8000), SimpleHTTPRequestHandler)
        server_thd = Thread(target=server.serve_forever, daemon=True)
        server_thd.start()
        print(f"Visit http://{ip}:8000/webviewer to for the viewer")

        viewer = WebViewer(scene_model, args.ip, args.port)
        viewer_thd = Thread(target=viewer.run, daemon=True)
        viewer_thd.start()

    n_active_keyframes = 0
    n_keyframes = 0
    needs_reboot = False
    bootstrap_keyframe_dicts = []
    bootstrap_desc_kpts = []
    # Rig mode: loop 1회가 ref-first N-view packet 1개다. 처음 B개 packet은
    # bootstrap에 모으고, 이후 packet은 shared rig pose 하나씩 incremental 등록한다.
    bootstrap_rig_data = []
    n_rig_bootstrap_ts = 0

    # Dict of runtimes for each step
    runtimes = ["Load", "BAB", "tri", "BAI", "Add", "Init", "Opt", "anc"]
    runtimes = {key: [0, 0] for key in runtimes}
    metrics = {}

    ## Scene reconstruction
    print(f"Starting reconstruction for {args.source_path}")
    # Rig mode에서는 loop iteration이 image 수가 아니라 timestep 수다.
    total_iters = (
        dataset.num_timesteps if args.use_rig else len(dataset)
    )
    pbar = tqdm(range(0, total_iters))
    reconstruction_start_time = time.time()
    for frameID in pbar:
        start_time = time.time()

        if args.viewer_mode == "web":
            viewer.trainer_state = "running"

            # Paused
            while viewer.state == "stop":
                pbar.set_postfix_str(
                    "\033[31mPaused. Press the Start button in the webviewer\033[0m"
                )
                time.sleep(0.1)
            
            # Finish reconstruction
            if viewer.state == "finish":
                viewer.trainer_state = "finish"
                break
        
        # === Rig mode: 한 timestep의 N-view packet을 한 번에 소비 ===
        if args.use_rig:
            image, info = dataset.getnext()
            if info["rig_view"] != dataset.ref_view:
                raise RuntimeError(
                    "Rig dataset desync: expected a ref-view frame at batch "
                    f"start, got view={info['rig_view']} "
                    f"source_ts={info.get('source_ts')}"
                )
            if "source_ts" not in info or "stream_idx" not in info:
                raise RuntimeError(f"Rig ref frame missing source_ts/stream_idx: {info!r}")
            ts = int(info["source_ts"])
            stream_idx = int(info["stream_idx"])
            rig_batch = {dataset.ref_view: (image, info, detector(image))}
            for _ in range(len(dataset.non_ref_views)):
                nr_img, nr_info = dataset.getnext()
                if "source_ts" not in nr_info or "stream_idx" not in nr_info:
                    raise RuntimeError(
                        f"Rig non-ref frame missing source_ts/stream_idx: {nr_info!r}"
                    )
                nr_ts = int(nr_info["source_ts"])
                nr_stream_idx = int(nr_info["stream_idx"])
                if nr_ts != ts or nr_stream_idx != stream_idx:
                    raise RuntimeError(
                        "rig batch desync: expected "
                        f"source_ts={ts}, stream_idx={stream_idx}; got "
                        f"source_ts={nr_ts}, stream_idx={nr_stream_idx}"
                    )
                rig_batch[nr_info["rig_view"]] = (nr_img, nr_info, detector(nr_img))
            increment_runtime(runtimes["Load"], start_time)

            B = args.num_keyframes_miniba_bootstrap
            view_order = list(dataset.rig.view_names)

            if n_rig_bootstrap_ts < B:
                bootstrap_rig_data.append({
                    "ts": ts,
                    "source_ts": ts,
                    "stream_idx": stream_idx,
                    "frames": rig_batch,
                })
                n_rig_bootstrap_ts += 1
                if n_rig_bootstrap_ts < B:
                    continue

                # --- Rig bootstrap: B timesteps x N views를 한 번에 MiniBARig로 초기화 ---
                start_time = time.time()
                desc_per_ts_per_view = [
                    {v: data["frames"][v][2] for v in view_order}
                    for data in bootstrap_rig_data
                ]
                rig_Rts, f_out, res, _xyz, _view_names = (
                    pose_initializer.initialize_bootstrap_rig(
                        desc_per_ts_per_view, dataset.rig,
                    )
                )
                focal = f_out.cpu().item()
                increment_runtime(runtimes["BAB"], start_time)

                # Bootstrap 결과를 photometric optimizer가 소유하는 shared rig pose로
                # 넘긴다. 각 slot은 SE(3) pose 하나이며 raw parameter는 6D rotation
                # + 3D translation이다. View pose는 `rel @ rig`로 파생된다.
                rig_R6D_init = mtx2sixD(rig_Rts[:, :3, :3].contiguous())
                rig_t_init = rig_Rts[:, :3, 3].contiguous()
                scene_model.register_rig_poses(rig_R6D_init, rig_t_init, lr=args.lr_poses)

                f_tensor = torch.tensor([focal], device="cuda", dtype=torch.float32)
                ref_kf_scene_indices = []  # indices (into scene_model.keyframes) of ref keyframes
                start_time = time.time()
                for ts_i, data in enumerate(bootstrap_rig_data):
                    rig_pose = rig_Rts[ts_i]
                    for v_name in view_order:
                        img, inf, desc = data["frames"][v_name]
                        rel = inf["rig_relative_Rt"].to("cuda")
                        Rt_view = rel @ rig_pose
                        # `ts_idx`는 optimizer slot이다. Keyframe은 이 값을 통해
                        # 자유 pose parameter 대신 shared rig pose를 참조한다.
                        inf["ts_idx"] = ts_i
                        inf["rig_view"] = v_name
                        kf = Keyframe(
                            img, inf, desc, Rt_view, n_keyframes, f_tensor,
                            dense_extractor, depth_estimator, triangulator, args,
                        )
                        # Push the focal into scene_model on the very first keyframe.
                        first = (ts_i == 0 and v_name == view_order[0])
                        scene_model.add_keyframe(kf, f_tensor if first else None)
                        if v_name == dataset.ref_view:
                            ref_kf_scene_indices.append(len(scene_model.keyframes) - 1)
                        n_keyframes += 1
                increment_runtime(runtimes["Add"], start_time)

                # Gaussian 초기화는 ref view 하나가 아니라 packet의 모든 view에서 만든다.
                # 각 view는 bootstrap time-axis match로 3D point를 채우고, packet 전체를
                # plan한 뒤 한 번 commit한다.
                start_time = time.time()
                first_bootstrap_scene_idx = (
                    len(scene_model.keyframes)
                    - len(bootstrap_rig_data) * len(view_order)
                )
                n_views = len(view_order)
                for packet_start in range(
                    first_bootstrap_scene_idx,
                    len(scene_model.keyframes),
                    n_views,
                ):
                    scene_model.add_new_gaussians_for_keyframes(
                        list(range(packet_start, packet_start + n_views))
                    )
                increment_runtime(runtimes["Init"], start_time)

                start_time = time.time()
                if is_stream:
                    scene_model.optimize_async(args.num_iterations)
                else:
                    scene_model.optimization_loop(args.num_iterations)
                increment_runtime(runtimes["Opt"], start_time)
                last_reboot = n_keyframes
                if args.viewer_mode not in ["none", "web"]:
                    viewer.reset_intrinsics("point_view")
                continue

            # --- Rig incremental (post-bootstrap) ---
            start_time = time.time()
            view_indices = {v: n_keyframes + i for i, v in enumerate(view_order)}
            desc_per_view = {v: rig_batch[v][2] for v in view_order}
            # 각 view는 자기 feature에 맞는 과거 keyframe pool을 따로 고른다. 카메라가
            # 크게 회전하면 ref view의 후보가 다른 view에는 나쁠 수 있다. 후보 pool을
            # 먼저 모은 뒤 unique prev keyframe만 한 번씩 3D point를 refresh한다.
            prev_per_view = {}
            target_centre = scene_model._live_centres_for_keyframe_ids(
                [len(scene_model.keyframes) - 1]
            )[0]
            for v in view_order:
                prev_per_view[v] = scene_model.get_prev_keyframes(
                    args.num_prev_keyframes_miniba_incr, False, desc_per_view[v],
                    target_centre=target_centre,
                )
            seen_prev = set()
            for prevs in prev_per_view.values():
                for kf in prevs:
                    if kf.index in seen_prev:
                        continue
                    seen_prev.add(kf.index)
                    kf.update_3dpts(scene_model.keyframes)
            increment_runtime(runtimes["tri"], start_time)

            start_time = time.time()
            rig_pose, rig_pnp_stats = pose_initializer.initialize_incremental_rig(
                prev_per_view, desc_per_view, view_indices, dataset.rig,
            )
            increment_runtime(runtimes["BAI"], start_time)
            scene_model.extra_metadata.setdefault("rig_pnp_per_timestep", []).append({
                "source_ts": int(ts),
                "stream_idx": int(stream_idx),
                "split": info.get("rig_eval_split", "tracking"),
                "stats": rig_pnp_stats,
            })
            if rig_pose is None:
                scene_model.record_rig_timestep_failure(
                    source_ts=ts,
                    stream_idx=stream_idx,
                    split=info.get("rig_eval_split", "tracking"),
                    reason="initialize_incremental_rig returned None",
                )
                continue

            # 새 timestep의 shared rig pose를 optimizer slot 하나로 추가한다.
            # 이후 N개 view는 모두 이 slot(`new_ts_idx`)에서 pose를 파생한다.
            new_R6D = mtx2sixD(rig_pose[:3, :3][None].contiguous())[0]
            new_t = rig_pose[:3, 3].contiguous()
            scene_model.append_rig_pose(new_R6D, new_t)
            new_ts_idx = len(scene_model.rig_R6D) - 1

            f_tensor = torch.tensor([focal], device="cuda", dtype=torch.float32)
            start_time = time.time()
            new_scene_indices = []
            for v_name in view_order:
                img, inf, desc = rig_batch[v_name]
                rel = inf["rig_relative_Rt"].to("cuda")
                Rt_view = rel @ rig_pose
                inf["ts_idx"] = new_ts_idx
                inf["rig_view"] = v_name
                kf = Keyframe(
                    img, inf, desc, Rt_view, n_keyframes, f_tensor,
                    dense_extractor, depth_estimator, triangulator, args,
                )
                scene_model.add_keyframe(kf)
                new_scene_indices.append(len(scene_model.keyframes) - 1)
                n_keyframes += 1
            # Timestep packet invariant: N개 view를 모두 등록한 뒤 spawn plan을 만들고,
            # Gaussian optimizer mutation은 한 번만 commit한다. 그래서 같은 packet 안의
            # later view가 already-mutated scene을 보는 일을 막는다.
            scene_model.add_new_gaussians_for_keyframes(new_scene_indices)
            increment_runtime(runtimes["Add"], start_time)
            start_time = time.time()
            if is_stream:
                scene_model.optimize_async(args.num_iterations)
            else:
                scene_model.optimization_loop(args.num_iterations)
            increment_runtime(runtimes["Opt"], start_time)
            start_time = time.time()
            scene_model.place_anchor_if_needed()
            increment_runtime(runtimes["anc"], start_time)
            continue

        # === Non-rig path ===
        if n_keyframes == 0:
            image, info = dataset.getnext()
            prev_desc_kpts = detector(image)
            bootstrap_keyframe_dicts = [{"image": image, "info": info}]
            bootstrap_desc_kpts = [prev_desc_kpts]
            n_keyframes += 1
            continue

        image, info = dataset.getnext()

        desc_kpts = detector(image)
        # Match features between the previous and current frame
        curr_prev_matches = matcher(desc_kpts, prev_desc_kpts)
        # Determine if we should add a keyframe based on the matches
        dist = torch.norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other, dim=-1)
        should_add_keyframe = (
            dist.median() > min_displacement
            and len(curr_prev_matches.kpts) > args.min_num_inliers
        )
        # Always add test frames so we estimate their poses
        should_add_keyframe |= info["is_test"]
        increment_runtime(runtimes["Load"], start_time)

        if should_add_keyframe:
            ## Bootstrap
            # Accumulate keyframes for pose initialization
            if n_keyframes < args.num_keyframes_miniba_bootstrap:
                bootstrap_keyframe_dicts.append({"image": image, "info": info})
                bootstrap_desc_kpts.append(desc_kpts)

            if n_keyframes == args.num_keyframes_miniba_bootstrap - 1:
                start_time = time.time()
                Rts, f, _ = pose_initializer.initialize_bootstrap(bootstrap_desc_kpts)
                focal = f.cpu().item()
                increment_runtime(runtimes["BAB"], start_time)
                for index, (keyframe_dict, desc_kpts, Rt) in enumerate(
                    zip(bootstrap_keyframe_dicts, bootstrap_desc_kpts, Rts)
                ):
                    start_time = time.time()
                    if args.use_colmap_poses:
                        Rt = keyframe_dict["info"]["Rt"]
                        f = keyframe_dict["info"]["focal"]
                    keyframe = Keyframe(
                        keyframe_dict["image"],
                        keyframe_dict["info"],
                        desc_kpts,
                        Rt,
                        index,
                        f,
                        dense_extractor,
                        depth_estimator,
                        triangulator,
                        args,
                    )
                    scene_model.add_keyframe(keyframe, f)
                    increment_runtime(runtimes["Add"], start_time)
                if args.viewer_mode not in ["none", "web"]:
                    viewer.reset_intrinsics("point_view")
                prev_keyframe = keyframe
                for index in range(args.num_keyframes_miniba_bootstrap):
                    start_time = time.time()
                    scene_model.add_new_gaussians(index)
                    increment_runtime(runtimes["Init"], start_time)
                start_time = time.time()
                # Run initial optimization on the bootstrap keyframes
                # If streaming, run async optimization until the next keyframe is added
                if is_stream:
                    scene_model.optimize_async(args.num_iterations)
                else:
                    scene_model.optimization_loop(args.num_iterations)
                increment_runtime(runtimes["Opt"], start_time)
                last_reboot = n_keyframes

            ## Reboot
            if (
                args.enable_reboot
                and scene_model.approx_cam_centres is not None
                and len(scene_model.anchors)
            ):
                # Check if the camera baseline is a lot smaller or larger than expected
                last_centers = scene_model.approx_cam_centres[-20:]
                rel_dist = torch.norm(
                    last_centers[1:] - last_centers[:-1], dim=-1
                ).mean()
                needs_reboot = (
                    rel_dist > 0.1 * 5 or rel_dist < 0.1 / 3
                ) and n_keyframes - last_reboot > 50
            if needs_reboot:
                # Reboot: run mini BA on the last 8 keyframes
                bs_kfs = scene_model.keyframes[-8:]
                bootstrap_desc_kpts = [bs_kf.desc_kpts for bs_kf in bs_kfs]
                in_Rts = torch.stack([kf.get_Rt() for kf in bs_kfs])
                Rts, _, final_residual = pose_initializer.initialize_bootstrap(
                    bootstrap_desc_kpts, rebooting=True
                )
                # Check if the reboot succeeded
                if final_residual < max_error * 0.5:
                    Rts = align_mean_up_fwd(Rts, in_Rts)
                    for Rt, keyframe in zip(Rts, bs_kfs):
                        keyframe.set_Rt(Rt)
                    # Reset the scene model and reinitialize the gaussians
                    scene_model.reset()
                    for i in range(3, 0, -1):
                        scene_model.add_new_gaussians(-i)
                    for _ in range(3 * args.num_iterations):
                        scene_model.optimization_step()
                    needs_reboot = False
                    last_reboot = n_keyframes

            ## Incremental reconstruction
            # Incremental pose initialization
            if n_keyframes >= args.num_keyframes_miniba_bootstrap:
                start_time = time.time()
                prev_keyframes = scene_model.get_prev_keyframes(
                    args.num_prev_keyframes_miniba_incr, True, desc_kpts
                )
                increment_runtime(runtimes["tri"], start_time)
                start_time = time.time()
                Rt = pose_initializer.initialize_incremental(
                    prev_keyframes, desc_kpts, n_keyframes, info["is_test"], image
                )
                increment_runtime(runtimes["BAI"], start_time)
                start_time = time.time()
                if Rt is not None:
                    if args.use_colmap_poses:
                        Rt = info["Rt"]
                    keyframe = Keyframe(
                        image,
                        info,
                        desc_kpts,
                        Rt,
                        n_keyframes,
                        f,
                        dense_extractor,
                        depth_estimator,
                        triangulator,
                        args,
                    )
                    scene_model.add_keyframe(keyframe)
                    prev_keyframe = keyframe
                    increment_runtime(runtimes["Add"], start_time)
                    # Gaussian initialization
                    start_time = time.time()
                    scene_model.add_new_gaussians()
                    increment_runtime(runtimes["Init"], start_time)
                    start_time = time.time()
                    # If streaming, run async optimization until the next keyframe is added
                    if is_stream:
                        scene_model.optimize_async(args.num_iterations)
                    else:
                        scene_model.optimization_loop(args.num_iterations)
                    increment_runtime(runtimes["Opt"], start_time)
                else:
                    should_add_keyframe = False

        if should_add_keyframe:
            ## Check if anchor creation is needed based on the primitives' size 
            start_time = time.time()
            scene_model.place_anchor_if_needed()
            increment_runtime(runtimes["anc"], start_time)

            n_keyframes += 1
            if not info["is_test"]:
                prev_desc_kpts = desc_kpts

            ## Intermediate evaluation
            if (
                n_keyframes % args.test_frequency == 0
                and args.test_frequency > 0
                and (args.test_hold > 0 or args.eval_poses)
            ):
                metrics = scene_model.evaluate(args.eval_poses)

            ## Save intermediate model
            if (
                frameID % args.save_every == 0
                and args.save_every > 0
            ):
                scene_model.save(
                    os.path.join(args.model_path, "progress", f"{frameID:05d}")
                )

            ## Display optimization progress and metrics
            bar_postfix = []
            for key, value in metrics.items():
                bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
            if args.display_runtimes:
                for key, value in runtimes.items():
                    if value[1] > 0:
                        bar_postfix += [
                            f"\033[35m{key}:{1000 * value[0] / value[1]:.1f}\033[0m"
                        ]
            bar_postfix += [
                f"\033[36mFocal:{focal:.1f}",
                f"\033[36mKeyframes:{n_keyframes}\033[0m",
                f"\033[36mGaussians:{scene_model.n_active_gaussians}\033[0m",
                f"\033[36mAnchors:{len(scene_model.anchors)}\033[0m",
            ]
            pbar.set_postfix_str(",".join(bar_postfix), refresh=False)

    reconstruction_time = time.time() - reconstruction_start_time

    # Set to inference mode so that the model can be rendered properly
    scene_model.enable_inference_mode()

    # === Post-hoc render evaluation (all keyframes) =========================
    # Rig eval is split after registration: render each registered keyframe from
    # its own pose, compare against GT RGB, and write separate train/test/
    # tracking metrics. This pass is no-grad and never feeds metric frames back
    # into Gaussian optimization.
    if getattr(args, "use_rig", False):
        import cv2
        import numpy as np
        from fused_ssim import fused_ssim
        from utils import psnr as psnr_fn
        try:
            import lpips as lpips_mod
            lpips_fn = lpips_mod.LPIPS(net="vgg").cuda().eval()
        except Exception:
            lpips_fn = None
        eval_dir = os.path.join(args.model_path, "render_eval")
        os.makedirs(eval_dir, exist_ok=True)
        per_frame = []
        skipped_frames = []
        mask_applied_count = 0
        with torch.no_grad():
            for kf in scene_model.keyframes:
                pkg = scene_model.render_from_id(kf.index, pyr_lvl=0)
                rendered = pkg["render"].clamp(0, 1)
                gt = kf.get_eval_image().to(rendered.device)
                if gt.shape[-2:] != rendered.shape[-2:]:
                    skipped_frames.append({
                        "name": kf.info.get("name", f"kf{kf.index:04d}"),
                        "source_ts": int(kf.info["source_ts"]),
                        "rig_view": kf.info.get("rig_view"),
                        "rig_eval_split": kf.info.get("rig_eval_split", "unknown"),
                        "reason": "render_gt_shape_mismatch",
                        "render_shape": list(rendered.shape),
                        "gt_shape": list(gt.shape),
                    })
                    continue
                metric_rendered = rendered
                metric_gt = gt
                mask = kf.get_eval_mask()
                mask_applied = False
                metric_mask = None
                if mask is not None:
                    mask = mask.to(rendered.device).float()
                    if mask.shape[-2:] != rendered.shape[-2:]:
                        mask = F.interpolate(
                            mask[None],
                            size=rendered.shape[-2:],
                            mode="nearest",
                        )[0]
                    mask = (mask > 0.5).expand_as(rendered)
                    metric_rendered = rendered * mask
                    metric_gt = gt * mask
                    metric_mask = mask
                    mask_applied = True
                    mask_applied_count += 1
                if metric_mask is not None and metric_mask.any():
                    p = float(psnr_fn(metric_rendered[metric_mask], metric_gt[metric_mask]))
                else:
                    p = float(psnr_fn(metric_rendered, metric_gt))
                s = float(fused_ssim(metric_rendered[None], metric_gt[None], train=False).item())
                if lpips_fn is not None:
                    r01 = metric_rendered[None] * 2 - 1
                    g01 = metric_gt[None] * 2 - 1
                    l = float(lpips_fn(r01, g01).item())
                else:
                    l = float("nan")
                name = kf.info.get("name", f"kf{kf.index:04d}")
                if not name.endswith(".png"):
                    name = name + ".png"
                # side-by-side: rendered | gt
                img_np = torch.cat([rendered, gt], dim=-1)
                img_np = (img_np.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(eval_dir, name), img_np)
                per_frame.append({
                    "name": name,
                    "source_ts": int(kf.info["source_ts"]),
                    "rig_view": kf.info.get("rig_view"),
                    "rig_eval_split": kf.info.get(
                        "rig_eval_split",
                        "test" if kf.info.get("is_test", False) else "train",
                    ),
                    "is_test": bool(kf.info.get("is_test", False)),
                    "mask_applied": mask_applied,
                    "psnr": p, "ssim": s, "lpips": l,
                })
        if per_frame:
            psnrs = [x["psnr"] for x in per_frame]
            ssims = [x["ssim"] for x in per_frame]
            lpipss = [x["lpips"] for x in per_frame if not (x["lpips"] != x["lpips"])]
            summary = {
                "num_frames": len(per_frame),
                "psnr_mean": float(np.mean(psnrs)),
                "ssim_mean": float(np.mean(ssims)),
                "lpips_mean": (float(np.mean(lpipss)) if lpipss else float("nan")),
                "psnr_min": float(np.min(psnrs)),
                "psnr_max": float(np.max(psnrs)),
                "skipped_count": len(skipped_frames),
                "mask_applied_frames": int(mask_applied_count),
            }
            rig_policy = dict(scene_model.rig_policy)
            split_meta = {
                "mode": (
                    "ob3d"
                    if getattr(args, "rig_train_timesteps_file", "")
                    else "timestep"
                    if getattr(args, "rig_test_timesteps_file", "")
                    else ("view" if getattr(args, "rig_holdout_view", "") else "none")
                ),
                "rig_holdout_view": getattr(args, "rig_holdout_view", ""),
                "rig_train_timesteps_file": getattr(args, "rig_train_timesteps_file", ""),
                "rig_test_timesteps_file": getattr(args, "rig_test_timesteps_file", ""),
                "pose_policy": rig_policy["pose_policy"],
                "radiance_policy": rig_policy["radiance_policy"],
                "metric_policy": rig_policy["metric_policy"],
                "split_policy": rig_policy["split_policy"],
                "rig_holdout_view_is_diagnostic": bool(getattr(args, "rig_holdout_view", "")),
                "claim_grade_split": bool(
                    getattr(args, "rig_train_timesteps_file", "")
                    and getattr(args, "rig_test_timesteps_file", "")
                ),
                "mask_policy": {
                    "masks_dir": getattr(args, "masks_dir", ""),
                    "mask_requested": bool(getattr(args, "masks_dir", "")),
                    "mask_applied_frames": int(mask_applied_count),
                },
                "skipped_count": len(skipped_frames),
                "skipped_frames": skipped_frames,
                "test_timesteps": sorted({
                    int(x["source_ts"]) for x in per_frame
                    if x.get("rig_eval_split") == "test"
                }),
                "train_timesteps": sorted({
                    int(x["source_ts"]) for x in per_frame
                    if x.get("rig_eval_split") == "train"
                }),
                "tracking_timesteps": sorted({
                    int(x["source_ts"]) for x in per_frame
                    if x.get("rig_eval_split") == "tracking"
                }),
                "test_views": sorted({
                    str(x["rig_view"]) for x in per_frame
                    if x.get("rig_eval_split") == "test"
                }),
            }
            import json as _json
            with open(os.path.join(eval_dir, "metrics.json"), "w") as _f:
                _json.dump(
                    {
                        "summary": summary,
                        "split": split_meta,
                        "skipped_frames": skipped_frames,
                        "per_frame": per_frame,
                    },
                    _f,
                    indent=2,
                )
            print(
                "[post-hoc render] "
                f"n={summary['num_frames']}  "
                f"PSNR={summary['psnr_mean']:.2f}  "
                f"SSIM={summary['ssim_mean']:.3f}  "
                f"LPIPS={summary['lpips_mean']:.3f}  "
                f"(range PSNR {summary['psnr_min']:.2f}–{summary['psnr_max']:.2f})"
            )
            # Rig NVS metric: split optimized training frames from held-out
            # frames. `--rig_test_timesteps_file` is the OB3D-style
            # split (all N virtual views from held-out EQR timesteps). The older
            # `--rig_holdout_view` split is a diagnostic for unseen direction at
            # seen timesteps.
            test_pf = [x for x in per_frame if x.get("rig_eval_split") == "test"]
            train_pf = [x for x in per_frame if x.get("rig_eval_split") == "train"]
            tracking_pf = [
                x for x in per_frame if x.get("rig_eval_split") == "tracking"
            ]
            registered_rows = _registered_rig_rows_from_keyframes(scene_model.keyframes)
            rig_completeness = _build_rig_completeness(
                scene_model.rig_expected_timesteps,
                registered_rows,
                scene_model.rig_failed_timesteps,
                len(dataset.rig.view_names),
            )
            scene_model.rig_completeness = rig_completeness
            scene_model.extra_metadata["rig_completeness"] = rig_completeness
            scene_model.extra_metadata["rig_policy"] = rig_policy
            def _summarize_rows(rows):
                if not rows:
                    return {
                        "num_frames": 0,
                        "psnr_mean": float("nan"),
                        "ssim_mean": float("nan"),
                        "lpips_mean": float("nan"),
                    }
                rp = [x["psnr"] for x in rows]
                rs = [x["ssim"] for x in rows]
                rl = [x["lpips"] for x in rows
                      if not (x["lpips"] != x["lpips"])]
                return {
                    "num_frames": len(rows),
                    "psnr_mean": float(np.mean(rp)),
                    "ssim_mean": float(np.mean(rs)),
                    "lpips_mean": (float(np.mean(rl)) if rl else float("nan")),
                }

            skipped_test_count = sum(
                1 for row in skipped_frames if row.get("rig_eval_split") == "test"
            )
            expected_test_ts = rig_completeness.get("expected_timesteps_test", [])
            if split_meta["mode"] == "view":
                expected_test_frames = len(expected_test_ts) * max(len(split_meta["test_views"]), 1)
            else:
                expected_test_frames = len(expected_test_ts) * len(dataset.rig.view_names)
            claim_metric_warnings = []
            if skipped_test_count:
                claim_metric_warnings.append(
                    f"{skipped_test_count} test frames were skipped during post-hoc metric"
                )
            if len(test_pf) != expected_test_frames:
                claim_metric_warnings.append(
                    f"rendered test frames {len(test_pf)} != expected {expected_test_frames}"
                )
            if rig_completeness.get("missing_timesteps_test"):
                claim_metric_warnings.append(
                    "registered test timesteps are missing from the trajectory"
                )
            claim_metric_complete = not claim_metric_warnings
            test_summary = _summarize_rows(test_pf)
            test_summary.update({
                "pose_policy": rig_policy["pose_policy"],
                "radiance_policy": rig_policy["radiance_policy"],
                "metric_policy": rig_policy["metric_policy"],
                "split_policy": rig_policy["split_policy"],
                "mask_policy": split_meta["mask_policy"],
                "skipped_count": len(skipped_frames),
                "skipped_test_count": int(skipped_test_count),
                "expected_test_frames": int(expected_test_frames),
                "claim_metric_complete": bool(claim_metric_complete),
                "claim_metric_warnings": claim_metric_warnings,
            })
            split_metrics = {
                "all": {**summary, **rig_policy, "mask_policy": split_meta["mask_policy"]},
                "test": test_summary,
                "train": _summarize_rows(train_pf),
                "tracking": _summarize_rows(tracking_pf),
                "split": split_meta,
                "policy": rig_policy,
                "skipped_frames": skipped_frames,
                "skipped_count": len(skipped_frames),
                "skipped_test_count": int(skipped_test_count),
                "claim_metric_complete": bool(claim_metric_complete),
                "rig_leakage_audit": dict(scene_model.rig_leakage_audit),
                "rig_completeness": rig_completeness,
            }
            with open(os.path.join(eval_dir, "split_metrics.json"), "w") as _f:
                _json.dump(split_metrics, _f, indent=2)
            with open(os.path.join(eval_dir, "metrics_claim_test.json"), "w") as _f:
                _json.dump(split_metrics["test"], _f, indent=2)
            with open(os.path.join(eval_dir, "metrics_diagnostic_all.json"), "w") as _f:
                _json.dump(split_metrics["all"], _f, indent=2)
            with open(os.path.join(eval_dir, "metrics_diagnostic_tracking.json"), "w") as _f:
                _json.dump(split_metrics["tracking"], _f, indent=2)
            with open(os.path.join(eval_dir, "rig_leakage_audit.json"), "w") as _f:
                _json.dump(split_metrics["rig_leakage_audit"], _f, indent=2)
            with open(os.path.join(eval_dir, "rig_completeness.json"), "w") as _f:
                _json.dump(rig_completeness, _f, indent=2)
            with open(os.path.join(eval_dir, "metrics.json"), "w") as _f:
                _json.dump(
                    {
                        "summary": summary,
                        "test_summary": split_metrics["test"],
                        "train_summary": split_metrics["train"],
                        "policy": rig_policy,
                        "claim_metric_complete": bool(claim_metric_complete),
                        "skipped_frames": skipped_frames,
                        "rig_leakage_audit": split_metrics["rig_leakage_audit"],
                        "rig_completeness": rig_completeness,
                        "split": split_meta,
                        "per_frame": per_frame,
                    },
                    _f,
                    indent=2,
                )
            if test_pf:
                hp = [x["psnr"] for x in test_pf]
                hs = [x["ssim"] for x in test_pf]
                hl = [x["lpips"] for x in test_pf
                      if not (x["lpips"] != x["lpips"])]
                tp = [x["psnr"] for x in train_pf]
                ts = [x["ssim"] for x in train_pf]
                tl = [x["lpips"] for x in train_pf
                      if not (x["lpips"] != x["lpips"])]
                if getattr(args, "rig_test_timesteps_file", ""):
                    held_ts = sorted({int(x["source_ts"]) for x in test_pf})
                    split_label = (
                        f"test timesteps={len(held_ts)} "
                        f"file={args.rig_test_timesteps_file}"
                    )
                else:
                    split_label = f"holdout view={args.rig_holdout_view}"
                print(
                    f"[{split_label}] "
                    f"n={len(test_pf)}  "
                    f"PSNR={float(np.mean(hp)):.2f}  "
                    f"SSIM={float(np.mean(hs)):.3f}  "
                    f"LPIPS={(float(np.mean(hl)) if hl else float('nan')):.3f}"
                )
                if tracking_pf:
                    print(f"[tracking-only frames] n={len(tracking_pf)}  metrics excluded")
                print(
                    f"[train views] "
                    f"n={len(train_pf)}  "
                    f"PSNR={float(np.mean(tp)):.2f}  "
                    f"SSIM={float(np.mean(ts)):.3f}  "
                    f"LPIPS={(float(np.mean(tl)) if tl else float('nan')):.3f}"
                )
    # ========================================================================

    # Save the model and metrics
    print("Saving the reconstruction to:", args.model_path)
    if getattr(args, "use_rig", False):
        scene_model.extra_metadata["rig_incremental_refinement"] = {
            "calls": int(getattr(pose_initializer, "_refine_call_count", 0)),
            "fallbacks": int(getattr(pose_initializer, "_refine_fail_count", 0)),
            "fallback_rate": (
                float(getattr(pose_initializer, "_refine_fail_count", 0))
                / max(int(getattr(pose_initializer, "_refine_call_count", 0)), 1)
            ),
            "fallback_policy": "MiniBARig incremental refinement failure falls back to rig PnP and is recorded here.",
        }
    metrics = scene_model.save(
        args.model_path,
        reconstruction_time,
        len(dataset),
        total_iters if getattr(args, "use_rig", False) else 0,
    )
    print(
        ", ".join(
            f"{metric}: {value:.3f}"
            if isinstance(value, float)
            else f"{metric}: {value}"
            for metric, value in metrics.items()
        )
    )

    # Fine tuning after initial reconstruction
    if len(args.save_at_finetune_epoch) > 0:
        finetune_epochs = max(args.save_at_finetune_epoch)
        torch.cuda.empty_cache()
        scene_model.inference_mode = False
        pbar = tqdm(range(0, finetune_epochs), desc="Fine tuning")
        for epoch in pbar:
            # Run one epoch of fine-tuning
            epoch_start_time = time.time()
            scene_model.finetune_epoch()
            epoch_time = time.time() - epoch_start_time
            reconstruction_time += epoch_time
            # Save the model and metrics
            if epoch + 1 in args.save_at_finetune_epoch:
                torch.cuda.empty_cache()
                scene_model.inference_mode = True
                metrics = scene_model.save(
                    os.path.join(args.model_path, str(epoch + 1)), reconstruction_time
                )
                bar_postfix = []
                for key, value in metrics.items():
                    bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
                pbar.set_postfix_str(",".join(bar_postfix))
                scene_model.inference_mode = False
                torch.cuda.empty_cache()
                
        # Set to inference mode so that the model can be rendered properly
        scene_model.inference_mode = True

    if args.viewer_mode != "none":
        # After the reconstruction loop, keep the viewer in stable inference
        # mode. During the loop, SceneModel.lock serializes live viewer renders
        # with Gaussian spawn/prune/optimization.
        scene_model.inference_mode = True

    if args.viewer_mode != "none":
        if args.viewer_mode == "web":
            while True:
                time.sleep(1)
        else:
            viewer.throttling = False # Disable throttling when done training
            # Loop to keep the viewer alive
            while viewer.running:
                time.sleep(1)
