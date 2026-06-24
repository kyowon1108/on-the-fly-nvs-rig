# OTF Rig Convention

이 문서는 `feat/rig-core`의 zero-baseline virtual panoramic rig가 어떤 단위를
훈련 단위로 삼고, 어떤 원본 OTF 기능을 그대로 포팅하면 안 되는지 고정한다.
논문/실험 문서에서는 이 파일의 용어를 그대로 사용한다.

## 1. 세 단위: timestep, view, keyframe

`timestep`은 원본 EQR 한 장에서 나온 모든 virtual pinhole view의 묶음이다.
OB3D의 `frame_00037.png` EQR을 12개 pinhole로 쪼갰다면, 그 12장은 같은
`rig_ts=37`을 가진다. `view`는 하나의 EQR에서 렌더링한 pinhole sensor이고,
`keyframe`은 `(timestep, view)` 하나다. 따라서 100 timestep, 12 view이면
`SceneModel.keyframes`에는 최대 1200개 keyframe이 들어가지만, pose 자유도는
100개 shared rig pose뿐이다.

코드 기준:

- `dataloaders/rig_image_dataset.py::RigImageDataset`
  - 모든 view 폴더의 공통 파일명을 교집합으로 잡아 `timestep_names`를 만든다.
  - `--start_at`은 image index가 아니라 timestep index다.
  - batch 순서는 `ref_view` 먼저, 나머지는 `rig_config.view_names` 순서다.
- `train.py`
  - `args.use_rig=True`이면 tqdm loop 한 번이 image 한 장이 아니라 timestep 하나다.

## 2. Zero-baseline rig geometry

Rig loader는 per-view relative transform을 COLMAP-style world-to-camera로 만든다.
핵심 관계는 다음 하나다.

$$
T^{(v)}_{w\to c}(t) = T^{(v)}_{\mathrm{rig}\to c}\;T^{\mathrm{rig}}_{w\to r}(t)
$$

여기서

$$
T^{(v)}_{\mathrm{rig}\to c} =
\begin{bmatrix}
R_v & 0 \\
0 & 1
\end{bmatrix}
$$

이다. 즉 모든 view의 `rel_t`는 정확히 0이다. view별 카메라 중심은

$$
C^{(v)}_t
= -\left(R_v R_t\right)^T \left(R_v t_t\right)
= -R_t^T t_t
$$

으로 동일하다. 그래서 같은 timestep 안의 view들은 회전만 다르고 baseline은 없다.

코드 기준:

- `rig/rig_loader.py::load_rig_config`
  - `relative_Rt[view]`를 만들고 translation을 0으로 강제한다.
- `scene/keyframe.py::Keyframe.get_R`, `get_t`, `get_centre`
  - rig mode keyframe pose는 자체 `nn.Parameter`가 아니라
    `scene_model.rig_R6D[ts_idx]`, `scene_model.rig_t[ts_idx]`에서 매번 유도된다.
- `scene/scene_model.py::get_live_rig_centres`
  - 위의 수식을 batch로 계산해 neighbour sort에 쓴다.

## 3. Same-timestep exclusion은 필수 constraint다

같은 timestep의 view들은 baseline이 0이므로 triangulation, PnP partner, guided MVS
partner로 쓰면 depth가 정의되지 않는다.

$$
\text{partner}(k_{t,v}) \notin \{k_{t,u}\mid u\neq v\}
$$

코드 기준:

- `poses/pose_initializer.py::initialize_bootstrap_rig`
  - bootstrap matching은 view별 시간축 exhaustive matching만 수행한다.
  - cross-view matching은 의도적으로 하지 않는다.
- `scene/scene_model.py::get_prev_keyframes`
  - `exclude_ts`가 있으면 같은 `rig_ts` 후보를 candidate pool에서 먼저 제거한다.
- `scene/scene_model.py::add_new_gaussians`
  - guided MVS neighbour 선택에도 `exclude_ts=keyframe.info["rig_ts"]`를 넘긴다.

마스크는 same-timestep exclusion의 본체가 아니다. `--masks_dir` 마스크는 loss/eval/spawn
영역을 제한하는 pixel gate이고, zero-baseline depth leakage 차단은 `exclude_ts`가 담당한다.

## 4. Bootstrap dataflow

Canonical command:

```bash
python train.py -s /home/kaprub22/otfrig/pinhole_rig/ob3d_rig/classroom_100 \
  --use_rig --rig_config /home/kaprub22/otfrig/tools/configs/rig12_panosfm.json \
  --ref_view E+0_A000 --init_focal 200 --fix_focal \
  --downsampling 1 --num_keyframes_miniba_bootstrap 8 \
  --num_iterations 270 --seed 0 --viewer_mode none \
  -m results/<run_name>
```

흐름:

1. `RigImageDataset.getnext()`가 ref view 한 장을 반환한다.
2. `train.py`가 같은 `rig_ts`의 나머지 N-1 view를 이어서 읽어 `rig_batch`를 만든다.
3. bootstrap 동안 B개 timestep을 모은다.
4. `PoseInitializer.initialize_bootstrap_rig()`가 `(B, N)` 관측을 MiniBARig 문제로 만든다.
5. 초기 3D point는 각 view의 첫 valid observation ray에 unit depth를 둔 뒤 rig frame으로
   lift한다. 이 단계에는 mono-depth seed가 없다.
6. `SceneModel.register_rig_poses()`가 B개 shared rig pose를 optimizer 소유로 등록한다.
7. 각 `(t,v)` keyframe은 `rel @ rig` pose로 생성되지만, 이후 pose는 shared rig slot에서
   유도된다.
8. `SceneModel.add_new_gaussians()`가 모든 view에서 Gaussian을 spawn한다.
9. `SceneModel.optimization_loop()`가 RGB/SSIM/depth loss와 Gaussian parameter, shared rig
   pose를 함께 최적화한다.

## 5. Incremental dataflow

bootstrap 이후 각 timestep은 다음 순서로 들어간다.

1. 모든 view의 detector output을 만든다.
2. view별로 `SceneModel.get_prev_keyframes(..., desc_kpts=view_desc)`를 호출한다.
3. 이전 keyframe의 3D point를 한 번씩 refresh한다.
4. `PoseInitializer.initialize_incremental_rig()`가 view별 2D-3D match를 만든다.
5. `rig/rig_pnp.py::rig_pnp_per_view()`가 view별 PnP pose candidate를 rig pose로 lift한다.
6. `rig/se3_utils.py::se3_robust_mean()`이 SE(3) robust mean으로 하나의 rig pose를 만든다.
7. 선택적으로 1-timestep `MiniBARig` refinement가 reprojection residual을 줄인다.
8. `SceneModel.append_rig_pose()`로 새 timestep pose slot을 추가한다.
9. N개 keyframe을 만들고, 각 view에서 Gaussian spawn 후 photometric optimization을 수행한다.

## 6. Depth Anything V2의 현재 역할

현재 rig 경로에서 `Depth Anything V2`는 bootstrap pose를 직접 만들지 않는다. 제거된
`--rig_mono_seed` 경로는 fork-local 실험이었고 upstream OTF 기능이 아니므로 convention에서
제외했다.

현재 남아 있는 역할은 두 가지다.

1. `scene/keyframe.py`의 `mono_idepth`, `mono_depth_conf`로부터 depth loss를 구성한다.
2. `scene/scene_model.py::add_new_gaussians`에서 `keyframe.align_depth()`와 guided MVS
   sampling prior에 간접적으로 쓰인다.

안전한 논문 표현:

> DA-V2는 metric depth sensor가 아니라 perspective crop별 ordinal/local depth prior로 쓰이며,
> rig bootstrap의 절대 깊이를 해결하는 핵심 장치로 주장하지 않는다.

`--depth_loss_weight_init 0` ablation은 이 주장을 검증하는 필수 실험이다.

## 7. 그대로 포팅하면 안 되는 upstream CLI

이 섹션의 옵션은 문서상 금지가 아니라 실행 금지다. `args.py`는 `--use_rig`와 아래
옵션이 함께 들어오면 training/dataloader/GPU 초기화 전에 즉시 실패해야 한다. 조용히
무시하는 것도 금지다. 조용한 무시는 실험 manifest와 실제 실행 조건이 달라지는 가장
위험한 실패 모드다.

### `--enable_reboot`

upstream non-rig reboot는 single-camera incremental tracking이 실패했을 때 일부 keyframe
window를 다시 잡는 기능이다. rig mode에 그대로 붙이면 timestep 내부 view 일부만 남는
불완전 rig, shared rig optimizer slot 불일치, same-timestep leakage가 생긴다. 포팅하려면
`reboot_timestep_window`로 새로 설계해야 하며, 지금은 포팅 금지다.

### `--test_hold`

upstream `--test_hold`는 매 N번째 image를 test로 태그한다. rig에서는 image가 아니라
timestep-view 구조이므로 그대로 쓰면 한 timestep 안의 일부 view가 train/test로 섞여
shared pose leakage가 생긴다. rig에서는 `--test_hold` 대신
`--rig_holdout_view <non_ref_view>`를 쓴다.

### `--use_colmap_poses`

upstream `--use_colmap_poses`는 image별 COLMAP pose를 직접 넣는 기능이다. rig에서 그대로
쓰면 N개 view가 독립 pose를 갖게 되어 zero-baseline shared-pose constraint가 깨진다.
허용되는 형태는 timestep별 rig-center trajectory를 import하고, 각 view pose는 항상
`relative_Rt[view] @ rig_pose[t]`로 유도하는 방식뿐이다.

## 8. 결과 디렉터리 규칙

claim-grade 재실행을 시작하기 전에는 `results/`를 직접 삭제하지 않는다. 기존 결과에는
중단 run, smoke test, 유효 ablation이 섞여 있어도 provenance가 남아야 한다. 새 전체
재실행은 아래 둘 중 하나로 시작한다.

1. 기존 `results/`를 timestamp archive로 이동한다.

   ```bash
   mv results "results_archive_$(date +%Y%m%d_%H%M%S)"
   mkdir -p results
   ```

2. 또는 `results/seed0_full_<date>/...`처럼 새 namespace를 만든다.

논문 표에 들어가는 run은 반드시 command, seed, scene, holdout view, git commit을 함께
기록한다. 중단된 run은 숫자를 재사용하지 않고 `aborted`로만 남긴다.

## 9. 문서/코드 주석 규칙

- view 수는 `9`, `12`로 쓰지 말고 `N` 또는 `len(rig.view_names)`로 쓴다.
- `keyframe`이라고 할 때는 반드시 `(timestep, view)`인지, ref-view keyframe인지 구분한다.
- `pose`라고 할 때는 `view pose`와 `shared rig pose`를 구분한다.
- pose accuracy는 train-view render metric과 섞지 않는다.
- `EQR->pinhole`은 novelty가 아니라 compatibility shim 또는 controlled stress probe로만 쓴다.

## 10. 외부 기준

- COLMAP rig/panorama docs: https://colmap.github.io/rigs.html
- COLMAP key concepts: https://colmap.github.io/concepts.html
- On-the-Fly NVS: https://arxiv.org/abs/2506.05558
- OmniGS native EQR 3DGS: https://arxiv.org/abs/2404.03202
- 360MonoDepth tangent-image split: https://openaccess.thecvf.com/content/CVPR2022/papers/Rey-Area_360MonoDepth_High-Resolution_360deg_Monocular_Depth_Estimation_CVPR_2022_paper.pdf
