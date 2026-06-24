# OTF Rig Ablation Plan

이 문서는 zero-baseline rig에서 OTF가 버티는 원인이 무엇인지 분리하기 위한 최소 ablation
matrix다. 모든 run은 같은 scene, 같은 split, 같은 seed, 같은 active-window policy로 실행한다.

## 1. 공통 실행 규칙

공통 command prefix:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate onthefly_nvs
cd /home/kaprub22/otfrig/on-the-fly-nvs

COMMON="-s /home/kaprub22/otfrig/pinhole_rig/ob3d_rig/classroom_100 \
  --use_rig --rig_config /home/kaprub22/otfrig/tools/configs/rig12_panosfm.json \
  --ref_view E+0_A000 --init_focal 200 --fix_focal \
  --downsampling 1 --num_keyframes_miniba_bootstrap 8 \
  --num_iterations 270 --seed 0 --viewer_mode none \
  --max_active_keyframes 60"
```

가능하면 모든 ablation에 같은 holdout view를 추가한다.

```bash
COMMON="$COMMON --rig_holdout_view E+0_A090"
```

기록해야 하는 metric:

- registration: registered timestep 수
- pose: OB3D GT center ATE meter, ATE %span, same-ts spread
- render: train-view PSNR/SSIM/LPIPS, holdout-view PSNR/SSIM/LPIPS
- efficiency: wall-clock, per-phase runtime(`--display_runtimes`), peak GPU memory
- model state: final Gaussian count, anchor count, crash 여부

## 2. Ablation matrix

| row | command suffix | 고립하려는 레버 | 성공 해석 |
| --- | --- | --- | --- |
| default | 없음 | 전체 rig OTF 경로 | 기준선 |
| frozen pose | `--freeze_rig_poses` | photometric BA가 shared rig pose를 개선하는가 | ATE/holdout이 악화되면 pose-coupled photometric refinement가 causal |
| no depth loss | `--depth_loss_weight_init 0` | DA-V2 ordinal depth regularization의 기여 | 변화가 작으면 depth prior가 핵심이 아님. 크게 악화되면 depth-primed baseline 필요 |
| no MVS spawn | `--init_proba_scaler 0` | Laplacian sampled guided-MVS Gaussian spawn의 기여 | pose는 유지되고 recon만 악화되면 spawn은 품질 레버. pose까지 무너지면 geometry feedback 레버 |
| active window control | `--max_active_keyframes 60` 고정 | memory/offload policy 차이 제거 | N이 다른 rig sweep에서도 timestep-equivalent context를 맞춤 |

## 3. Copy-runnable commands

### 3.1 Default

```bash
python train.py $COMMON \
  -m results/abl_classroom_default_seed0
```

### 3.2 `--freeze_rig_poses`

```bash
python train.py $COMMON --freeze_rig_poses \
  -m results/abl_classroom_freeze_rig_poses_seed0
```

격리 대상:

- `scene/scene_model.py::optimization_step`에서 `rig_optimizer.step()`을 건너뛰게 한다.
- bootstrap/incremental PnP/MiniBA pose는 그대로 쓰지만, photometric optimization이 pose를
  보정하지 못한다.

해석:

- `ATE_freeze >> ATE_default`이고 holdout도 악화:
  - shared rig-pose photometric refinement가 causal.
- `ATE_freeze ~= ATE_default`:
  - OTF 우위가 photometric BA가 아니라 initializer/PnP/temporal matching에서 왔을 가능성이 크다.
- `freeze`가 더 좋음:
  - photometric pose update가 drift를 만든다. pose LR, depth loss, spawn schedule을 재검토한다.

### 3.3 `--depth_loss_weight_init 0`

```bash
python train.py $COMMON --depth_loss_weight_init 0 \
  -m results/abl_classroom_no_depth_loss_seed0
```

격리 대상:

- `scene/keyframe.py`의 depth loss weight를 0으로 시작한다.
- DA-V2 model computation 자체와 guided-MVS 관련 path를 완전히 제거하는 실험은 아니다.

해석:

- pose/recon 변화가 작음:
  - DA-V2는 현재 rig novelty의 핵심이 아니다.
- pose/recon이 크게 악화:
  - depth prior가 중요한 causal lever다.
  - vanilla COLMAP만 비교하면 불공정하고 MP-SfM/depth-primed COLMAP/NoPe-NeRF류 comparator가
    필요하다.

### 3.4 `--init_proba_scaler 0`

```bash
python train.py $COMMON --init_proba_scaler 0 \
  -m results/abl_classroom_no_mvs_spawn_seed0
```

격리 대상:

- `scene/scene_model.py::add_new_gaussians`의 Laplacian probability sampling을 0으로 만든다.
- sampled UV가 없어 guided-MVS spawn branch는 skip된다.
- triangulated match points spawn은 남는다.

해석:

- pose ATE는 비슷하고 holdout PSNR만 하락:
  - guided-MVS spawn은 reconstruction density/quality 레버다.
- pose ATE도 하락:
  - Gaussian spawn이 photometric pose refinement의 basin을 만든다.
- Gaussian 수가 과도히 줄어 crash는 사라지지만 품질이 무너짐:
  - crash fix를 scaler clamp로 해결하면 안 되고, non-finite/extreme spawn row-drop이 맞다.

### 3.5 `--max_active_keyframes` consistency

`scene/scene_model.py::add_keyframe`은 rig mode에서 다음처럼 active cap을 view 수만큼 키운다.

$$
K_{\mathrm{active}} = K_{\mathrm{flag}}\cdot N_{\mathrm{views}}
$$

따라서 `--max_active_keyframes 60`은 "60 timestep 상당의 context"라는 뜻으로 해석한다.

규칙:

- 모든 ablation에서 같은 값을 쓴다.
- view-count/FOV sweep에서는 같은 timestep horizon을 뜻하도록 같은 flag 값을 쓴다.
- OOM이 나면 값을 낮추되 모든 row를 같은 값으로 다시 돌린다.

## 4. Mechanism decision rule

### Claim A: online photometric shared-rig-pose refinement

조건:

- default가 frozen pose보다 ATE와 holdout metric에서 명확히 좋다.
- `--depth_loss_weight_init 0`에서도 default 대비 pose가 크게 무너지지 않는다.
- `--init_proba_scaler 0`은 주로 recon 품질을 낮추고 pose는 유지된다.

허용 문장:

> Zero-baseline virtual rig에서 OTF의 핵심 이득은 timestep 간 temporal parallax로 얻은
> 초기 pose를, N-view shared rig constraint 아래 photometric Gaussian optimization으로
> 보정하는 데 있다.

### Claim B: depth-prior-driven reconstruction

조건:

- `--depth_loss_weight_init 0`에서 ATE/holdout이 크게 악화된다.
- frozen pose보다 depth-off의 영향이 더 크다.

필수 후속:

- vanilla COLMAP은 불충분한 comparator다.
- MP-SfM, depth-primed COLMAP, NoPe-NeRF/pose-free depth-regularized Gaussian 계열을 추가한다.

### Claim C: spawn/reconstruction coupling

조건:

- `--init_proba_scaler 0`에서 pose도 함께 악화된다.

허용 문장:

> Rig OTF의 pose robustness는 sparse PnP만의 결과가 아니라, incremental Gaussian spawn이 만든
> photometric surface와 pose update가 서로 안정화되는 coupling 현상이다.

## 5. Minimum scene set

| split | scenes | 이유 |
| --- | --- | --- |
| ego indoor | classroom, restroom | low-parallax지만 texture/geometry 안정 |
| ego outdoor fail set | sun-temple, emerald-square, lone-monk | pinhole-rig COLMAP failure가 있던 핵심 stress |
| non-ego outdoor | sun-temple_ne100, emerald-square_ne100, lone-monk_ne100 | parallax 증가 시 causal trend 확인 |

모든 scene은 seed 0/1/2를 돌린다. single seed 표는 appendix diagnostic으로만 둔다.

## 6. 실행 순서

1. `classroom_100` default/freeze/depth0/proba0를 seed 0으로 돌려 command와 parser를 검증한다.
2. 같은 4 rows를 ego outdoor fail set에 돌린다.
3. emerald non-ego crash diagnostic을 별도 crash protocol로 처리한다.
4. 결과가 안정적이면 seed 1/2를 추가한다.
5. native-EQR SfM/GLOMAP baseline과 같은 ATE evaluator로 pose 표를 만든다.

## 7. Kill criteria

아래 중 하나면 method-style claim은 중단한다.

- frozen pose와 default가 pose/recon에서 동률이다.
- depth-off가 무너지고 depth-primed COLMAP/MP-SfM이 같은 결과를 재현한다.
- native-EQR SfM이 accuracy와 runtime에서 모두 우위이며, online latency claim도 남지 않는다.
- holdout-view metric이 train-view보다 크게 낮아져 reconstruction generalization이 성립하지 않는다.

## 8. 외부 기준

- On-the-Fly NVS: https://arxiv.org/abs/2506.05558
- ORB-SLAM low-parallax/keyframe network motivation: https://ar5iv.labs.arxiv.org/html/1502.00956
- GLOMAP baseline: https://arxiv.org/abs/2407.20219
- Depth Anything V2: https://arxiv.org/abs/2406.09414
