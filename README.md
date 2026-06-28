# On-the-Fly NVS Rig Fork

GraphDeco/Inria의 **On-the-Fly NVS**를 기반으로,
단일 360도 EQR 이미지를 여러 개의 virtual pinhole view로 나누어
rotation-only zero-baseline rig로 처리함.

원본 프로젝트:

- Paper / project: <https://repo-sam.inria.fr/nerphys/on-the-fly-nvs/>
- Upstream code: <https://github.com/graphdeco-inria/on-the-fly-nvs>

### 목표

> 360 panorama는 한 번의 physical capture에서 넓은 angular coverage를 주지만,
> same-timestep stereo baseline은 0임. 이 조건에서 online pose-coupled
> Gaussian reconstruction이 temporal parallax, shared-pose refinement,
> depth prior, densification에 얼마나 의존하는지 평가함.

## 현재 환경

개발/실험 기준 환경은 다음과 같음.

| 항목 | 값 |
| --- | --- |
| OS | WSL2 / Ubuntu 22.04 LTS |
| GPU | NVIDIA CUDA GPU (RTX 4060Ti 16GB) |
| Conda env | `onthefly_nvs` |

## 설치

처음 clone한 환경에서는 submodule과 CUDA extension이 필요함.
zip snapshot에는 큰 submodule 산출물이 빠질 수 있으므로, 학습 환경에서는
submodule을 다시 준비해야 함.

```bash
git submodule update --init --recursive

conda create -n onthefly_nvs python=3.12 -y
conda activate onthefly_nvs

pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
pip install cupy-cuda12x
pip install -r requirements.txt
```

## 데이터 구조

이 repo의 rig loader는 이미 EQR에서 추출된 pinhole rig dataset을 읽음.
원본 EQR, 변환된 pinhole rig dataset, 외부 baseline 결과는 repo 밖의
작업 폴더에서 관리하는 것을 권장함.

입력 scene은 다음 형태를 기대함.

```text
<scene>/
  extraction_meta.json        # optional, init_focal 등을 자동 로드
  images/
    E+0_A000/
      frame_000000.png
      frame_000001.png
      ...
    E+0_A090/
      frame_000000.png
      frame_000001.png
      ...
    ...
  masks/                      # optional
    E+0_A000/
      frame_000000.png
      ...
  gt_centers.json             # optional, OB3D ATE 평가용
```

## EQR to pinhole 예시

EQR 원본을 12-view pinhole rig로 자르는 예시 파일은 repo 안에 같이 둠.
GitHub에서 구조를 바로 볼 수 있게 아래 3개를 snapshot으로 포함함.

| 파일 | 용도 |
| --- | --- |
| `examples/panoramic_rig/eqr_to_pinhole.py` | EQR image/video를 `images/<view>/frame_xxxxxx.png` 구조로 추출 |
| `examples/panoramic_rig/rig12_panosfm.json` | OTF-Rig와 변환기용 12-view Blender-style rig |
| `examples/panoramic_rig/colmap_rig_panosfm.json` | COLMAP `rig_configurator`용 panorama-SfM style rig |

단일 EQR image 변환 예시는 다음과 같음.

```bash
export OTFRIG_ROOT=/path/to/otfrig

python examples/panoramic_rig/eqr_to_pinhole.py \
  --eqr_image /path/to/00000_rgb.png \
  --out_dir "$OTFRIG_ROOT/pinhole_rig/example_scene" \
  --rig_config examples/panoramic_rig/rig12_panosfm.json \
  --ref_view E+0_A000 \
  --fov 90 \
  --size 960 \
  --write_masks \
  --device cuda
```

video에서 stride를 두고 추출하는 예시는 다음과 같음.

```bash
export OTFRIG_ROOT=/path/to/otfrig

python examples/panoramic_rig/eqr_to_pinhole.py \
  --video /path/to/input_eqr.mp4 \
  --out_dir "$OTFRIG_ROOT/pinhole_rig/example_video" \
  --rig_config examples/panoramic_rig/rig12_panosfm.json \
  --ref_view E+0_A000 \
  --fov 90 \
  --size 960 \
  --stride 2 \
  --frame_limit 100 \
  --write_masks \
  --device cuda
```

이 변환기는 `extraction_meta.json`도 같이 저장함.
OTF 학습에서는 이 metadata에서 focal을 자동으로 읽을 수 있음.
예를 들어 FOV 90도, 960x960이면 \(f=480\)이고,
400x400 prepared OB3D rig라면 \(f=200\)임.

12-view layout은 다음 규약을 사용함.

| ring | views |
| --- | --- |
| `E+35` | `A045`, `A135`, `A225`, `A315` |
| `E+0` | `A000`, `A090`, `A180`, `A270` |
| `E-35` | `A000`, `A090`, `A180`, `A270` |

`rig12_panosfm.json`의 형태는 다음과 같음.
중요한 점은 모든 view의 `location`이 `[0, 0, 0]`이라는 것임.
최종 OTF loader에서도 relative translation은 0으로 강제됨.

```json
[
  {
    "name": "Pano",
    "cameras": [
      {
        "name": "E+35_A045",
        "location": [0.0, 0.0, 0.0],
        "rotation": [0.4266000929, 0.8194911539, -0.3394443502, -0.1767035442]
      },
      {
        "name": "E+0_A000",
        "location": [0.0, 0.0, 0.0],
        "rotation": [-0.5, -0.5, 0.5, 0.5]
      },
      {
        "name": "E-35_A000",
        "location": [0.0, 0.0, 0.0],
        "rotation": [0.6272113751, 0.3265055756, -0.3265055756, -0.6272113751]
      }
    ]
  }
]
```

위 snippet은 전체 12개 중 일부만 보인 것임.
전체 view list는 `examples/panoramic_rig/rig12_panosfm.json`을 기준으로 함.

COLMAP baseline은 같은 image folder를 쓰지만 config 형식이 다름.
`colmap_rig_panosfm.json`은 `image_prefix`, `PINHOLE` intrinsics,
`cam_from_rig_rotation`, `cam_from_rig_translation`을 담음.
여기서도 translation은 모두 0이어야 함.

```json
{
  "image_prefix": "E+35_A045/",
  "camera_model_name": "PINHOLE",
  "camera_params": [200.0, 200.0, 200.0, 200.0],
  "cam_from_rig_translation": [0.0, 0.0, 0.0]
}
```

## 용어 정리

### 입력/데이터 단위

| 이름 | 의미 | 사용처 |
| --- | --- | --- |
| EQR | 360도 이미지를 위도/경도 좌표로 펼친 이미지임 | 원본 panorama 입력 |
| virtual pinhole view | EQR에서 특정 방향만 잘라 만든 일반 pinhole camera 이미지임 | OTF 입력 view |
| timestep | 하나의 physical capture 시점임. EQR 한 장에 해당함 | train/test split 단위 |
| packet | 같은 timestep에서 나온 N개 virtual view 묶음임 | rig streaming 단위 |
| `source_ts` | 파일명에서 파싱한 원본 EQR frame index | split, ATE, report |
| `stream_idx` | loader가 정렬한 online 입력 순서 | streaming loop |
| `ts_idx` / rig slot | 등록된 shared rig pose parameter index | `rig_R6D`, `rig_t` indexing |
| keyframe | OTF 내부에서 pose, feature, depth, image cache를 들고 있는 view-level 상태임 | `scene_model.keyframes` |

즉 `frame_000010.png`는 `source_ts=10`이어야 함.
sorted rank를 claim split이나 ATE에 쓰면 안 됨.

### Rig geometry

| 이름 | 의미 | 코드/수식에서의 역할 |
| --- | --- | --- |
| rig | 여러 camera/view를 하나의 묶음으로 보는 구조임 | 한 timestep의 N개 view를 같이 처리함 |
| zero-baseline rig | 같은 timestep의 모든 view가 같은 optical center를 공유하는 rig임 | same-timestep depth를 만들 수 없음 |
| ref view | packet 안에서 기준으로 삼는 view임 | 보통 `E+0_A000` |
| relative pose | ref view 기준으로 각 view가 어느 방향을 보는지 나타내는 고정 회전임 | `relative_Rt[view]` |
| shared rig pose | 한 timestep 전체가 공유하는 하나의 SE(3) pose임 | `rig_R6D[ts_idx]`, `rig_t[ts_idx]` |
| view pose | fixed relative pose와 shared rig pose를 합성한 각 view pose임 | `view_w2c = relative_Rt @ rig_w2c` |
| optical center | camera ray들이 출발하는 3D 위치임 | zero-baseline 여부 확인 |
| baseline | 두 optical center 사이 거리임 | depth triangulation 가능성 결정 |
| parallax | camera 위치 변화 때문에 같은 물체가 이미지에서 다르게 보이는 정도임 | depth를 만드는 핵심 신호 |
| temporal parallax | 서로 다른 timestep 사이의 이동으로 생기는 parallax임 | 이 rig의 주된 depth source |

### Reconstruction

| 이름 | 의미 | 왜 중요한가 |
| --- | --- | --- |
| triangulation | 서로 다른 위치에서 본 feature match로 3D point를 구하는 과정임 | 같은 timestep에서는 금지해야 함 |
| same-timestep leakage | 같은 timestep view가 triangulation/MVS partner로 섞이는 버그임 | zero-baseline claim을 깨는 문제임 |
| depth prior | 이미지에서 예상 깊이 순서나 형태를 주는 보조 신호임 | texture/low-parallax 구간에서 도움 가능 |
| guided MVS | 기존 3D/feature/depth 신호를 이용해 새 point depth를 찾는 과정임 | Gaussian spawn 후보 생성에 관여함 |
| Gaussian spawn | 새 3D Gaussian을 scene에 추가하는 단계임 | reconstruction density를 늘림 |
| densification | 부족한 영역을 채우기 위해 Gaussian을 늘리는 과정임 | NVS 품질에 직접 영향 |
| photometric optimization | render와 target image 차이를 줄이도록 pose/Gaussian을 최적화하는 과정임 | shared rig pose refinement의 핵심 |

### Evaluation

| 이름 | 의미 | 주의점 |
| --- | --- | --- |
| pose-assisted online eval | test/tracking frame도 pose tracking stream에는 남기는 평가 방식임 | Gaussian 학습/metric과 분리해야 함 |
| train timestep | Gaussian spawn과 photometric loss에 쓰는 timestep임 | `rig_eval_split == "train"` |
| test timestep | 최종 NVS metric에 쓰는 held-out timestep임 | `rig_eval_split == "test"` |
| tracking-only timestep | pose tracking에는 쓰지만 train/test metric에는 쓰지 않는 timestep임 | diagnostic 성격임 |
| claim metric | 논문 표에 올릴 수 있는 metric임 | test split만 사용해야 함 |
| diagnostic metric | 디버깅용 metric임 | headline number로 쓰면 안 됨 |
| ATE | 추정 trajectory와 GT trajectory의 위치 오차임 | Sim(3) 정렬 뒤 계산 |
| registration recall | 기대 timestep 중 pose 등록에 성공한 비율임 | missing timestep을 숨기면 안 됨 |
| center spread | 같은 timestep view center들이 얼마나 벌어졌는지임 | zero-baseline이면 거의 0이어야 함 |

## Rig 수식

각 EQR timestep \(t\)에서 \(N\)개의 virtual pinhole view
\(I_{t,v}\)를 사용함. 모든 view는 같은 optical center를 공유하고,
view마다 고정된 relative rotation만 다름.

world-to-rig pose:

$$
T_t^{rig} =
\begin{bmatrix}
R_t & t_t \\
0 & 1
\end{bmatrix}
$$

view별 fixed relative transform:

$$
T_v^{rel} =
\begin{bmatrix}
R_v^{rel} & 0 \\
0 & 1
\end{bmatrix}
$$

view pose는 다음처럼 합성함.

$$
T_{t,v}^{cam} = T_v^{rel} T_t^{rig}
$$

따라서

$$
R_{t,v} = R_v^{rel} R_t,\quad
t_{t,v} = R_v^{rel} t_t
$$

camera center는

$$
C_{t,v} = -R_{t,v}^{T} t_{t,v}
= -(R_v^{rel}R_t)^T(R_v^{rel}t_t)
= -R_t^T t_t
$$

즉 같은 timestep의 모든 view center가 동일함.
same-timestep view pair는 baseline이 0이라 triangulation depth를 만들 수 없음.
depth-bearing geometry는 temporal parallax, depth prior, Gaussian densification에서 나옴.

## 평가 정책

이 fork의 OB3D 논문용 평가는 **timestep packet split**을 사용함.
한 EQR timestep에 속한 모든 virtual view가 train 또는 test packet으로 같이 움직임.

중요한 구분:

| 항목 | 정책 |
| --- | --- |
| pose tracking | online pose-assisted 평가임. bootstrap/tracking은 등록된 frame을 pose 추정에 사용할 수 있음 |
| Gaussian spawn | `rig_eval_split == "train"`만 사용 |
| photometric optimization | `rig_eval_split == "train"`만 사용 |
| triangulation/MVS partner | train + cross-`source_ts` partner만 허용 |
| test NVS metric | `rig_eval_split == "test"`만 사용 |
| `rig_holdout_view` | diagnostic 전용. OB3D 논문용 metric으로 쓰지 않음 |

"test frame을 절대 사용하지 않음"이라고 쓰면 틀림.
정확한 표현은 다음임.

> test timestep은 online pose tracking stream에는 남아 있지만,
> Gaussian spawn, radiance optimization, claim metric에서는 분리됨.

## 실행 예시

### 1. Preflight

```bash
export OTFRIG_ROOT=/path/to/otfrig

python scripts/preflight_ob3d_rig_scene.py \
  --scene "$OTFRIG_ROOT/pinhole_rig/ob3d_rig/classroom_100" \
  --rig-config examples/panoramic_rig/rig12_panosfm.json \
  --ref-view E+0_A000 \
  --train-timesteps-file splits/ob3d/egocentric/classroom/train.txt \
  --test-timesteps-file splits/ob3d/egocentric/classroom/test.txt
```

### 2. Train

`extraction_meta.json`에 focal이 있으면 `--init_focal`은 자동 로드됨.
없으면 400x400, 90도 FOV 기준으로 `--init_focal 200`을 명시함.

```bash
python train.py -s "$OTFRIG_ROOT/pinhole_rig/ob3d_rig/classroom_100" \
  --use_rig \
  --rig_config examples/panoramic_rig/rig12_panosfm.json \
  --ref_view E+0_A000 \
  --fix_focal \
  --downsampling 1 \
  --num_iterations 270 \
  --seed 0 \
  --viewer_mode none \
  --rig_train_timesteps_file splits/ob3d/egocentric/classroom/train.txt \
  --rig_test_timesteps_file splits/ob3d/egocentric/classroom/test.txt \
  -m results/example/classroom_seed0
```

### 3. ATE

```bash
python scripts/eval_ob3d_rig_ate.py \
  --run results/example/classroom_seed0 \
  --gt-centers "$OTFRIG_ROOT/pinhole_rig/ob3d_rig/classroom_100/gt_centers.json" \
  --fail-on-missing
```

### 4. Protocol artifact check

```bash
python scripts/check_rig_protocol_artifacts.py \
  --run results/example/classroom_seed0 \
  --expected-num-views 12 \
  --fail-on-missing
```

통과 조건은 대략 다음임.

```text
missing_timesteps_all == []
missing_timesteps_test == []
views_per_timestep_min == 12
views_per_timestep_max == 12
same_ts_spread_max_m < 1e-6
triangulation_partner_count_same_ts == 0
triangulation_partner_count_test == 0
triangulation_partner_count_tracking == 0
spawn_count_test == 0
spawn_count_tracking == 0
```

## 주요 CLI

| 인자 | 의미 |
| --- | --- |
| `--use_rig` | rig-aware loader와 shared rig pose path 활성화 |
| `--rig_config` | virtual rig JSON. view별 `cam_from_rig_rotation` 사용 |
| `--ref_view` | timestep packet에서 기준 view. OB3D rig는 보통 `E+0_A000` |
| `--rig_train_timesteps_file` | train `source_ts` 목록 |
| `--rig_test_timesteps_file` | test `source_ts` 목록 |
| `--rig_holdout_view` | view holdout diagnostic. OB3D 논문용 metric 아님 |
| `--freeze_rig_poses` | photometric shared-pose update를 끄는 ablation |
| `--rig_min_success_views` | 한 timestep pose 승인에 필요한 성공 view 수. 기본 2 |
| `--rig_huber_trans` | per-view PnP candidate fusion의 translation robust threshold |
| `--rig_bootstrap_outlier_dist` | bootstrap sparse point distance prune threshold |
| `--masks_dir` | optional mask directory |
| `--init_focal`, `--init_fov`, `--fix_focal` | focal 초기화 및 고정 |

rig mode에서 금지되는 upstream option:

| 인자 | 이유 |
| --- | --- |
| `--enable_reboot` | single-camera reboot이며 rig packet state reset과 맞지 않음 |
| `--test_hold` | image-stride holdout이라 같은 timestep view leakage를 만들 수 있음 |
| `--use_colmap_poses` | per-image pose import라 shared rig pose 규약과 충돌함 |

## 코드 맵

| 파일 | 역할 |
| --- | --- |
| `dataloaders/rig_image_dataset.py` | N-view packet loader, `source_ts` parsing, timestep split, focal/mask loading |
| `rig/rig_loader.py` | rig JSON 로드, reference view 정렬, `rel_t = 0` 강제 |
| `rig/rig_pnp.py` | view별 PnP candidate를 shared rig pose로 합침 |
| `rig/triangulation_policy.py` | train + cross-timestep triangulation partner만 허용 |
| `poses/pose_initializer.py` | rig bootstrap / incremental pose initialization |
| `poses/mini_ba_rig.py` | shared rig pose MiniBA |
| `scene/keyframe.py` | `view_w2c = relative_Rt @ rig_w2c` 합성, keyframe state 관리 |
| `scene/scene_model.py` | shared rig pose parameter, Gaussian spawn, leakage audit, completeness metadata |
| `train.py` | timestep packet streaming, train/test/tracking split, post-hoc render eval |
| `scripts/eval_ob3d_rig_ate.py` | OB3D GT center 기반 Sim(3) ATE |
| `scripts/check_rig_protocol_artifacts.py` | rig protocol artifact sanity check |
| `scripts/smoke_rig_protocol_*.py` | split/triangulation/spawn/timestep/completeness smoke test |

## 출력 파일

학습 run의 핵심 산출물은 다음임.

| 파일 | 용도 |
| --- | --- |
| `metadata.json` | 저장된 keyframe, rig pose, policy, registered keyframe만 반영한 COLMAP export |
| `render_eval/metrics_claim_test.json` | 논문용 held-out timestep metric |
| `render_eval/metrics_diagnostic_all.json` | 전체 frame diagnostic. headline metric으로 쓰지 않음 |
| `render_eval/metrics_diagnostic_tracking.json` | tracking-only diagnostic |
| `render_eval/split_metrics.json` | split별 metric과 policy 요약 |
| `render_eval/rig_completeness.json` | expected/registered/failed/missing timestep |
| `render_eval/rig_leakage_audit.json` | same-ts/test/tracking leakage counter와 finite reject 통계 |

## Citation

이 fork는 원본 On-the-Fly NVS 위에 구현됨.
원본 코드를 사용하는 경우 아래 citation을 유지해야 함.

```bibtex
@article{meuleman2025onthefly,
  title={On-the-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images},
  author={Meuleman, Andreas and Shah, Ishaan and Lanvin, Alexandre and Kerbl, Bernhard and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={44},
  number={4},
  year={2025}
}
```
