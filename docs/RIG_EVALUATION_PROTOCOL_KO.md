# OTF Rig Evaluation Protocol

이 문서는 rig 실험에서 pose metric, train-view render metric, holdout-view render metric을
분리해 보고하기 위한 실행 규칙이다.

## 1. 절대 원칙: pose와 NVS metric을 섞지 않는다

Rig OTF의 주장은 두 축이다.

1. pose-coupled reconstruction이 low-parallax에서 등록을 유지하는가
2. 등록된 pose와 Gaussian으로 held-out view를 얼마나 잘 렌더링하는가

첫 번째는 OB3D GT center ATE/RRA/RTA로 평가한다. 두 번째는 PSNR/SSIM/LPIPS로 평가한다.
train-view PSNR은 optimization self-render라서 generalization metric이 아니다.

## 2. `--rig_holdout_view`

Canonical holdout command:

```bash
python train.py -s /home/kaprub22/otfrig/ob3d_rig/classroom_100 \
  --use_rig --rig_config /home/kaprub22/otfrig/rig12_panosfm.json \
  --ref_view E+0_A000 --rig_holdout_view E+0_A045 \
  --init_focal 200 --fix_focal --downsampling 1 \
  --num_iterations 270 --seed 0 --viewer_mode none \
  -m results/classroom_holdout_Ep0_A045_seed0
```

동작:

- `RigImageDataset`가 해당 view의 `info["is_test"]`를 `True`로 표시한다.
- `SceneModel.add_new_gaussians()`는 test keyframe에서 Gaussian을 spawn하지 않는다.
- `SceneModel.get_prev_keyframes()`는 holdout/test keyframe을 PnP/MVS/triangulation partner에서
  제외한다.
- post-hoc render eval은 모든 keyframe을 렌더한 뒤 `is_test=True`와 train views를 분리한다.

금지:

- `--rig_holdout_view`를 ref view로 지정하지 않는다.
- `--test_hold`와 섞어 쓰지 않는다.
- holdout view 이미지를 Gaussian spawn 또는 previous-keyframe pool에 넣지 않는다.

## 3. Train-view vs holdout-view metric

현재 post-hoc eval은 `results/<run>/render_eval/metrics.json`에 `summary`와 `per_frame`을
저장한다. `summary`는 모든 frame 평균이므로 논문 표에는 그대로 쓰지 않는다.
`per_frame[*].is_test`로 나눠 다음 세 줄을 보고한다.

| split | 의미 | 논문 표 사용 |
| --- | --- | --- |
| `train_views` | Gaussian optimization에 들어간 view | 진단용 |
| `holdout_view` | Gaussian spawn/optimization에서 제외된 view | NVS 주 metric |
| `all_views` | train+holdout 혼합 | 참고용만 |

Metric 정의:

$$
\mathrm{PSNR}(I,\hat I)= -10\log_{10}\left(\mathrm{MSE}(I,\hat I)\right)
$$

SSIM/LPIPS는 코드의 `fused_ssim`과 `lpips` VGG backend를 따른다.

간단 파서:

```bash
python - <<'PY'
import json, math, statistics as st
from pathlib import Path

path = Path("results/classroom_holdout_Ep0_A045_seed0/render_eval/metrics.json")
d = json.loads(path.read_text())
for name, rows in {
    "holdout_view": [r for r in d["per_frame"] if r["is_test"]],
    "train_views": [r for r in d["per_frame"] if not r["is_test"]],
}.items():
    lp = [r["lpips"] for r in rows if not math.isnan(r["lpips"])]
    print(name, "n", len(rows),
          "psnr", st.mean(r["psnr"] for r in rows),
          "ssim", st.mean(r["ssim"] for r in rows),
          "lpips", st.mean(lp) if lp else float("nan"))
PY
```

## 4. OB3D GT center ATE

OB3D는 exact Blender camera parameters를 제공하므로 metric ATE를 반드시 meter로 보고한다.
`%span`은 scene-size-normalized 보조 지표일 뿐이다.

각 timestep의 estimated center는 N개 view center가 같아야 한다.

$$
\hat C_t = \frac{1}{N}\sum_v \hat C_{t,v}
$$

먼저 same-timestep spread를 검증한다.

$$
s_t = \max_v \left\|\hat C_{t,v} - \hat C_t\right\|_2
$$

Pass 기준:

- `max_t s_t < 1e-5 m`: shared rig pose invariant 정상
- `1e-5 m <= max_t s_t < 1e-4 m`: WARN
- `>= 1e-4 m`: FAIL, view pose가 독립 drift했을 가능성

Estimated centers $\hat C_t$와 GT centers $C_t$를 Sim(3)로 align한다.

$$
C_t \approx sR\hat C_t + t
$$

ATE는 align 이후 RMSE다.

$$
\mathrm{ATE}_{m}
= \sqrt{\frac{1}{T}\sum_t
\left\|C_t - (sR\hat C_t+t)\right\|_2^2}
$$

Scene span normalized value는 다음으로만 보조 보고한다.

$$
\mathrm{ATE}_{\%span}
= 100\cdot \frac{\mathrm{ATE}_{m}}
{\max_i C_i - \min_i C_i \text{ 의 bbox diagonal}}
$$

필수 보고 항목:

- registered timesteps / total timesteps
- same-timestep center spread max/mean
- `ATE_RMSE_m`, `ATE_mean_m`, `ATE_median_m`, `ATE_max_m`
- `ATE_RMSE_pct_span`
- Sim3 scale

## 5. OB3D-compatible reporting

OB3D paper는 3D reconstruction, NVS, camera pose estimation(CPE)을 evaluation protocol로
제공한다. CPE 표는 RRA, RTA, AUC@5, ATE를 사용하고, NVS는 train/test split에서
PSNR/SSIM/LPIPS를 사용한다. 따라서 본 rig 논문 표는 다음 구조가 가장 방어 가능하다.

| scene | trajectory | method | registered/total | ATE m | ATE %span | RRA deg | RTA deg | holdout PSNR | holdout SSIM | holdout LPIPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

주의:

- 현재 pinhole holdout PSNR은 OB3D native EQR PSNR과 직접 비교하지 않는다.
- OB3D NVS 리더보드에 맞추려면 rendered pinholes를 EQR로 restitch하거나 native-EQR render path를
  구현해야 한다.
- train-view PSNR은 appendix diagnostic으로만 둔다.

## 6. 실행 스니펫: ATE

현재 repo에 고정 evaluator script가 없으면 아래 스니펫으로 계산하고, 이후
`scripts/eval_ob3d_rig_ate.py`로 승격한다.

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np

run = Path("results/classroom_holdout_Ep0_A045_seed0")
gt_path = Path("/home/kaprub22/otfrig/ob3d_rig/classroom_100/gt_centers.json")
meta = json.loads((run / "metadata.json").read_text())
gt = json.loads(gt_path.read_text())

def center_from_w2c(T):
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3,:3], T[:3,3]
    return -R.T @ t

by_ts = {}
for k in meta["keyframes"]:
    info = k.get("info", {})
    ts = info.get("rig_ts")
    if ts is not None:
        by_ts.setdefault(int(ts), []).append(center_from_w2c(k["Rt"]))

ts = sorted(set(by_ts) & set(map(int, gt.keys())))
est = np.stack([np.mean(by_ts[i], axis=0) for i in ts])
ref = np.stack([np.asarray(gt[str(i)], dtype=np.float64) for i in ts])
spread = [max(np.linalg.norm(c - np.mean(c, axis=0), axis=1))
          for c in [np.stack(by_ts[i]) for i in ts]]

mu_e, mu_g = est.mean(0), ref.mean(0)
Xe, Xg = est - mu_e, ref - mu_g
cov = (Xg.T @ Xe) / len(ts)
U, S, Vt = np.linalg.svd(cov)
D = np.eye(3)
if np.linalg.det(U @ Vt) < 0:
    D[-1, -1] = -1
R = U @ D @ Vt
scale = np.trace(np.diag(S) @ D) / (np.sum(Xe * Xe) / len(ts))
t = mu_g - scale * R @ mu_e
aligned = (scale * (R @ est.T)).T + t
err = np.linalg.norm(ref - aligned, axis=1)
span = np.linalg.norm(ref.max(0) - ref.min(0))

print("num_ts", len(ts))
print("same_ts_spread_max", float(max(spread)))
print("same_ts_spread_mean", float(np.mean(spread)))
print("ATE_RMSE_m", float(np.sqrt(np.mean(err**2))))
print("ATE_mean_m", float(np.mean(err)))
print("ATE_median_m", float(np.median(err)))
print("ATE_max_m", float(np.max(err)))
print("GT_bbox_diag_m", float(span))
print("ATE_RMSE_pct_span", float(100*np.sqrt(np.mean(err**2))/span))
print("Sim3_scale_est_to_gt", float(scale))
PY
```

## 7. 외부 기준

- OB3D dataset/protocol: https://arxiv.org/html/2505.20126v1
- OB3D GitHub evaluation code: https://github.com/gsisaoki/Omnidirectional_Blender_3D_Dataset
- TUM RGB-D ATE/RPE convention: https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf
- Visual odometry trajectory evaluation tutorial: https://rpg.ifi.uzh.ch/docs/IROS18_Zhang.pdf
