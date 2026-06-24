# Rig OTF Documentation

이 폴더는 `feat/rig-core`의 zero-baseline virtual panoramic rig 실험 계약을 고정한다.

읽는 순서:

1. [RIG_CONVENTIONS_KO.md](RIG_CONVENTIONS_KO.md)
   - timestep/view/keyframe 단위
   - `rel @ rig`, `rel_t=0`, same-timestep exclusion
   - `enable_reboot`, `test_hold`, `use_colmap_poses`를 그대로 포팅하면 안 되는 이유
2. [RIG_EVALUATION_PROTOCOL_KO.md](RIG_EVALUATION_PROTOCOL_KO.md)
   - `--rig_train_timesteps_file` + `--rig_test_timesteps_file` as the OB3D-style claim-grade split
   - `--rig_holdout_view` as a diagnostic only
   - OB3D GT center ATE
   - train-view metric과 test-timestep metric 분리
   - 실행 스크립트:
     - `scripts/eval_ob3d_rig_ate.py`
     - `scripts/summarize_rig_render_eval.py`
3. [RIG_ABLATION_PLAN_KO.md](RIG_ABLATION_PLAN_KO.md)
   - default / `--freeze_rig_poses`
   - `--depth_loss_weight_init 0`
   - MVS spawn ablation은 `--init_proba_scaler 0`을 그대로 쓰지 않는다는 보류 규칙
   - `--max_active_keyframes` consistency
   - seed0 전체 OB3D default rerun 순서

논문 claim을 쓸 때의 최소 안전 문장:

> A zero-baseline virtual panoramic rig is a controlled low-parallax stress test
> for online pose-coupled Gaussian reconstruction.

이 폴더의 문서는 실행 결과 표가 아니라 protocol contract다. 실험 숫자는 run별
`results/<run>/`와 별도 manifest에 기록한다.
