# OB3D Splits

Source dataset: Kaggle `shintacs/ob3d-dataset`.

Downloaded files:

```text
OB3D/<scene>/Egocentric/train.txt
OB3D/<scene>/Egocentric/test.txt
```

The files contain original OB3D frame indices, not prepared rig filenames. In
rig mode, use both:

```bash
--rig_train_timesteps_file splits/ob3d/egocentric/<scene>/train.txt
--rig_test_timesteps_file  splits/ob3d/egocentric/<scene>/test.txt
```

This expands the official EQR split onto the virtual rig:

```text
train EQR timestep t -> all N rig views at t are train
test EQR timestep t  -> all N rig views at t are test
other timestep t     -> all N rig views at t are tracking-only
```

For OB3D Egocentric this gives 25 train timesteps, 25 test timesteps, and 50
tracking-only timesteps per scene. `--rig_holdout_view` remains a diagnostic for
unseen direction at already-seen timesteps and should not be used as the main
OB3D NVS metric.
