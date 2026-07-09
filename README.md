# ContinAuth recreation on the Apple-Watch IMU dataset


============================================================================
CONTINAUTH A/B/C COMPARISON  (his Siamese-FCN + OCSVM on your dataset)
============================================================================
config                        ch   OCSVM EER  cosine EER    rank-1
----------------------------------------------------------------------------
A_his-feat_his-window          9      23.78%       8.87%    34.82%
B_his-feat_my-window           9      21.31%       8.79%    35.27%
C_my-28ch_my-window           28       4.23%       2.68%    50.89%
D_my-28ch+mag_my-window       31       3.61%       2.23%    57.14%
============================================================================

A **self-contained** recreation of [dynobo/ContinAuth](https://github.com/dynobo/ContinAuth)
(Buech, 2019 — *"Continuous Authentication via Smartphone Inertial Sensors"*),
adapted to test **his model on your dataset**. It does **not** import your
`apw_network.py` / `verification.py`, so you can run and compare it independently
of your own work.

## What his model is

The ContinAuth final model (`VALID-FCN-ROBUST`) is a two-stage pipeline:

1. **Siamese fully-convolutional network (FCN)** trained with **contrastive loss**
   on same/different-subject window pairs. The shared base network is:

   ```
   Conv1D(32, k=8, same) → BN → ReLU → Dropout(0.1)
   Conv1D(64, k=5, same) → BN → ReLU → Dropout(0.1)
   Conv1D(32, k=3, same) → BN → ReLU
   GlobalAveragePooling1D
   Dense(32) → L2-normalise    # 32-d deep feature (embedding)
   ```
   Trained with **contrastive loss** (margin = 1, Adam lr 1e-3) over
   identity-balanced P×K batches. See *Notes* for the two deliberate upgrades
   over his exact recipe (L2-normalised embedding instead of sigmoid; val-tuned
   OCSVM) that keep the baseline fair on this data.

2. **Per-user one-class SVM** (RBF) fit on the owner's deep features;
   verification score = the SVM decision function. `nu`/`gamma` are **tuned on
   the val split** (his H-MOG `nu=0.165, gamma=8.296` do not transfer). Reported
   as **Equal Error Rate (EER)** averaged over owners (`roc_curve` + `brentq`).

His original recipe used **25 Hz**, **5 s non-overlapping windows** (125 samples),
`acc_x/y/z` features, **RobustScaler per subject** (fit on train only), and a
**subject-disjoint** split.

Ported to **PyTorch** (his code is Keras/TF) so it runs in your torch
environment; the FCN architecture and contrastive loss match his.

## The three experiments

The **model is identical** in all three; only the **data regime** changes — this
is exactly the "how does his model hold up against my work" comparison.

| Config | Features | Windowing / split | Normalisation |
|--------|----------|-------------------|---------------|
| **A** — `his-feat / his-window` | acc + gyro + **magnetometer** (9 ch)¹ | his 5 s **non-overlapping** windows; temporal 70/30 enroll→verify per subject | RobustScaler **per subject** |
| **B** — `his-feat / my-window` | acc + gyro + **magnetometer** (9 ch)¹ | **your** 15 s / 300-sample windows, stride 150 (50 % overlap), contiguous enroll→verify with 1-min guard gap | 8 Hz low-pass + global z-score (train stats) |
| **C** — `my-28ch / my-window` | **your 28 channels** (11 raw + 17 derived) | same as B | same as B |
| **D** — `my-28ch+mag / my-window` | **your 28 channels + magnetometer** (31 ch) | same as B | same as B |

Config **D** is C plus the 3 magnetometer channels appended (channels 0–27 are
byte-for-byte C's; 28–30 are the magnetometer), to test whether adding the
magnetometer to your derived feature set helps. C→D isolates the magnetometer's
contribution on top of your features.

¹ ContinAuth's input is **acc + gyro + magnetometer (9 ch)** — the default for
A/B. The dataset is a CoreMotion/SensorLog export, so the magnetometer columns
use the same `motion*` naming as the rest and are hardcoded to:

```
motionMagneticFieldX(µT)
motionMagneticFieldY(µT)
motionMagneticFieldZ(µT)
```

They are matched to the CSV header tolerantly **only** for the micro-sign
variant (µ/μ) and whitespace — it is the same column, never a substitute.
**If the magnetometer columns are not found the run ABORTS with an error — there
is NO fallback** to a magnetometer-free feature set. If your export spells them
differently, pass the exact names with `--mag_cols 'COLX' 'COLY' 'COLZ'`. Opt-in
alternatives via `--feature_set` (only if you deliberately want them): `acc3`
(his exact acc-only final model), `acc_gyro6` (acc+gyro, no magnetometer).

## Metrics reported

For every config, `evaluate.py` prints two numbers so you can compare on both
his terms and yours:

* **HIS metric** — per-user OCSVM mean EER (his protocol).
* **Comparable** — cosine gallery/probe **EER + rank-1** (mean-enroll template),
  the same gallery/probe protocol common in this space, so B/C line up against
  the numbers your own pipeline reports.

## Standalone

This bundle is **fully self-contained**: it imports nothing from your
`apw_network.py` / `verification.py`, so it can be dropped onto a different
machine and run against **only the dataset folder**. You do **not** need your
`split_ids.json` — if you don't pass one, a subject-disjoint train/val/test split
is built from the CSVs on disk (ContinAuth's own methodology), controlled by
`--train_frac` / `--val_frac` / `--seed`. Pass `--split_file` only if you want to
reuse an existing split. (Only requirements: `numpy scipy scikit-learn pandas
torch`.)

## Running

Point `--data_dirs` at the folder of per-subject CSVs — nothing else required.

```bash
# one-shot: train + evaluate A, B, C and print a comparison table (standalone split)
python -m continauth.run_all --data_dirs /path/to/dataset --device cuda

# or per-config, train then evaluate (use the SAME --seed so the split matches):
python -m continauth.train    --config c --data_dirs /path/to/dataset \
    --out continauth/checkpoints/c.pt --seed 712 --device cuda
python -m continauth.evaluate --config c --checkpoint continauth/checkpoints/c.pt \
    --data_dirs /path/to/dataset --split test --seed 712 --device cuda

# if your magnetometer columns are spelled differently, name them explicitly:
python -m continauth.run_all --data_dirs /path/to/dataset \
    --mag_cols 'motionMagneticFieldX(µT)' 'motionMagneticFieldY(µT)' 'motionMagneticFieldZ(µT)'

# reuse an existing split file instead of the internal split:
python -m continauth.run_all --data_dirs /path/to/dataset --split_file split_ids.json

# hold the normalisation CONSTANT across A/B/C/D (removes the scaler confounder,
# so configs differ only in features/windowing) — e.g. his RobustScaler for all:
python -m continauth.run_all --data_dirs /path/to/dataset \
    --split_file split_ids.json --scaler robust_subject

# his exact final model (acc only) for A:
python -m continauth.train --config a --feature_set acc3 --data_dirs /path/to/dataset
```

Verify the code end-to-end with no dataset (synthetic subjects):

```bash
python -m continauth.selftest
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | `ExperimentConfig` + the A/B/C presets, feature-set definitions |
| `data.py` | CSV loading, magnetometer resolution (hard error, no fallback), 8 Hz low-pass + 28-ch derived features, both windowing regimes, scaling, enroll/verify split, internal subject split |
| `model.py` | FCN encoder (L2-normalised embedding), in-batch contrastive loss |
| `pairs.py` | identity-balanced P×K batch iterator |
| `ocsvm_eval.py` | EER, per-user OCSVM protocol, val OCSVM tuning, cosine gallery/probe EER + rank-1 |
| `train.py` | CLI: train the Siamese net for one config |
| `evaluate.py` | CLI: extract deep features + report OCSVM/cosine metrics |
| `run_all.py` | CLI: train+evaluate A/B/C and print a comparison table |
| `selftest.py` | synthetic end-to-end smoke test |

## Notes / deviations from the original

* **Framework**: PyTorch (his is Keras/TF) — same FCN architecture, loss, OCSVM.
* **Embedding**: L2-normalised (his final Dense had a `sigmoid`). The sigmoid
  confines features to the positive orthant, which cripples cosine separation
  and mis-scales the contrastive margin; normalising to the unit sphere makes
  distance ∈ [0,2], `margin=1` meaningful, and cosine usable — a fairer baseline.
* **Training loss**: in-batch contrastive over identity-balanced **P×K batches**
  (P subjects × K windows), with the positive and negative terms averaged
  separately (his balanced pos/neg pairing). Tunables: `subjects_per_batch`,
  `windows_per_subject`, `batches_per_epoch` in `config.py`.
* **OCSVM**: `nu`/`gamma` are **grid-searched on the val split** (objective =
  mean per-owner EER), not his H-MOG constants `nu=0.165, gamma=8.296` (which do
  not transfer to this feature space). Pass `--no_tune_ocsvm` to use fixed values.
* **Sampling rate**: your data is **20 Hz** (his was 25 Hz). His 5 s window is
  kept as *seconds* → 100 samples at 20 Hz for config A.
* **Sessions**: H-MOG has ~24 sessions/subject; your dataset is one continuous
  recording/subject, so per-owner enroll/verify is a **temporal split** of that
  single session (contiguous, with a guard gap in the `mine` modes).
* **Magnetometer**: included for A/B (acc+gyro+mag, 9 ch) to match his input,
  using the hardcoded `motionMagneticField{X,Y,Z}(µT)` columns; **aborts with an
  error if they are absent (no fallback)**. Override names with `--mag_cols`.
* **Independence**: no import of your `apw_network.py` / `verification.py`; runs
  on any machine from the dataset folder alone (own subject split if no
  `--split_file`). The 28-channel derived features in config C reimplement the
  same transforms your pipeline uses so the comparison is apples-to-apples, but
  the code here is a standalone copy.
