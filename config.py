"""
continauth.config
=================
Experiment presets for the ContinAuth (Buech 2019 / dynobo) recreation, adapted
to the Apple-Watch IMU dataset.

Three experiment configurations are defined (see the module-level constants at
the bottom):

  VERSION_A  — HIS features (acc+gyro) + HIS windowing/split
               (5 s non-overlapping windows, RobustScaler per subject, a
                temporal 70/30 enroll->verify split per subject).

  VERSION_B  — HIS features (acc+gyro) + MY windowing/split
               (15 s / 300-sample windows @ 20 Hz, stride 150 = 50 % overlap,
                8 Hz low-pass + global z-score, contiguous enroll->verify with a
                1-min guard gap, split_ids.json subject split).

  VERSION_C  — MY 28 channels (11 raw + 17 derived) + MY windowing/split.

The MODEL is the same in every configuration: a Siamese fully-convolutional
network (FCN) trained with contrastive loss, whose 32-d deep features feed a
per-user one-class SVM.  Only the *data regime* changes between A / B / C, which
is exactly the comparison requested: how does his model hold up on my data and
my windowing vs his own.

Original ContinAuth reference (VALID-FCN-ROBUST, chapter-5-5-siamese-cnn.ipynb):
  frequency=25 Hz, window=5 s (125 samples), step=125 (non-overlap),
  feature_cols=["acc_x","acc_y","acc_z"], scaler="robust" (scope=subject,
  fit on train only), FCN filters=[32,64,32], Adam lr=1e-3, batch=300,
  epochs=40, contrastive margin=1, OCSVM rbf nu=0.165 gamma=8.296.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Raw sensor columns present in the Apple-Watch CSVs (same names as the user's
# apw_network.FEATURES), plus the magnetometer that ContinAuth used.
# ---------------------------------------------------------------------------
ACC_COLS = [
    "motionUserAccelerationX(G)",
    "motionUserAccelerationY(G)",
    "motionUserAccelerationZ(G)",
]
GYRO_COLS = [
    "motionRotationRateX(rad/s)",
    "motionRotationRateY(rad/s)",
    "motionRotationRateZ(rad/s)",
]
GRAVITY_COLS = [
    "motionGravityX(G)",
    "motionGravityY(G)",
    "motionGravityZ(G)",
]
ANGLE_COLS = ["motionRoll(rad)", "motionPitch(rad)"]

# The 11 raw columns the user's pipeline loads (order matters for the 28-ch mode)
RAW_FEATURES = ACC_COLS + GYRO_COLS + GRAVITY_COLS + ANGLE_COLS

# ---- Magnetometer -----------------------------------------------------------
# ContinAuth's input is acc + gyro + MAGNETOMETER (9 channels). The dataset is a
# CoreMotion / SensorLog export, so the magnetometer columns use the SAME
# "motion*" naming as the other channels (motionUserAcceleration, motionRotationRate,
# motionGravity, ...). The known, expected magnetometer column names are:
MAG_COLS = [
    "motionMagneticFieldX(µT)",
    "motionMagneticFieldY(µT)",
    "motionMagneticFieldZ(µT)",
]
# These names are matched tolerantly to the CSV header (only the micro-sign
# variant µ/μ and surrounding whitespace are normalised — it is the SAME column,
# not a substitute). If any of the three cannot be found, loading ABORTS with an
# error: there is NO fallback to a magnetometer-free feature set. Use --mag_cols
# to supply the exact names if your export spells them differently.

# ---- Feature-set presets ---------------------------------------------------
ACC3 = list(ACC_COLS)                              # his actual FINAL best model (acc only)
ACC_GYRO6 = ACC_COLS + GYRO_COLS                   # acc + gyro (explicit opt-in only)
ACC_GYRO_MAG9 = ACC_COLS + GYRO_COLS + MAG_COLS    # his real input: acc + gyro + magnetometer

FEATURE_SETS = {
    "acc3": ACC3,
    "acc_gyro6": ACC_GYRO6,
    "acc_gyro_mag9": ACC_GYRO_MAG9,
    "mine28": "DERIVED_28",   # special sentinel: full 28-channel derived set
}

SAMPLING_RATE = 20  # Apple-Watch CoreMotion export is 20 Hz


@dataclass
class ExperimentConfig:
    """All parameters needed to run one A/B/C experiment."""

    name: str

    # --- data / features ---
    feature_set: str            # key into FEATURE_SETS
    lowpass: bool               # apply 8 Hz zero-phase low-pass before windowing
    derive: bool                # append the 17 derived channels (=> 28 total)

    # --- windowing ---
    window_mode: str            # "his" (5 s non-overlap) | "mine" (300/150 + enroll/verify)
    window_seconds: float       # only used by window_mode="his"
    step_seconds: float         # only used by window_mode="his"
    window_samples: int         # only used by window_mode="mine"
    window_stride: int          # only used by window_mode="mine"

    # --- enroll / verify partition ---
    enroll_ratio: float         # fraction of the (per-subject) session used to enroll
    gap_seconds: float          # guard gap dropped at the enroll->verify boundary (mine)

    # --- normalisation ---
    scaler: str                 # "robust_subject" | "zscore_global"

    # --- siamese network (his FCN) ---
    filters: List[int] = field(default_factory=lambda: [32, 64, 32])
    embed_dim: int = 32
    margin: float = 1.0
    epochs: int = 40
    batch_size: int = 300
    lr: float = 1e-3
    pairs_per_epoch: int = 20000

    # --- one-class SVM (his tuned defaults) ---
    ocsvm_nu: float = 0.165
    ocsvm_gamma: float = 8.296

    def resolve_feature_cols(self) -> List[str]:
        """
        Return the feature-set column template (may contain MAG_SENTINELS, which
        data.py replaces with the concrete detected magnetometer columns).
        For mine28 the 11 raw columns are loaded and derived to 28 downstream.
        """
        if self.feature_set == "mine28":
            return list(RAW_FEATURES)          # load all 11 raw, derive to 28 later
        return list(FEATURE_SETS[self.feature_set])

    @property
    def uses_magnetometer(self) -> bool:
        fs = FEATURE_SETS.get(self.feature_set, [])
        return isinstance(fs, list) and any(c in MAG_COLS for c in fs)

    @property
    def n_channels(self) -> int:
        if self.feature_set == "mine28":
            return 28
        return len(FEATURE_SETS[self.feature_set])


# ---------------------------------------------------------------------------
# The three requested experiments.  A/B use his real input — acc + gyro +
# MAGNETOMETER (9 ch). If the magnetometer columns are missing the run ABORTS
# (no fallback). Override with --feature_set acc3 (his acc-only final model) or
# acc_gyro6 only if you deliberately want to drop the magnetometer.
# ---------------------------------------------------------------------------

VERSION_A = ExperimentConfig(
    name="A_his-feat_his-window",
    feature_set="acc_gyro_mag9",
    lowpass=False,             # his pipeline: resample + robust scale only
    derive=False,
    window_mode="his",
    window_seconds=5.0,
    step_seconds=5.0,          # non-overlapping, like his step_width == window_size
    window_samples=0,
    window_stride=0,
    enroll_ratio=0.7,
    gap_seconds=0.0,           # his OCSVM samples windows; no contiguous guard gap
    scaler="robust_subject",
)

VERSION_B = ExperimentConfig(
    name="B_his-feat_my-window",
    feature_set="acc_gyro_mag9",
    lowpass=True,              # my pipeline: 8 Hz low-pass + global z-score
    derive=False,
    window_mode="mine",
    window_seconds=0.0,
    step_seconds=0.0,
    window_samples=300,        # 15 s @ 20 Hz
    window_stride=150,         # 50 % overlap
    enroll_ratio=0.7,
    gap_seconds=60.0,
    scaler="zscore_global",
)

VERSION_C = ExperimentConfig(
    name="C_my-28ch_my-window",
    feature_set="mine28",
    lowpass=True,
    derive=True,
    window_mode="mine",
    window_seconds=0.0,
    step_seconds=0.0,
    window_samples=300,
    window_stride=150,
    enroll_ratio=0.7,
    gap_seconds=60.0,
    scaler="zscore_global",
)

CONFIGS = {"a": VERSION_A, "b": VERSION_B, "c": VERSION_C}
