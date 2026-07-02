"""
continauth.data
===============
Self-contained data layer for the ContinAuth recreation.  Deliberately does NOT
import the user's apw_network / verification modules so this bundle can be run
independently of their own work.

Responsibilities
----------------
  * Read Apple-Watch CSVs (same columns / cleaning as the user's pipeline).
  * Optional 8 Hz zero-phase low-pass and the 17 derived channels (28-ch mode) —
    ported verbatim from apw_network so VERSION_C matches their features.
  * Two windowing regimes:
       "his"  — fixed-length non-overlapping windows (5 s), RobustScaler fit per
                subject, then a temporal enroll/verify split of each subject.
       "mine" — 300-sample / stride-150 windows with a contiguous enroll/verify
                partition and a guard gap, global z-score from TRAIN stats.
  * split_ids.json loading (train / val / test subject ids).

The output of `build_subject_windows` is, per subject, an (enroll, verify) pair
of channels-first window tensors ready for the network:  each (N, C, T) float32.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ExperimentConfig, SAMPLING_RATE, MAG_COLS

log = logging.getLogger("continauth.data")

_USERACCEL_IDX = [0, 1, 2]     # positions of acc XYZ within the 11 raw columns
_GYRO_IDX = [3, 4, 5]          # positions of gyro XYZ within the 11 raw columns


# ============================================================================
# Splits & file discovery
# ============================================================================

def load_split(split_file) -> Dict[str, List[str]]:
    """Read split_ids.json -> {'train':[...], 'val':[...], 'test':[...]}."""
    with open(split_file) as f:
        raw = json.load(f)
    return {
        "train": list(raw["train_ids"]),
        "val": list(raw.get("validation_ids", raw.get("val_ids", []))),
        "test": list(raw["test_ids"]),
    }


def make_subject_split(subject_ids: List[str], train_frac: float = 0.6,
                       val_frac: float = 0.1, seed: int = 712) -> Dict[str, List[str]]:
    """
    Build a subject-disjoint train/val/test split from the subjects on disk —
    ContinAuth's own methodology (all sessions of a subject fall in one split).
    Used when no split_ids.json is supplied, so this bundle runs on any system
    with just the dataset folder and NO dependency on the user's own split file.
    """
    ids = sorted(subject_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    return {"train": ids[:n_train],
            "val": ids[n_train:n_train + n_val],
            "test": ids[n_train + n_val:]}


def build_subject_to_path(*dirs) -> Dict[str, Path]:
    """Scan directories -> {subject_id (CSV stem): path}."""
    mapping: Dict[str, Path] = {}
    for d in dirs:
        d = Path(d)
        if d.exists():
            for p in sorted(d.glob("*.csv")):
                mapping[p.stem] = p
    return mapping


# ============================================================================
# Magnetometer columns (ContinAuth used acc + gyro + magnetometer)
# ============================================================================

def _read_header(csv_path: Path) -> List[str]:
    return list(pd.read_csv(csv_path, nrows=0, low_memory=False).columns)


def _norm(s: str) -> str:
    """Normalise a column name for tolerant matching: lowercase, drop spaces,
    and unify the micro-sign variants (µ U+00B5 / μ U+03BC). This resolves the
    SAME column across exports — it never substitutes a different signal."""
    return (s.lower().replace(" ", "")
            .replace("µ", "u").replace("μ", "u"))


def resolve_load_columns(cfg: ExperimentConfig, sample_csv: Path,
                         mag_cols: Optional[List[str]] = None) -> List[str]:
    """
    Map the feature-set template to the concrete CSV columns present in the file.

    Every requested column must exist in the header (matched tolerantly for the
    µ unit sign only). If ANY requested column is missing the run ABORTS — in
    particular, if the magnetometer columns are not present there is NO fallback
    to a magnetometer-free feature set.
    """
    template = cfg.resolve_feature_cols()
    # allow an explicit magnetometer override to replace the known MAG_COLS names
    if mag_cols is not None:
        if len(mag_cols) != 3:
            raise ValueError("--mag_cols needs exactly 3 columns (X Y Z)")
        repl = dict(zip(MAG_COLS, mag_cols))
        template = [repl.get(c, c) for c in template]

    header = _read_header(csv_path=sample_csv)
    header_by_norm = {_norm(h): h for h in header}

    resolved, missing = [], []
    for col in template:
        h = header_by_norm.get(_norm(col))
        if h is None:
            missing.append(col)
        else:
            resolved.append(h)

    if missing:
        mag_like = [c for c in header if "magnet" in c.lower()]
        wanted_mag = [c for c in missing if c in MAG_COLS or (mag_cols and c in mag_cols)]
        if wanted_mag:
            raise RuntimeError(
                f"Magnetometer columns not found in {sample_csv.name}: {wanted_mag}. "
                f"Magnetometer-like columns present in the header: {mag_like or 'NONE'}. "
                "Aborting (no fallback). Pass the exact names with "
                "--mag_cols 'COLX' 'COLY' 'COLZ' if your export spells them differently.")
        raise RuntimeError(
            f"Required columns not found in {sample_csv.name}: {missing}.")
    return resolved


# ============================================================================
# Signal processing  (ported from apw_network so the 28-ch set is identical)
# ============================================================================

def lowpass_filter(data: np.ndarray, cutoff_hz: float = 8.0,
                   fs: float = float(SAMPLING_RATE), order: int = 4) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    nyq = 0.5 * fs
    sos = butter(order, cutoff_hz / nyq, btype="low", analog=False, output="sos")
    return sosfiltfilt(sos, data, axis=0).astype(data.dtype)


def bandpass_filter(data: np.ndarray, low_hz: float, high_hz: float,
                    fs: float = float(SAMPLING_RATE), order: int = 4) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    nyq = 0.5 * fs
    lo = max(low_hz / nyq, 1e-4)
    hi = min(high_hz / nyq, 1.0 - 1e-4)
    sos = butter(order, [lo, hi], btype="band", analog=False, output="sos")
    return sosfiltfilt(sos, data, axis=0).astype(data.dtype)


def append_derived_features(data: np.ndarray) -> np.ndarray:
    """(T, 11) raw -> (T, 28). Identical to apw_network.append_derived_features."""
    ua_mag = np.linalg.norm(data[:, _USERACCEL_IDX], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(data[:, _GYRO_IDX], axis=1, keepdims=True)
    accel_diff = np.diff(data[:, _USERACCEL_IDX], axis=0,
                         prepend=data[:1, _USERACCEL_IDX]) * float(SAMPLING_RATE)
    gyro_diff = np.diff(data[:, _GYRO_IDX], axis=0,
                        prepend=data[:1, _GYRO_IDX]) * float(SAMPLING_RATE)
    jerk_mag = np.linalg.norm(accel_diff, axis=1, keepdims=True)
    ang_jerk_mag = np.linalg.norm(gyro_diff, axis=1, keepdims=True)
    tremor_lo_ua = bandpass_filter(data[:, _USERACCEL_IDX], 3.0, 7.0)
    tremor_lo_gyro = bandpass_filter(data[:, _GYRO_IDX], 3.0, 7.0)
    tremor_hi_ua = bandpass_filter(data[:, _USERACCEL_IDX], 7.0, 9.5)
    tremor_hi_gyro = bandpass_filter(data[:, _GYRO_IDX], 7.0, 9.5)
    walking_act = bandpass_filter(ua_mag, 0.5, 2.0)
    return np.concatenate(
        [data, ua_mag, gyro_mag, jerk_mag, ang_jerk_mag,
         tremor_lo_ua, tremor_lo_gyro, tremor_hi_ua, tremor_hi_gyro,
         walking_act], axis=1)


# ============================================================================
# Per-subject raw signal loading
# ============================================================================

def load_subject_signal(csv_path: Path, cfg: ExperimentConfig,
                        load_cols: List[str]) -> Optional[np.ndarray]:
    """
    Load one subject CSV -> a (T, C) float32 signal, cleaned & (optionally)
    filtered / derived according to the experiment config.

    `load_cols` are the concrete CSV columns to read (magnetometer sentinels
    already resolved).  For feature_set="mine28" the 11 raw columns are loaded,
    low-passed and expanded to 28 channels.  Otherwise the requested
    acc/gyro/mag columns are kept (and low-passed iff cfg.lowpass).
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False, usecols=load_cols)
    except Exception as e:                                            # noqa: BLE001
        log.warning("[skip] %s: %s", csv_path.stem, e)
        return None

    df = df[load_cols].apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill()
    if df.isna().any().any():
        log.warning("[skip] unfillable NaNs in %s", csv_path.stem)
        return None

    data = df.values.astype(np.float32)

    if cfg.lowpass:
        data = lowpass_filter(data)
    if cfg.derive:
        # cfg.derive is only set for mine28 where load_cols == 11 raw columns
        data = append_derived_features(data)
    return data


# ============================================================================
# Windowing
# ============================================================================

def sliding_windows(data: np.ndarray, win: int, stride: int) -> np.ndarray:
    """(T, C) -> (N, C, T_win) channels-first windows at the given stride."""
    T = data.shape[0]
    if T < win:
        return np.empty((0, data.shape[1], win), dtype=np.float32)
    starts = range(0, T - win + 1, stride)
    w = np.stack([data[s:s + win] for s in starts])      # (N, win, C)
    return w.transpose(0, 2, 1).astype(np.float32)       # (N, C, win)


def _his_window_params(cfg: ExperimentConfig) -> Tuple[int, int]:
    win = int(round(cfg.window_seconds * SAMPLING_RATE))
    step = int(round(cfg.step_seconds * SAMPLING_RATE))
    return win, step


def window_signal(data: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    if cfg.window_mode == "his":
        win, step = _his_window_params(cfg)
        return sliding_windows(data, win, step)
    elif cfg.window_mode == "mine":
        return sliding_windows(data, cfg.window_samples, cfg.window_stride)
    raise ValueError(f"unknown window_mode {cfg.window_mode!r}")


def split_enroll_verify(windows: np.ndarray, cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Contiguous temporal split of one subject's windows into (enroll, verify).

    A guard gap of `gap_seconds` (converted to a whole number of windows) is
    dropped at the boundary so no window straddles enroll and verify.  With the
    "his" mode gap_seconds is 0 (his OCSVM samples windows freely).
    """
    n = len(windows)
    if n == 0:
        return windows, windows
    cut = int(round(n * cfg.enroll_ratio))
    if cfg.window_mode == "mine":
        win = cfg.window_samples
        # number of windows that fit inside half the guard gap on each side
        gap_win = int(np.ceil((cfg.gap_seconds * SAMPLING_RATE) / max(1, cfg.window_stride)))
        half = gap_win // 2
    else:
        half = 0
    enroll = windows[: max(0, cut - half)]
    verify = windows[cut + half:]
    return enroll, verify


# ============================================================================
# Normalisation
# ============================================================================

def robust_scale_per_subject(windows: np.ndarray) -> np.ndarray:
    """
    RobustScaler (median / IQR) fit on THIS subject's own windows, per channel —
    ContinAuth's scaler="robust", scope="subject".  windows: (N, C, T).
    """
    if len(windows) == 0:
        return windows
    x = windows.transpose(0, 2, 1).reshape(-1, windows.shape[1])   # (N*T, C)
    med = np.median(x, axis=0)
    q1 = np.percentile(x, 25, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    iqr = (q3 - q1)
    iqr[iqr == 0] = 1.0
    scaled = (windows - med[None, :, None]) / iqr[None, :, None]
    return scaled.astype(np.float32)


def zscore_apply(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Global per-channel z-score with pre-computed TRAIN statistics."""
    if len(windows) == 0:
        return windows
    return ((windows - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def fit_channel_stats(windows_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over a list of (N, C, T) window arrays (TRAIN only)."""
    n = 0
    C = windows_list[0].shape[1]
    ssum = np.zeros(C, dtype=np.float64)
    ssq = np.zeros(C, dtype=np.float64)
    for w in windows_list:
        if len(w) == 0:
            continue
        x = w.transpose(0, 2, 1).reshape(-1, C).astype(np.float64)
        ssum += x.sum(0)
        ssq += (x * x).sum(0)
        n += x.shape[0]
    if n == 0:
        raise RuntimeError("fit_channel_stats: no windows")
    mean = ssum / n
    std = np.sqrt(np.maximum(ssq / n - mean ** 2, 0.0)) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


# ============================================================================
# Top-level dataset builder
# ============================================================================

class ContinAuthData:
    """
    Loads a set of subjects and produces per-subject (enroll, verify) windows,
    handling the config-specific windowing + scaling.

    Usage
    -----
      data = ContinAuthData(cfg, data_dirs, split_file)
      # training windows for the siamese net (train subjects, enroll+verify pooled):
      Xtr, ytr = data.training_windows(data.split["train"])
      # per-subject enroll/verify for OCSVM evaluation (test subjects):
      per_subj = data.enroll_verify(data.split["test"])
    """

    def __init__(self, cfg: ExperimentConfig, data_dirs, split_file=None,
                 mag_cols: Optional[List[str]] = None,
                 train_frac: float = 0.6, val_frac: float = 0.1, seed: int = 712):
        self.cfg = cfg
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.subject_to_path = build_subject_to_path(*data_dirs)
        self._channel_mean: Optional[np.ndarray] = None
        self._channel_std: Optional[np.ndarray] = None

        # Subject split: use split_ids.json if one is supplied, otherwise build a
        # subject-disjoint split from the CSVs on disk (his methodology). This
        # keeps the bundle self-contained — no dependency on the user's split file.
        if split_file and Path(split_file).exists():
            self.split = load_split(split_file)
            split_src = f"file {split_file}"
        else:
            self.split = make_subject_split(list(self.subject_to_path), train_frac, val_frac, seed)
            split_src = f"internal (train={train_frac}, val={val_frac}, seed={seed})"

        # Resolve + validate the concrete load columns against a real CSV header.
        # Missing columns (esp. the magnetometer) abort here — no fallback.
        self.load_cols = cfg.resolve_feature_cols()
        if self.subject_to_path:
            sample = next(iter(self.subject_to_path.values()))
            self.load_cols = resolve_load_columns(cfg, sample, mag_cols)

        log.info("Config %s | subjects on disk=%d | split %s -> train=%d val=%d test=%d | cols=%d",
                 cfg.name, len(self.subject_to_path), split_src, len(self.split["train"]),
                 len(self.split["val"]), len(self.split["test"]), len(self.load_cols))

    # -- raw windows per subject (before global z-score) --------------------
    def _subject_windows(self, sid: str) -> Optional[np.ndarray]:
        path = self.subject_to_path.get(sid)
        if path is None:
            log.warning("[skip] no CSV for %s", sid)
            return None
        sig = load_subject_signal(path, self.cfg, self.load_cols)
        if sig is None:
            return None
        w = window_signal(sig, self.cfg)
        if len(w) == 0:
            log.warning("[skip] no windows for %s", sid)
            return None
        if self.cfg.scaler == "robust_subject":
            w = robust_scale_per_subject(w)
        return w

    # -- fit global z-score stats on TRAIN subjects ------------------------
    def fit_global_stats(self, train_ids: List[str]) -> None:
        if self.cfg.scaler != "zscore_global":
            return
        pooled = []
        for sid in train_ids:
            w = self._subject_windows(sid)
            if w is not None:
                pooled.append(w)
        if not pooled:
            raise RuntimeError("fit_global_stats: no train windows")
        self._channel_mean, self._channel_std = fit_channel_stats(pooled)
        log.info("Global z-score stats fit on %d train subjects.", len(pooled))

    def set_global_stats(self, mean, std) -> None:
        self._channel_mean = np.asarray(mean, dtype=np.float32)
        self._channel_std = np.asarray(std, dtype=np.float32)

    @property
    def channel_mean(self):
        return self._channel_mean

    @property
    def channel_std(self):
        return self._channel_std

    def _finalize(self, w: np.ndarray) -> np.ndarray:
        if self.cfg.scaler == "zscore_global":
            assert self._channel_mean is not None, "call fit_global_stats first"
            w = zscore_apply(w, self._channel_mean, self._channel_std)
        return w

    # -- public: pooled training windows + integer subject labels ----------
    def training_windows(self, subject_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        label_of: Dict[str, int] = {}
        for sid in subject_ids:
            w = self._subject_windows(sid)
            if w is None:
                continue
            w = self._finalize(w)
            label_of.setdefault(sid, len(label_of))
            Xs.append(w)
            ys.append(np.full(len(w), label_of[sid], dtype=np.int64))
        if not Xs:
            raise RuntimeError("training_windows: nothing loaded")
        return np.concatenate(Xs, 0), np.concatenate(ys, 0)

    # -- public: per-subject enroll/verify windows -------------------------
    def enroll_verify(self, subject_ids: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for sid in subject_ids:
            w = self._subject_windows(sid)
            if w is None:
                continue
            enroll, verify = split_enroll_verify(w, self.cfg)
            if len(enroll) == 0 or len(verify) == 0:
                log.warning("[skip] %s: empty enroll/verify (%d/%d)",
                            sid, len(enroll), len(verify))
                continue
            out[sid] = {"enroll": self._finalize(enroll),
                        "verify": self._finalize(verify)}
        return out
