"""
continauth_data.py
==================
Dedicated data path for the ContinAuth "his way" variant, which uses
accel + gyro + MAGNETOMETER (9 channels) — the faithful Centeno/ContinAuth input.

Our main pipeline (verification.py) deliberately BANS the magnetometer (location
leakage) and caches 28 derived channels. The magnetometer is still present in the
raw CSVs, so this module opts back into it *only for the baseline*, in a SEPARATE
cache, without touching the main pipeline or its banned-channel guard.

It subclasses VerificationData and overrides just the three channel-dependent
methods (caching, channel stats, window access); splits, the enroll/verify
partition and everything downstream (run_verification, compute_deep_embeddings)
are inherited unchanged and work on the 9-channel windows transparently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from apw_network import FEATURES, WINDOW_SAMPLES, lowpass_filter
from verification import VerificationData, valid_window_starts

log = logging.getLogger("continauth_data")

# accel XYZ + gyro XYZ from the raw FEATURES order (indices 0-5).
_ACCEL_GYRO_COLS: List[str] = list(FEATURES[0:6])


def detect_mag_columns(subject_to_path) -> List[str]:
    """Find the 3 magnetometer axis columns (X/Y/Z) from a CSV header, skipping
    accuracy/calibration columns. Returns [x_col, y_col, z_col]."""
    csv = next((p for p in subject_to_path.values() if Path(p).exists()), None)
    if csv is None:
        raise RuntimeError("detect_mag_columns: no CSV found to read a header from.")
    header = list(pd.read_csv(csv, nrows=0).columns)
    cand = [c for c in header
            if ("magnetic" in c.lower() or "magnetometer" in c.lower())
            and not any(bad in c.lower() for bad in ("accuracy", "calibration", "accur"))]
    axes = {}
    for axis in ("x", "y", "z"):
        for c in cand:
            stem = c.split("(")[0].strip().lower()          # drop unit suffix like (µT)
            if stem.endswith(axis):
                axes[axis] = c
                break
    if len(axes) != 3:
        raise RuntimeError(
            "Could not auto-detect magnetometer X/Y/Z columns. Candidates were: "
            f"{cand}. Pass them explicitly via --mag_cols 'colX,colY,colZ'.")
    cols = [axes["x"], axes["y"], axes["z"]]
    log.info("Magnetometer columns detected: %s", cols)
    return cols


class ContinAuthMagData(VerificationData):
    """VerificationData variant serving 9-channel (accel+gyro+mag) windows.

    IMPORTANT: point this at its OWN cache_dir — the per-subject .npy filenames
    collide with the 28-channel cache otherwise.
    """

    def __init__(self, *args, mag_cols: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mag_cols = mag_cols or detect_mag_columns(self.subject_to_path)
        self.sensor_cols = _ACCEL_GYRO_COLS + list(self.mag_cols)   # 9 columns
        self.n_channels = len(self.sensor_cols)                     # 9
        log.info("ContinAuth 'his way' input: %d channels %s",
                 self.n_channels, self.sensor_cols)

    def prepare_cache(self, subject_ids, force: bool = False) -> List[str]:
        """Stream each CSV -> accel+gyro+mag -> 8 Hz low-pass -> disk (no derived)."""
        ok: List[str] = []
        for sid in subject_ids:
            dp, sp = self._data_path(sid), self._starts_path(sid)
            if dp.exists() and sp.exists() and not force:
                ok.append(sid); continue
            csv = self.subject_to_path.get(sid)
            if csv is None:
                log.warning("[skip] no CSV for %s", sid); continue
            try:
                df = pd.read_csv(csv, low_memory=False, usecols=self.sensor_cols)
            except Exception as e:                                   # noqa: BLE001
                log.warning("[skip] %s: %s", sid, e); continue
            df = df.apply(pd.to_numeric, errors="coerce")
            nan_mask = df.isna().any(axis=1).values
            df = df.ffill().bfill()
            if df.isna().any().any():
                log.warning("[skip] unfillable NaNs in %s", sid); continue
            data = df[self.sensor_cols].values.astype(np.float32)   # (T, 9), CSV order
            data = lowpass_filter(data)                             # per-channel 8 Hz
            starts = valid_window_starts(nan_mask, nan_ratio=self.nan_ratio)
            if len(starts) == 0:
                log.warning("[skip] no valid windows for %s", sid); continue
            np.save(dp, data); np.save(sp, starts)
            ok.append(sid)
        log.info("Cached %d/%d subjects (9ch accel+gyro+mag) in %s",
                 len(ok), len(subject_ids), self.cache_dir)
        return ok

    def fit_channel_stats(self, train_ids) -> None:
        cached = [s for s in train_ids if self._data_path(s).exists()]
        n = 0
        ssum = np.zeros(self.n_channels); ssq = np.zeros(self.n_channels)
        for sid in cached:
            d = np.asarray(self._load(sid), dtype=np.float64)
            ssum += d.sum(0); ssq += (d * d).sum(0); n += d.shape[0]
        if n == 0:
            raise RuntimeError("fit_channel_stats: no cached train data found.")
        mean = ssum / n
        std = np.sqrt(np.maximum(ssq / n - mean ** 2, 0.0)) + 1e-8
        self.channel_mean, self.channel_std = mean.astype(np.float32), std.astype(np.float32)
        log.info("Channel stats (9ch) from %d subjects, %d frames.", len(cached), n)

    def get_windows(self, sid, starts) -> np.ndarray:
        assert self.channel_mean is not None, "set channel stats first"
        d = self._load(sid)
        out = np.empty((len(starts), self.n_channels, WINDOW_SAMPLES), dtype=np.float32)
        for i, s in enumerate(starts):
            w = np.asarray(d[s:s + WINDOW_SAMPLES], dtype=np.float32)
            out[i] = ((w - self.channel_mean) / self.channel_std).T
        return out
