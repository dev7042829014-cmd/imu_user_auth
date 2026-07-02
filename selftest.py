"""
continauth.selftest
===================
End-to-end smoke test on SYNTHETIC data (no real dataset needed).  Fabricates a
handful of subject CSVs with subject-distinct motion, writes a split_ids.json,
then runs train + evaluate for each of the three configurations to prove the
whole pipeline executes and produces finite metrics.

Run:  python -m continauth.selftest
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .config import CONFIGS, RAW_FEATURES
from .train import main as train_main
from .evaluate import evaluate
from .config import CONFIGS as _C


def _make_subject_csv(path: Path, sid: str, n: int, fs: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    # subject-specific tremor frequency + posture offsets so identity is learnable
    f_tremor = 3.5 + (int(sid) % 7) * 0.6
    phase = (int(sid) % 5)
    cols = {}
    for i, c in enumerate(RAW_FEATURES):
        base = 0.0
        if "Acceleration" in c:
            base = 0.05 * np.sin(2 * np.pi * f_tremor * t + phase + i)
        elif "RotationRate" in c:
            base = 0.10 * np.sin(2 * np.pi * (f_tremor + 1) * t + phase + i)
        elif "Gravity" in c:
            base = 0.5 + 0.01 * (i + phase)
        else:  # roll/pitch
            base = 0.2 * np.sin(2 * np.pi * 0.1 * t + phase)
        cols[c] = (base + 0.02 * rng.standard_normal(n)).astype(np.float32)
    # magnetometer columns (ContinAuth used acc+gyro+mag) — with the µT unit
    # suffix so the auto-detector is exercised.
    for j, ax in enumerate("XYZ"):
        base = 20.0 + 5.0 * np.sin(2 * np.pi * 0.05 * t + phase + j) + 2.0 * phase
        cols[f"motionMagneticField{ax}(µT)"] = (base + 0.5 * rng.standard_normal(n)).astype(np.float32)
    cols["motionMagneticFieldAccuracy(int)"] = np.full(n, 2, dtype=np.int32)
    pd.DataFrame(cols).to_csv(path, index=False)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    log = logging.getLogger("continauth.selftest")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir = tmp / "dataset"
        data_dir.mkdir()

        # 8 train + 6 test subjects, ~600 s each
        train_ids = [str(100 + i) for i in range(8)]
        test_ids = [str(200 + i) for i in range(6)]
        for k, sid in enumerate(train_ids + test_ids):
            _make_subject_csv(data_dir / f"{sid}.csv", sid, n=600 * 20, seed=k)

        split_file = tmp / "split_ids.json"
        split_file.write_text(json.dumps(
            {"train_ids": train_ids, "validation_ids": test_ids[:2], "test_ids": test_ids}))

        results = []
        for key in CONFIGS:
            cfg = CONFIGS[key]
            log.info("========== CONFIG %s (%s) ==========", key, cfg.name)
            ckpt = tmp / f"{key}.pt"
            # tiny training run
            # config 'a' uses the standalone internal split (no split_file) to
            # exercise that path; b/c use the split_ids.json file.
            sf = None if key == "a" else str(split_file)
            targs = SimpleNamespace(
                config=key, feature_set=None, variant="fcn", mag_cols=None,
                data_dirs=[str(data_dir)], split_file=sf,
                train_frac=0.6, val_frac=0.15,
                out=str(ckpt), epochs=3, device="cpu", workers=0, seed=712)
            # shrink pairs for speed via the config object
            cfg.pairs_per_epoch = 1500
            cfg.epochs = 3
            train_main(targs)

            eargs = SimpleNamespace(
                config=key, feature_set=None, checkpoint=str(ckpt), mag_cols=None,
                data_dirs=[str(data_dir)], split_file=sf,
                train_frac=0.6, val_frac=0.15,
                split="test", impostors_per_owner=2000, out_json=None,
                batch_size=128, device="cpu", seed=712)
            res = evaluate(_C[key], eargs)
            assert np.isfinite(res["ocsvm_mean_eer"]), f"{key}: non-finite OCSVM EER"
            assert np.isfinite(res["cosine_eer"]), f"{key}: non-finite cosine EER"
            results.append(res)

        print("\n\n=== SELFTEST SUMMARY (synthetic data) ===")
        for r in results:
            print(f"  {r['config']:26s} | ch={r['n_channels']:2d} | "
                  f"OCSVM EER {r['ocsvm_mean_eer']*100:5.1f}% | "
                  f"cosine EER {r['cosine_eer']*100:5.1f}% | rank1 {r['cosine_rank1']*100:5.1f}%")
        print("=== all configs ran end-to-end OK ===")


if __name__ == "__main__":
    main()
