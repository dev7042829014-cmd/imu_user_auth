"""
continauth.run_all
==================
Convenience runner: train + evaluate all three configurations (A, B, C) in one
go and print a side-by-side comparison table.

Example
-------
  python -m continauth.run_all \
      --data_dirs /path/to/dataset --split_file split_ids.json --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

from .config import CONFIGS
from .train import main as train_main
from .evaluate import evaluate


def main():
    p = argparse.ArgumentParser(description="Train+evaluate ContinAuth A/B/C")
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--split_file", default=None,
                   help="optional split_ids.json; omitted => standalone internal split")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--ckpt_dir", default="continauth/checkpoints")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--impostors_per_owner", type=int, default=2000)
    p.add_argument("--mag_cols", nargs=3, default=None,
                   help="explicit magnetometer column names (X Y Z) for configs A/B; "
                        "default = auto-detect from the CSV header")
    p.add_argument("--no_tune_ocsvm", dest="tune_ocsvm", action="store_false",
                   help="skip the val OCSVM grid-search (use fixed nu/gamma)")
    p.set_defaults(tune_ocsvm=True)
    p.add_argument("--scaler", choices=["robust_subject", "zscore_global"], default=None,
                   help="force the SAME normalisation for A/B/C/D so they differ only in "
                        "features/windowing (e.g. --scaler robust_subject to use his scaler "
                        "everywhere). Default keeps each config's own.")
    p.add_argument("--seed", type=int, default=712)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for key in CONFIGS:
        cfg = CONFIGS[key]
        if args.epochs:
            cfg.epochs = args.epochs
        ckpt = str(Path(args.ckpt_dir) / f"{key}.pt")
        train_main(SimpleNamespace(
            config=key, feature_set=None, variant="fcn", mag_cols=args.mag_cols,
            scaler=args.scaler,
            data_dirs=args.data_dirs, split_file=args.split_file, out=ckpt,
            train_frac=args.train_frac, val_frac=args.val_frac,
            epochs=args.epochs, device=args.device, workers=args.workers, seed=args.seed))
        res = evaluate(cfg, SimpleNamespace(
            config=key, feature_set=None, checkpoint=ckpt, mag_cols=args.mag_cols,
            data_dirs=args.data_dirs, split_file=args.split_file, split=args.split,
            train_frac=args.train_frac, val_frac=args.val_frac, tune_ocsvm=args.tune_ocsvm,
            impostors_per_owner=args.impostors_per_owner, out_json=ckpt.replace(".pt", ".json"),
            batch_size=256, device=args.device, seed=args.seed))
        results.append(res)

    print("\n" + "=" * 76)
    print("CONTINAUTH A/B/C COMPARISON  (his Siamese-FCN + OCSVM on your dataset)")
    print("=" * 76)
    print(f"{'config':28s}{'ch':>4s}{'OCSVM EER':>12s}{'cosine EER':>12s}{'rank-1':>10s}")
    print("-" * 76)
    for r in results:
        print(f"{r['config']:28s}{r['n_channels']:>4d}"
              f"{r['ocsvm_mean_eer']*100:>11.2f}%"
              f"{r['cosine_eer']*100:>11.2f}%"
              f"{r['cosine_rank1']*100:>9.2f}%")
    print("=" * 76)


if __name__ == "__main__":
    main()
