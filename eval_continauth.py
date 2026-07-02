"""
eval_continauth.py
==================
Evaluate the ContinAuth / Centeno Siamese-FCN baseline on OUR data with BOTH
verification protocols, reusing our harness (run_verification):

  * OCSVM  — ContinAuth's actual protocol: a per-user One-Class SVM on the deep
             features. nu/gamma are TUNED on the validation split (grid search,
             objective = EER), exactly as they tuned on H-MOG, then locked and
             reported once on test. (Their H-MOG values nu=0.165/gamma=8.296 are
             dataset-specific and are NOT reused — we re-tune for our feature space.)
  * cosine — our template protocol, for a like-for-like line against our models.

Example
-------
  python eval_continauth.py --channels accel_gyro_mag \
      --checkpoint checkpoints_continauth/continauth_accel_gyro_mag.best.pt \
      --split test --data_dirs dataset --cache_dir cache_verification \
      --tune_ocsvm --plot_dir plots/continauth_accel_gyro_mag
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from verification import (
    VerificationData, DEFAULT_DATA_DIRS,
    compute_deep_embeddings, CosineScorer, OCSVMScorer,
    run_verification, format_report, result_to_dict, compute_score_matrix,
    bootstrap_metrics,
)
from continauth_model import build_continauth_model


# Grid for the val OCSVM search (their objective = EER). gamma includes sklearn's
# 'scale' plus a log range; nu spans loose->tight boundaries.
_NU_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
_GAMMA_GRID = ["scale", 0.01, 0.1, 1.0, 4.0, 8.0, 16.0]


def tune_ocsvm(emb_val, seed, impostors_per_user, log):
    """Grid-search (nu, gamma) on VAL embeddings, minimizing EER (ContinAuth's
    tuning objective). Returns (best_nu, best_gamma, best_val_eer)."""
    best = (float("inf"), None, None)
    for nu in _NU_GRID:
        for g in _GAMMA_GRID:
            try:
                r = run_verification(emb_val, OCSVMScorer(nu=nu, gamma=g),
                                     seed, impostors_per_user)
            except Exception as e:                                   # noqa: BLE001
                log.warning("OCSVM nu=%.3f gamma=%s failed: %s", nu, g, e)
                continue
            if r.eer == r.eer and r.eer < best[0]:
                best = (r.eer, nu, g)
    log.info("OCSVM tuned on val: nu=%s gamma=%s (val EER=%.2f%%)",
             best[1], best[2], best[0] * 100 if best[0] < float("inf") else float("nan"))
    return best[1], best[2], best[0]


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("eval_continauth")
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    channels = ckpt.get("channels", args.channels)

    if channels == "accel_gyro_mag":
        from continauth_data import ContinAuthMagData
        data = ContinAuthMagData(f"{args.cache_dir}_mag9", args.split_file,
                                 [Path(d) for d in args.data_dirs], seed=args.seed,
                                 enroll_ratio=args.enroll_ratio, gap_seconds=args.gap_seconds,
                                 mag_cols=ckpt.get("mag_cols"))
    else:
        data = VerificationData(args.cache_dir, args.split_file,
                                [Path(d) for d in args.data_dirs], seed=args.seed,
                                enroll_ratio=args.enroll_ratio, gap_seconds=args.gap_seconds)

    model = build_continauth_model(channels=channels,
                                   embed_dim=ckpt.get("embed_dim", 32)).to(device)
    model.load_state_dict(ckpt["network"])
    data.set_channel_stats(ckpt["channel_mean"], ckpt["channel_std"])

    eval_ids = data.prepare_cache(data.split[args.split])
    if len(eval_ids) < 2:
        raise RuntimeError(f"Need >=2 cached {args.split} subjects; got {len(eval_ids)}.")
    log.info("Evaluating %d %s subjects | channels=%s", len(eval_ids), args.split, channels)

    config = {"model": f"continauth_{channels}", "split": args.split, "seed": args.seed,
              "channels": channels, "n_subjects": len(eval_ids),
              "enroll_ratio": args.enroll_ratio, "gap_seconds": args.gap_seconds,
              "impostors_per_user": args.impostors_per_user,
              "checkpoint": args.checkpoint}

    # Resolve OCSVM hyper-params: tune on VAL (default) or take the CLI values.
    nu, gamma = args.ocsvm_nu, args.ocsvm_gamma
    if args.tune_ocsvm:
        val_ids = data.prepare_cache(data.split["val"])
        emb_val = compute_deep_embeddings(model, data, val_ids, device, args.batch_size)
        nu, gamma, val_eer = tune_ocsvm(emb_val, args.seed, args.impostors_per_user, log)
        config["ocsvm_tuned_on"] = "val"; config["ocsvm_val_eer"] = val_eer
    config["ocsvm_nu"], config["ocsvm_gamma"] = nu, gamma

    # Test embeddings + both scorers.
    emb = compute_deep_embeddings(model, data, eval_ids, device, args.batch_size)
    results = [run_verification(emb, CosineScorer(), args.seed, args.impostors_per_user),
               run_verification(emb, OCSVMScorer(nu=nu, gamma=gamma),
                                args.seed, args.impostors_per_user)]

    report = format_report(f"continauth_{channels}", results, config)

    if args.bootstrap > 0:
        sids, M = compute_score_matrix(emb, CosineScorer())
        b = bootstrap_metrics(M, n_boot=args.bootstrap, seed=args.seed)
        e, r = b["eer"], b["rank1"]
        report += (f"\n[ cosine bootstrap 95% CI over {len(sids)} subjects, "
                   f"{b['n_boot']} resamples ]"
                   f"\n  EER    : {e[0]*100:6.2f} %   95% CI [{e[1]*100:.2f}, {e[2]*100:.2f}]"
                   f"\n  Rank-1 : {r[0]*100:6.2f} %   95% CI [{r[1]*100:.2f}, {r[2]*100:.2f}]"
                   f"\n\n{'=' * 72}")
    print("\n" + report)

    if args.plot_dir:
        try:
            from plot_metrics import plot_score_matrix
            sids, M = compute_score_matrix(emb, CosineScorer())
            outp = Path(args.plot_dir) / f"continauth_{channels}_{args.split}_matrix.png"
            outp.parent.mkdir(parents=True, exist_ok=True)
            plot_score_matrix(M, sids, outp,
                              title=f"ContinAuth ({channels}) {args.split}: probe vs gallery")
            log.info("Wrote heatmap → %s", outp)
        except Exception as e:                                       # noqa: BLE001
            log.warning("heatmap skipped (%s)", e)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"config": config, "results": [result_to_dict(r) for r in results]},
                      f, indent=2, default=str)
        log.info("Wrote results → %s", args.out_json)


def build_argparser():
    p = argparse.ArgumentParser(description="Evaluate ContinAuth Siamese-FCN baseline")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--channels", choices=["accel_gyro_mag", "full"],
                   default="accel_gyro_mag",
                   help="fallback if the checkpoint doesn't record its channel set")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--data_dirs", nargs="+", default=list(DEFAULT_DATA_DIRS))
    p.add_argument("--split_file", default="split_ids.json")
    p.add_argument("--cache_dir", default="./cache_verification")
    p.add_argument("--tune_ocsvm", action="store_true",
                   help="grid-search nu/gamma on val (EER) before test (recommended)")
    p.add_argument("--ocsvm_nu", type=float, default=0.1,
                   help="used only if --tune_ocsvm is NOT set")
    p.add_argument("--ocsvm_gamma", default="scale",
                   help="float or 'scale'/'auto'; used only if --tune_ocsvm is NOT set")
    p.add_argument("--enroll_ratio", type=float, default=0.7)
    p.add_argument("--gap_seconds", type=float, default=60.0)
    p.add_argument("--impostors_per_user", type=int, default=2000)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--plot_dir", default=None)
    p.add_argument("--out_json", default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p


def _coerce_gamma(g):
    try:
        return float(g)
    except (TypeError, ValueError):
        return g


if __name__ == "__main__":
    a = build_argparser().parse_args()
    a.ocsvm_gamma = _coerce_gamma(a.ocsvm_gamma)
    main(a)
