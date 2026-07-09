"""
continauth.evaluate
===================
Load a trained ContinAuth Siamese checkpoint, extract 32-d deep features for the
enroll/verify windows of the TEST subjects, and report:

  * his metric   — per-user one-class SVM Equal Error Rate (mean over owners);
  * comparable   — cosine gallery/probe EER + rank-1 (the user's own protocol).

Example
-------
  python -m continauth.evaluate --config c \
      --checkpoint continauth/checkpoints/c.pt \
      --data_dirs /path/to/dataset --split_file split_ids.json \
      --split test --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .config import CONFIGS, FEATURE_SETS
from .config import OCSVM_NU_GRID, OCSVM_GAMMA_GRID
from .data import ContinAuthData
from .model import build_base
from .ocsvm_eval import per_owner_ocsvm_eer, cosine_eer_rank1, tune_ocsvm


def _batched(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i:i + n]


def extract_features(model, feats_windows: Dict[str, Dict[str, np.ndarray]],
                     device, batch_size: int = 256) -> Dict[str, Dict[str, np.ndarray]]:
    model.eval()
    edim = getattr(model, "embed_dim", 32)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for sid, roles in feats_windows.items():
            entry = {}
            for role, w in roles.items():
                chunks = [model(torch.from_numpy(b).to(device)).cpu().numpy()
                          for b in _batched(w, batch_size)]
                entry[role] = (np.concatenate(chunks, 0) if chunks
                               else np.zeros((0, edim), np.float32))
            out[sid] = entry
    return out


def evaluate(cfg, args) -> dict:
    log = logging.getLogger("continauth.eval")
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = build_base(ckpt["variant"], ckpt["n_channels"], ckpt["window"],
                       ckpt["filters"], ckpt["embed_dim"]).to(device)
    model.load_state_dict(ckpt["base_state"])

    # Preprocess exactly as at train time (the scaler was recorded in the ckpt).
    cfg.scaler = ckpt.get("scaler", cfg.scaler)

    # Reuse the exact magnetometer columns resolved at train time (if any), so
    # eval loads identical channels regardless of which CSV header is sampled.
    mag_cols = getattr(args, "mag_cols", None)
    data = ContinAuthData(cfg, args.data_dirs, args.split_file, mag_cols=mag_cols,
                          train_frac=args.train_frac, val_frac=args.val_frac,
                          seed=args.seed)
    if mag_cols is None and ckpt.get("load_cols"):
        data.load_cols = list(ckpt["load_cols"])   # trust the trained checkpoint's columns
    if cfg.scaler == "zscore_global":
        if ckpt.get("channel_mean") is None:
            raise RuntimeError("checkpoint has no channel stats but config uses z-score")
        data.set_global_stats(ckpt["channel_mean"], ckpt["channel_std"])

    # --- OCSVM hyper-params: tune (nu, gamma) on the VAL split by default -----
    tune = getattr(args, "tune_ocsvm", cfg.tune_ocsvm)
    nu, gamma = cfg.ocsvm_nu, cfg.ocsvm_gamma
    if tune:
        val_windows = data.enroll_verify(data.split["val"])
        if len(val_windows) >= 2:
            val_feats = extract_features(model, val_windows, device, args.batch_size)
            nu, gamma, veer = tune_ocsvm(val_feats, OCSVM_NU_GRID, OCSVM_GAMMA_GRID,
                                         args.impostors_per_owner, args.seed)
            log.info("OCSVM tuned on val: nu=%s gamma=%s (val mean EER=%.2f%%)",
                     nu, gamma, veer * 100)
        else:
            log.warning("val split too small to tune OCSVM; using nu=%s gamma=%s", nu, gamma)

    windows = data.enroll_verify(data.split[args.split])
    log.info("Evaluating %d %s subjects.", len(windows), args.split)
    feats = extract_features(model, windows, device, args.batch_size)

    ocsvm = per_owner_ocsvm_eer(feats, nu=nu, gamma=gamma,
                                impostors_per_owner=args.impostors_per_owner,
                                seed=args.seed)
    cos = cosine_eer_rank1(feats, impostors_per_owner=args.impostors_per_owner,
                           seed=args.seed)

    bar = "=" * 68
    report = [
        "\n" + bar,
        f"CONTINAUTH RECREATION — {cfg.name}",
        bar,
        f"  feature_set        : {cfg.feature_set} ({cfg.n_channels} ch)",
        f"  window_mode        : {cfg.window_mode}   scaler: {cfg.scaler}",
        f"  test subjects      : {ocsvm['n_owners']}",
        bar,
        "  [ HIS metric: per-user OCSVM (rbf nu=%s gamma=%s%s) ]"
        % (nu, gamma, ", val-tuned" if tune else ""),
        f"    mean EER         : {ocsvm['mean_eer']*100:6.2f} %",
        "  [ Comparable: cosine gallery/probe (mean-enroll template) ]",
        f"    EER              : {cos['eer']*100:6.2f} %",
        f"    Rank-1           : {cos['rank1']*100:6.2f} %",
        bar,
    ]
    print("\n".join(report))

    result = {"config": cfg.name, "feature_set": cfg.feature_set,
              "n_channels": cfg.n_channels, "window_mode": cfg.window_mode,
              "ocsvm_nu": nu, "ocsvm_gamma": gamma, "ocsvm_tuned": bool(tune),
              "ocsvm_mean_eer": ocsvm["mean_eer"], "n_owners": ocsvm["n_owners"],
              "cosine_eer": cos["eer"], "cosine_rank1": cos["rank1"]}
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({**result, "per_owner_eer": ocsvm["per_owner"]}, f, indent=2, default=str)
        log.info("Wrote results -> %s", args.out_json)
    return result


def build_argparser():
    p = argparse.ArgumentParser(description="Evaluate ContinAuth Siamese + OCSVM (A/B/C)")
    p.add_argument("--config", choices=list(CONFIGS), required=True)
    p.add_argument("--feature_set", choices=list(FEATURE_SETS), default=None)
    p.add_argument("--mag_cols", nargs=3, default=None,
                   help="explicit magnetometer column names (X Y Z); default = use the "
                        "columns saved in the checkpoint, else auto-detect")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--split_file", default=None,
                   help="optional split_ids.json; if omitted, the same internal "
                        "subject-disjoint split is rebuilt (use the same --seed as train)")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--no_tune_ocsvm", dest="tune_ocsvm", action="store_false",
                   help="skip val grid-search; use the fixed cfg nu/gamma instead")
    p.set_defaults(tune_ocsvm=True)
    p.add_argument("--impostors_per_owner", type=int, default=2000)
    p.add_argument("--out_json", default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=712)
    return p


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = CONFIGS[args.config]
    if args.feature_set:
        cfg.feature_set = args.feature_set
        cfg.derive = args.feature_set in ("mine28", "mine28_mag")
    evaluate(cfg, args)


if __name__ == "__main__":
    main(build_argparser().parse_args())
