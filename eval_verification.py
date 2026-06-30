"""
eval_verification.py
====================
Shared evaluation CLI for all models. Builds embeddings for the chosen model,
runs the ONE gallery/probe verification harness in verification.py, and prints
EER, FAR/FRR@EER and rank-1, with the within-session caveat.

Gallery/probe protocol: each subject yields one gallery template (mean enroll)
and one probe template (mean of all verify windows); scoring is template-vs-
template, not window-by-window.

  B0 — no checkpoint; runs cosine-to-enrolled-mean AND a per-user one-class SVM.
  B1/B2/M1 — load a checkpoint from train_verification.py; cosine scorer.

Examples
--------
  python eval_verification.py --model b0 --split test \
      --data_dirs dataset --cache_dir cache_verification

  python eval_verification.py --model b1 --split test \
      --checkpoint checkpoints_verification/b1.pt \
      --data_dirs dataset --cache_dir cache_verification
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from verification import (
    VerificationData, DEFAULT_DATA_DIRS, build_model,
    compute_deep_embeddings, compute_b0_embeddings, fit_b0_scaler,
    CosineScorer, OCSVMScorer, run_verification, format_report, result_to_dict,
    compute_score_matrix, bootstrap_metrics,
)


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("eval")

    data = VerificationData(args.cache_dir, args.split_file,
                            [Path(d) for d in args.data_dirs], seed=args.seed,
                            enroll_ratio=args.enroll_ratio, gap_seconds=args.gap_seconds)

    eval_ids = data.prepare_cache(data.split[args.split])
    if len(eval_ids) < 2:
        raise RuntimeError(f"Need >=2 cached {args.split} subjects; got {len(eval_ids)}.")
    log.info("Evaluating %d %s subjects.", len(eval_ids), args.split)

    config = {"model": args.model, "split": args.split, "seed": args.seed,
              "n_subjects": len(eval_ids), "enroll_ratio": args.enroll_ratio,
              "gap_seconds": args.gap_seconds,
              "impostors_per_user": args.impostors_per_user}

    results = []
    if args.model == "b0":
        train_ids = data.prepare_cache(data.split["train"])   # stats + scaler from TRAIN only
        data.fit_channel_stats(train_ids)
        scaler = fit_b0_scaler(data, train_ids, seed=args.seed)
        emb = compute_b0_embeddings(data, eval_ids, scaler)
        for scorer in (CosineScorer(), OCSVMScorer(nu=args.ocsvm_nu)):
            log.info("scorer: %s", scorer.name)
            results.append(run_verification(emb, scorer, args.seed, args.impostors_per_user))
    else:
        import torch
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for B1/B2/M1.")
        device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                              else "cpu")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = build_model(ckpt.get("model", args.model),
                            hidden_size=ckpt.get("hidden_size", args.hidden_size),
                            embed_dim=ckpt.get("embed_dim", 128)).to(device)
        model.load_state_dict(ckpt["network"])
        data.set_channel_stats(ckpt["channel_mean"], ckpt["channel_std"])
        emb = compute_deep_embeddings(model, data, eval_ids, device, args.batch_size)
        config["checkpoint"] = args.checkpoint
        results.append(run_verification(emb, CosineScorer(), args.seed, args.impostors_per_user))
        if args.ocsvm:
            results.append(run_verification(emb, OCSVMScorer(nu=args.ocsvm_nu),
                                            args.seed, args.impostors_per_user))

    # N x N probe-vs-gallery similarity matrix (diagonal = genuine).
    if args.matrix_out or args.plot_dir:
        sids, M = compute_score_matrix(emb, CosineScorer())
        log.info("Score matrix %s (mean diag=%.3f).", M.shape,
                 float(np.mean(np.diag(M))) if M.size else float("nan"))
        if args.matrix_out:
            Path(args.matrix_out).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.matrix_out, matrix=M, sids=np.array(sids, dtype=object))
            log.info("Wrote score matrix → %s", args.matrix_out)
        if args.plot_dir:
            try:
                from plot_metrics import plot_score_matrix
                outp = Path(args.plot_dir) / f"{args.model}_{args.split}_matrix.png"
                plot_score_matrix(M, sids, outp,
                                  title=f"{args.model.upper()} {args.split}: probe vs gallery")
                log.info("Wrote heatmap → %s", outp)
            except Exception as e:                                   # noqa: BLE001
                log.warning("heatmap skipped (%s)", e)

    report = format_report(args.model, results, config)

    # Subject-level bootstrap 95% CIs (cosine), so EER / Rank-1 gaps between
    # models can be judged against the noise. Resamples the test SUBJECTS.
    boot = None
    if args.bootstrap > 0:
        sids, M = compute_score_matrix(emb, CosineScorer())
        boot = bootstrap_metrics(M, n_boot=args.bootstrap, seed=args.seed)
        e, r = boot["eer"], boot["rank1"]
        report += (
            f"\n[ bootstrap 95% CI over {len(sids)} subjects, {boot['n_boot']} resamples ]"
            f"\n  EER    : {e[0]*100:6.2f} %   95% CI [{e[1]*100:.2f}, {e[2]*100:.2f}]"
            f"\n  Rank-1 : {r[0]*100:6.2f} %   95% CI [{r[1]*100:.2f}, {r[2]*100:.2f}]"
            f"\n\n{'=' * 72}")

    print("\n" + report)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"config": config,
                       "results": [result_to_dict(r) for r in results],
                       "bootstrap": boot},
                      f, indent=2, default=str)
        log.info("Wrote results → %s", args.out_json)


def build_argparser():
    p = argparse.ArgumentParser(description="Evaluate B0/B1/B2/M1 verification models")
    p.add_argument("--model",
                   choices=["b0", "b1", "b2", "m1", "m1a", "m1b", "m2", "m2a", "m2b",
                            "g2", "g2a", "g2b"], required=True)
    p.add_argument("--checkpoint", default=None, help="required for b1/b2/m1/m2/g2")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--data_dirs", nargs="+", default=list(DEFAULT_DATA_DIRS))
    p.add_argument("--split_file", default="split_ids.json")
    p.add_argument("--cache_dir", default="./cache_verification")
    p.add_argument("--out_json", default=None)
    p.add_argument("--matrix_out", default=None,
                   help="save the N×N probe-vs-gallery score matrix to this .npz "
                        "(arrays 'matrix' and 'sids')")
    p.add_argument("--plot_dir", default=None,
                   help="if set, save a <model>_<split>_matrix.png heatmap here")
    p.add_argument("--enroll_ratio", type=float, default=0.7,
                   help="fraction of the session (first, contiguous) used for enroll; rest = verify")
    p.add_argument("--gap_seconds", type=float, default=60.0,
                   help="total guard gap (default 1 min) dropped at the single enroll->verify boundary")
    p.add_argument("--impostors_per_user", type=int, default=2000,
                   help="cap on the number of OTHER subjects' probes scored against each gallery")
    p.add_argument("--ocsvm", action="store_true", help="also run OC-SVM for b1/b2/m1")
    p.add_argument("--ocsvm_nu", type=float, default=0.1)
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="subject-level bootstrap resamples for 95%% CIs on EER/Rank-1 "
                        "(0 disables)")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
