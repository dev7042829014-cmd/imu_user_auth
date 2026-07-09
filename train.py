"""
continauth.train
================
Train the ContinAuth Siamese network on the TRAIN subjects for one of the three
experiment configurations (A / B / C), then save a checkpoint that evaluate.py
consumes.

Example
-------
  python -m continauth.train --config c \
      --data_dirs /path/to/dataset --split_file split_ids.json \
      --out continauth/checkpoints/c.pt --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from .config import CONFIGS, FEATURE_SETS
from .data import ContinAuthData
from .model import build_base, contrastive_loss_inbatch
from .pairs import iter_pk_batches

from .evaluate import evaluate
from types import SimpleNamespace

import shutil
import json

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg, args) -> dict:
    log = logging.getLogger("continauth.train")
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    set_seed(args.seed)

    data = ContinAuthData(cfg, args.data_dirs, args.split_file,
                          mag_cols=getattr(args, "mag_cols", None),
                          train_frac=args.train_frac, val_frac=args.val_frac,
                          seed=args.seed)
    if cfg.scaler == "zscore_global":
        data.fit_global_stats(data.split["train"])

    X, y = data.training_windows(data.split["train"])
    log.info("Training windows: %s over %d subjects.", X.shape, len(np.unique(y)))

    model = build_base(args.variant, cfg.n_channels, X.shape[-1],
                       cfg.filters, cfg.embed_dim).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)   # Adam 1e-3, as in ContinAuth

    ckpt_path = Path(args.out or f"continauth/checkpoints/{cfg.name.split('_')[0].lower()}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history = []

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total, n = 0.0, 0
        for xb, yb in iter_pk_batches(X, y, cfg.subjects_per_batch,
                                      cfg.windows_per_subject, cfg.batches_per_epoch,
                                      seed=args.seed + epoch):
            xb, yb = xb.to(device), yb.to(device)
            emb = model(xb)
            loss = contrastive_loss_inbatch(emb, yb, margin=cfg.margin)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); n += 1
        log.info("Epoch %03d/%d | contrastive %.4f", epoch, cfg.epochs, total / max(1, n))

        if epoch % 10 == 0:
            
            ckpt = ckpt_path.parent / f"{args.config}_epoch{epoch}.pt"

            torch.save({
                "config_key": args.config,
                "config_name": cfg.name,
                "variant": args.variant,
                "n_channels": cfg.n_channels,
                "window": int(X.shape[-1]),
                "filters": cfg.filters,
                "embed_dim": cfg.embed_dim,
                "base_state": model.state_dict(),
                "channel_mean": data.channel_mean,
                "channel_std": data.channel_std,
                "feature_set": cfg.feature_set,
                "scaler": cfg.scaler,
                "load_cols": data.load_cols, }, ckpt)

            result = evaluate( cfg, SimpleNamespace(config=args.config,
                    feature_set=None,
                    checkpoint=str(ckpt),
                    mag_cols=args.mag_cols,
                    data_dirs=args.data_dirs,
                    split_file=args.split_file,
                    split="val",
                    train_frac=args.train_frac,
                    val_frac=args.val_frac,
                    tune_ocsvm=True,
                    impostors_per_owner=2000,
                    out_json=None,
                    batch_size=256,
                    device=args.device,
                    seed=args.seed,),)
            
            history.append({
                "epoch": epoch,
                "train_loss": total / max(1, n),
                "eer": result["ocsvm_mean_eer"],
                "rank1": result["cosine_rank1"],
                "checkpoint": str(ckpt),})

            

            with open(
                ckpt_path.parent / f"{args.config}_history.json",
                "w"
            ) as f:
                json.dump(history, f, indent=2)

            log.info("Saved checkpoint %s", ckpt)
    
    

    torch.save({
        "config_key": args.config,
        "config_name": cfg.name,
        "variant": args.variant,
        "n_channels": cfg.n_channels,
        "window": int(X.shape[-1]),
        "filters": cfg.filters,
        "embed_dim": cfg.embed_dim,
        "base_state": model.state_dict(),
        "channel_mean": data.channel_mean,
        "channel_std": data.channel_std,
        "feature_set": cfg.feature_set,
        "scaler": cfg.scaler,                  # so eval preprocesses identically
        "load_cols": data.load_cols,           # concrete columns incl. resolved magnetometer
    }, ckpt_path)
    log.info("Saved checkpoint -> %s", ckpt_path)
    
    if not history:
        raise RuntimeError(
            "No validation checkpoints were created. "
            "Increase epochs or reduce the checkpoint interval."
        )

    best = min(
    history,
    key=lambda x: (
        x["eer"],
        -x["rank1"], ),)
    
    best_path = ckpt_path.parent / f"{args.config}_best.pt"
    
    shutil.copy(
    best["checkpoint"],
    best_path,)

    log.info(
        "Best checkpoint: epoch %d | EER %.4f | Rank1 %.4f",
        best["epoch"],
        best["eer"],
        best["rank1"],
    )


    return {"checkpoint": str(ckpt_path)}




def build_argparser():
    p = argparse.ArgumentParser(description="Train ContinAuth Siamese net (A/B/C)")
    p.add_argument("--config", choices=list(CONFIGS), required=True,
                   help="a=his feat+his window, b=his feat+my window, c=my 28ch+my window")
    p.add_argument("--feature_set", choices=list(FEATURE_SETS), default=None,
                   help="override the config's feature set (e.g. acc3 for his exact final model, "
                        "acc_gyro_mag9 for his acc+gyro+magnetometer input)")
    p.add_argument("--mag_cols", nargs=3, default=None,
                   help="explicit magnetometer column names (X Y Z); default = auto-detect")
    p.add_argument("--variant", choices=["fcn", "1d"], default="fcn")
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--split_file", default=None,
                   help="optional split_ids.json; if omitted, a subject-disjoint "
                        "split is built from the dataset folder (fully standalone)")
    p.add_argument("--train_frac", type=float, default=0.6,
                   help="train fraction for the internal split (when no --split_file)")
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="val fraction for the internal split (rest = test)")
    p.add_argument("--scaler", choices=["robust_subject", "zscore_global"], default=None,
                   help="force the SAME normalisation across configs (default keeps each "
                        "config's own: A=robust_subject, B/C/D=zscore_global)")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=None, help="override cfg.epochs")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=712)
    return p


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = CONFIGS[args.config]
    if args.feature_set:
        cfg.feature_set = args.feature_set
        cfg.derive = args.feature_set in ("mine28", "mine28_mag")
    if getattr(args, "scaler", None):
        cfg.scaler = args.scaler
    if args.epochs:
        cfg.epochs = args.epochs
    train(cfg, args)


if __name__ == "__main__":
    main(build_argparser().parse_args())
