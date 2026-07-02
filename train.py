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
from torch.utils.data import DataLoader
from torch.optim import Adam

from .config import CONFIGS, FEATURE_SETS
from .data import ContinAuthData
from .model import build_base, SiameseNet, contrastive_loss
from .pairs import PairDataset


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

    base = build_base(args.variant, cfg.n_channels, X.shape[-1], cfg.filters, cfg.embed_dim)
    net = SiameseNet(base).to(device)
    opt = Adam(net.parameters(), lr=cfg.lr)

    ds = PairDataset(X, y, n_pairs=cfg.pairs_per_epoch, seed=args.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=False)

    net.train()
    for epoch in range(1, cfg.epochs + 1):
        ds.new_epoch()
        total, n = 0.0, 0
        for left, right, lab in loader:
            left, right, lab = left.to(device), right.to(device), lab.to(device)
            dist = net(left, right)
            loss = contrastive_loss(dist, lab, margin=cfg.margin)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(lab); n += len(lab)
        log.info("Epoch %03d/%d | contrastive %.4f", epoch, cfg.epochs, total / max(1, n))

    ckpt_path = Path(args.out or f"continauth/checkpoints/{cfg.name.split('_')[0].lower()}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config_key": args.config,
        "config_name": cfg.name,
        "variant": args.variant,
        "n_channels": cfg.n_channels,
        "window": int(X.shape[-1]),
        "filters": cfg.filters,
        "embed_dim": cfg.embed_dim,
        "base_state": net.base.state_dict(),
        "channel_mean": data.channel_mean,
        "channel_std": data.channel_std,
        "feature_set": cfg.feature_set,
        "load_cols": data.load_cols,           # concrete columns incl. resolved magnetometer
    }, ckpt_path)
    log.info("Saved checkpoint -> %s", ckpt_path)
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
        cfg.derive = (args.feature_set == "mine28")
    if args.epochs:
        cfg.epochs = args.epochs
    train(cfg, args)


if __name__ == "__main__":
    main(build_argparser().parse_args())
