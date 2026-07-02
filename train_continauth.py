"""
train_continauth.py
===================
Train the ContinAuth / Centeno Siamese-FCN baseline (contrastive loss) on OUR
Apple-Watch dataset, reusing verification.py's data layer, P×K identity sampler
and enroll/verify partition. Kept separate from train_verification.py so the
baseline never entangles with our own models.

Examples
--------
  # his way — accel + gyro + magnetometer (9ch)
  python train_continauth.py --channels accel_gyro_mag \
      --data_dirs dataset --split_file split_ids.json \
      --cache_dir cache_verification --epochs 40 --device cuda

  # our way — full 28 derived channels
  python train_continauth.py --channels full \
      --data_dirs dataset --cache_dir cache_verification --epochs 40 --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from verification import (
    VerificationData, DEFAULT_DATA_DIRS,
    compute_deep_embeddings, CosineScorer, run_verification,
)
from train_verification import (
    WindowDataset, IdentityBalancedSampler, collate, set_seed, write_history,
)
from continauth_model import build_continauth_model, contrastive_loss, CHANNEL_PRESETS


def make_data(args):
    """VerificationData for our 28-ch cache, or ContinAuthMagData (9-ch accel+gyro
    +mag, its OWN cache) for the faithful 'his way' variant."""
    if args.channels == "accel_gyro_mag":
        from continauth_data import ContinAuthMagData
        mag_cols = args.mag_cols.split(",") if args.mag_cols else None
        cache = f"{args.cache_dir}_mag9"                    # separate cache (avoids collisions)
        return ContinAuthMagData(cache, args.split_file,
                                 [Path(d) for d in args.data_dirs], seed=args.seed,
                                 enroll_ratio=args.enroll_ratio, gap_seconds=args.gap_seconds,
                                 mag_cols=mag_cols)
    return VerificationData(args.cache_dir, args.split_file,
                            [Path(d) for d in args.data_dirs], seed=args.seed,
                            enroll_ratio=args.enroll_ratio, gap_seconds=args.gap_seconds)


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("train_continauth")
    set_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    log.info("Device %s | ContinAuth Siamese-FCN | channels=%s | seed %d",
             device, args.channels, args.seed)

    data = make_data(args)
    train_ids = data.prepare_cache(data.split["train"])
    if not train_ids:
        raise RuntimeError("No train subjects cached — check --data_dirs / --split_file.")
    data.fit_channel_stats(train_ids)

    ds = WindowDataset(data, train_ids)
    log.info("Dataset: %d identities, %d windows.", ds.n_classes, len(ds))
    data._mmap.clear()
    sampler = IdentityBalancedSampler(ds, args.subjects_per_batch,
                                      args.windows_per_subject,
                                      args.batches_per_epoch, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                        num_workers=args.workers,
                        persistent_workers=(args.workers > 0))

    model = build_continauth_model(channels=args.channels,
                                   embed_dim=args.embed_dim).to(device)
    log.info("Trainable parameters: %d",
             sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = Adam(model.parameters(), lr=args.lr)   # Adam 1e-3, as in ContinAuth

    ckpt_path = Path(args.out or f"./checkpoints_continauth/continauth_{args.channels}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history, history_path = [], ckpt_path.with_name(ckpt_path.stem + "_history.csv")
    best_eer = float("inf")

    log.info("Training %d epochs (%d batches/epoch, batch=%d, margin=%.2f)...",
             args.epochs, args.batches_per_epoch,
             args.subjects_per_batch * args.windows_per_subject, args.margin)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            emb = model(x)
            loss = contrastive_loss(emb, y, margin=args.margin)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(loader))
        msg = f"Epoch {epoch:03d}/{args.epochs} | contrastive {train_loss:.4f}"

        val_eer = float("nan"); val_rank1 = float("nan")
        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            try:
                val_ids = data.prepare_cache(data.split["val"])
                emb = compute_deep_embeddings(model, data, val_ids, device, args.batch_size)
                res = run_verification(emb, CosineScorer(), seed=args.seed,
                                       impostors_per_user=args.impostors_per_user)
                val_eer, val_rank1 = res.eer, res.rank1
                msg += f" | val EER {val_eer*100:.2f}% rank1 {val_rank1*100:.1f}%"
            except Exception as e:                                   # noqa: BLE001
                log.warning("val eval skipped: %s", e)

        # ContinAuth selects on EER — keep that for the baseline.
        is_best = val_eer == val_eer and val_eer < best_eer
        if is_best:
            best_eer = val_eer; msg += "  ★ best"
        log.info(msg)

        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "lr": args.lr,
                        "val_eer": ("" if val_eer != val_eer else round(val_eer, 6)),
                        "val_rank1": ("" if val_rank1 != val_rank1 else round(val_rank1, 6))})
        write_history(history_path, history)

        ckpt = {"model": "continauth", "channels": args.channels,
                "mag_cols": getattr(data, "mag_cols", None),
                "network": model.state_dict(),
                "channel_mean": data.channel_mean, "channel_std": data.channel_std,
                "embed_dim": args.embed_dim, "epoch": epoch,
                "val_eer": val_eer, "val_rank1": val_rank1, "args": vars(args)}
        torch.save(ckpt, ckpt_path)
        if is_best:
            torch.save(ckpt, ckpt_path.with_suffix(".best.pt"))

    log.info("Done. Best val EER: %.2f%% → %s",
             best_eer * 100 if best_eer < float("inf") else float("nan"), ckpt_path)


def build_argparser():
    p = argparse.ArgumentParser(description="Train ContinAuth Siamese-FCN baseline")
    p.add_argument("--channels", choices=list(CHANNEL_PRESETS), default="accel_gyro_mag",
                   help="accel_gyro_mag = his way (accel+gyro+MAGNETOMETER, 9ch); "
                        "full = our way (28 derived channels)")
    p.add_argument("--mag_cols", default=None,
                   help="comma-separated magnetometer column names for accel_gyro_mag "
                        "(auto-detected from the CSV header if omitted)")
    p.add_argument("--data_dirs", nargs="+", default=list(DEFAULT_DATA_DIRS))
    p.add_argument("--split_file", default="split_ids.json")
    p.add_argument("--cache_dir", default="./cache_verification")
    p.add_argument("--out", default=None,
                   help="ckpt path; default ./checkpoints_continauth/continauth_<channels>.pt")
    p.add_argument("--epochs", type=int, default=40)          # ContinAuth ~40 epochs
    p.add_argument("--margin", type=float, default=1.0)       # contrastive margin
    p.add_argument("--subjects_per_batch", type=int, default=32)
    p.add_argument("--windows_per_subject", type=int, default=8)
    p.add_argument("--batches_per_epoch", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)          # Adam 1e-3
    p.add_argument("--embed_dim", type=int, default=32)       # FCN feature size
    p.add_argument("--enroll_ratio", type=float, default=0.7)
    p.add_argument("--gap_seconds", type=float, default=60.0)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--impostors_per_user", type=int, default=2000)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
