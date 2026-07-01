"""
train_verification.py
=====================
Train a metric-learning embedding network (B1 = CNN, B2 = GRU) for open-set user
verification, using SUBJECT IDENTITY as the label and supervised-contrastive loss.

Only B1/B2 need training; B0 (classical) is training-free — evaluate it directly
with eval_verification.py. Subject identities are read from split_ids.json; only
the TRAIN identities are used here (val/test people are never seen).

Example
-------
  python train_verification.py --model b1 \
      --data_dirs dataset --split_file split_ids.json \
      --cache_dir cache_verification --epochs 50 \
      --subjects_per_batch 16 --windows_per_subject 8 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from verification import (
    VerificationData, DEFAULT_DATA_DIRS, EMBED_DIM_DEFAULT,
    build_model, supcon_loss, build_arcface_loss, build_margin_loss, MARGIN_LOSSES,
    build_pairwise_loss, PAIRWISE_LOSSES,
    compute_deep_embeddings, CosineScorer, run_verification,
)


# Model -> default loss. ?a = SupCon (the workhorse), ?b = a margin loss.
# m1a/m1b legacy: M1 + SupCon / M1 + ArcFace. m2/g2 follow the same convention.
# For a bare "m1"/"m2"/"g2" (or b1/b2) the loss falls back to --loss (default supcon).
# The 'b' default stays plain "arcface" so existing m1b/m2b numbers keep their
# meaning; pass --loss subcenter or --loss adacos to use the better margin losses.
_MODEL_DEFAULT_LOSS = {"m1a": "supcon", "m1b": "arcface",
                       "m2a": "supcon", "m2b": "arcface",
                       "g2a": "supcon", "g2b": "arcface"}


def write_history(path, history):
    """Dump the per-epoch metric history to CSV (epoch, train_loss, lr, val_eer,
    val_rank1). val_rank1 is the model-selection metric (val_eer kept for plots)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "lr",
                                          "val_eer", "val_rank1"])
        w.writeheader()
        for row in history:
            w.writerow(row)


def set_arcface_margin_deg(arcface, deg: float):
    """Set the current ArcFace angular margin (degrees) for margin warm-up.
    pytorch_metric_learning's ArcFaceLoss stores `.margin` in RADIANS (it converts
    the constructor's degrees at init) and reads it each forward, so we convert
    here. (The disabled built-in exposed set_margin_deg; kept as a fallback path.)"""
    if hasattr(arcface, "set_margin_deg"):
        arcface.set_margin_deg(deg)
    elif hasattr(arcface, "margin"):
        import math
        arcface.margin = math.radians(float(deg))


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# --- identity-labelled window dataset + balanced P×K sampler ----------------

class WindowDataset(Dataset):
    """Each item: (window (28,300) tensor, identity label int)."""

    def __init__(self, data: VerificationData, subject_ids: List[str]):
        self.data = data
        self.index: List[Tuple[str, int]] = data.all_windows_index(subject_ids)
        identities = sorted({sid for sid, _ in self.index})
        self.label_of: Dict[str, int] = {s: i for i, s in enumerate(identities)}
        self.n_classes = len(identities)
        self.rows_by_label: Dict[int, List[int]] = {i: [] for i in range(self.n_classes)}
        for row, (sid, _) in enumerate(self.index):
            self.rows_by_label[self.label_of[sid]].append(row)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, row: int):
        sid, start = self.index[row]
        w = self.data.get_windows(sid, np.array([start], dtype=np.int64))[0]
        return torch.from_numpy(w), self.label_of[sid]


class IdentityBalancedSampler(Sampler):
    """Each batch = P subjects × K windows, so SupCon always has positive pairs."""

    def __init__(self, ds: WindowDataset, subjects_per_batch, windows_per_subject,
                 num_batches, seed=42):
        self.ds = ds
        self.P, self.K = subjects_per_batch, windows_per_subject
        self.num_batches = num_batches
        self.rng = np.random.default_rng(seed)
        self.labels = [l for l, rows in ds.rows_by_label.items() if rows]

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for lab in self.rng.choice(self.labels, size=min(self.P, len(self.labels)),
                                       replace=False):
                rows = self.ds.rows_by_label[lab]
                sel = self.rng.choice(rows, size=self.K, replace=len(rows) < self.K)
                batch += [int(r) for r in sel]
            yield batch


def collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch], dtype=torch.long))


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("train")
    set_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    log.info("Device %s | model %s | seed %d", device, args.model, args.seed)

    data = VerificationData(args.cache_dir, args.split_file,
                            [Path(d) for d in args.data_dirs], seed=args.seed,
                            enroll_ratio=args.enroll_ratio, gap_seconds=args.gap_seconds)

    train_ids = data.prepare_cache(data.split["train"])
    if not train_ids:
        raise RuntimeError("No train subjects cached — check --data_dirs / --split_file.")
    data.fit_channel_stats(train_ids)

    ds = WindowDataset(data, train_ids)
    log.info("Dataset: %d identities, %d windows.", ds.n_classes, len(ds))
    data._mmap.clear()                                   # workers reopen lazily
    sampler = IdentityBalancedSampler(ds, args.subjects_per_batch,
                                      args.windows_per_subject,
                                      args.batches_per_epoch, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                        num_workers=args.workers,
                        persistent_workers=(args.workers > 0))

    model = build_model(args.model, hidden_size=args.hidden_size,
                        embed_dim=args.embed_dim).to(device)

    # Resolve which loss to train with: an explicit --loss wins; otherwise the
    # model name decides (m?a->supcon, m?b->arcface); otherwise supcon.
    loss_type = args.loss or _MODEL_DEFAULT_LOSS.get(args.model.lower()) or "supcon"
    log.info("Loss: %s", loss_type)

    # The margin losses (arcface / subcenter / adacos) own a trainable class
    # matrix, so it joins the optimizer. SupCon and the pairwise losses (triplet /
    # multisim / circle) have NO parameters.
    arcface = None                       # kept name: holds the margin-loss module
    if loss_type in MARGIN_LOSSES:
        arcface = build_margin_loss(loss_type, ds.n_classes, args.embed_dim,
                                    scale=args.arcface_scale,
                                    margin_deg=args.arcface_margin,
                                    sub_centers=args.sub_centers).to(device)

    # Pairwise open-set losses: a parameter-free callable loss_fn(emb, labels).
    pairwise_fn = build_pairwise_loss(loss_type) if loss_type in PAIRWISE_LOSSES else None

    trainable = list(model.parameters()) + (list(arcface.parameters()) if arcface else [])
    log.info("Trainable parameters: %d",
             sum(p.numel() for p in trainable if p.requires_grad))

    # Param groups. The ArcFace class-prototype matrix gets its OWN group with
    # NO weight decay (decaying prototypes shrinks the angular margins and hurts)
    # and a separate, usually higher LR than the backbone.
    param_groups = [{"params": model.parameters(), "lr": args.lr,
                     "weight_decay": args.weight_decay}]
    if arcface is not None:
        param_groups.append({"params": arcface.parameters(), "lr": args.arcface_lr,
                             "weight_decay": 0.0})
        log.info("%s head: lr=%.1e, weight_decay=0%s.", loss_type, args.arcface_lr,
                 "" if loss_type == "adacos"
                 else f", margin warm-up over {args.arcface_margin_warmup_epochs} epoch(s)")
    optimizer = AdamW(param_groups)

    # Optional linear LR warm-up, then cosine anneal. Warm-up stabilises ArcFace
    # (the angular-margin penalty is large before the backbone has organised the
    # embedding space) and is harmless for SupCon.
    warm = max(0, int(args.warmup_epochs))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warm),
                               eta_min=args.lr * 1e-3)
    if warm > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warm)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warm])
    else:
        scheduler = cosine

    # --out defaults to a model-specific name so B2 never overwrites B1.
    ckpt_path = Path(args.out or f"./checkpoints_verification/{args.model}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # Checkpoint selection metric: --select_by {eer,rank1}. EER -> lower is better,
    # rank1 -> higher is better. (rank1 avoids the saturated-EER argmin trap, but
    # EER selection is back as the default per request; both are logged regardless.)
    sel_eer = (args.select_by == "eer")
    best_metric = float("inf") if sel_eer else -1.0

    # Per-epoch metric history (training loss + val EER) → CSV for plotting.
    history = []
    history_path = ckpt_path.with_name(ckpt_path.stem + "_history.csv")

    log.info("Training %d epochs (%d batches/epoch, batch=%d)...", args.epochs,
             args.batches_per_epoch, args.subjects_per_batch * args.windows_per_subject)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if arcface is not None:
            arcface.train()
            # Margin warm-up: ramp 0 -> target over the first N epochs so the
            # network is not fighting the full angular penalty from a random init.
            # (AdaCos has no margin / auto-scales, so there is nothing to warm up.)
            if loss_type != "adacos":
                if args.arcface_margin_warmup_epochs > 0:
                    frac = min(1.0, epoch / float(args.arcface_margin_warmup_epochs))
                else:
                    frac = 1.0
                set_arcface_margin_deg(arcface, args.arcface_margin * frac)
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            emb = model(x)
            if arcface is not None:
                loss = arcface(emb, y)
            elif pairwise_fn is not None:
                loss = pairwise_fn(emb, y)
            else:
                loss = supcon_loss(emb, y, args.temperature)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(loader))
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        msg = (f"Epoch {epoch:03d}/{args.epochs} | {loss_type} {train_loss:.4f} "
               f"| LR {cur_lr:.2e}")

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

        # Select on the chosen metric (eer: lower better; rank1: higher better).
        cur = val_eer if sel_eer else val_rank1
        is_best = cur == cur and (cur < best_metric if sel_eer else cur > best_metric)
        if is_best:
            best_metric = cur; msg += "  ★ best"
        log.info(msg)

        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "lr": cur_lr,
                        "val_eer": ("" if val_eer != val_eer else round(val_eer, 6)),
                        "val_rank1": ("" if val_rank1 != val_rank1 else round(val_rank1, 6))})
        write_history(history_path, history)

        ckpt = {"model": args.model, "loss": loss_type, "network": model.state_dict(),
                "arcface": arcface.state_dict() if arcface is not None else None,
                "channel_mean": data.channel_mean, "channel_std": data.channel_std,
                "hidden_size": args.hidden_size, "embed_dim": args.embed_dim,
                "epoch": epoch, "val_eer": val_eer, "val_rank1": val_rank1,
                "args": vars(args)}
        torch.save(ckpt, ckpt_path)
        if is_best:
            torch.save(ckpt, ckpt_path.with_suffix(".best.pt"))

    log.info("Done. Best val %s: %.4f → %s", args.select_by,
             best_metric if abs(best_metric) != float("inf") and best_metric >= 0
             else float("nan"), ckpt_path)
    log.info("Per-epoch history → %s  (plot: python plot_metrics.py curves --history %s)",
             history_path, history_path)


def build_argparser():
    p = argparse.ArgumentParser(description="Train B1/B2/M1 verification embedding")
    p.add_argument("--model",
                   choices=["b1", "b2", "m1", "m1a", "m1b", "m2", "m2a", "m2b",
                            "g2", "g2a", "g2b"], default="b1",
                   help="b1/b2 baselines; m1 = CNN+SE; m2 = slow/fast split SE-CNN; "
                        "g2 = frequency-hybrid GRU. ?a=SupCon, ?b=ArcFace (the model "
                        "name sets the default loss; --loss overrides it).")
    p.add_argument("--loss",
                   choices=["supcon", "arcface", "subcenter", "adacos",
                            "triplet", "multisim", "circle"], default=None,
                   help="metric-learning loss; default follows the model name "
                        "(?a->supcon, ?b->arcface). Open-set recommended: supcon, "
                        "adacos, triplet (FaceNet), multisim, circle. "
                        "subcenter/arcface kept for the comparison baseline.")
    p.add_argument("--select_by", choices=["eer", "rank1"], default="eer",
                   help="val metric used to pick the .best.pt checkpoint")
    p.add_argument("--arcface_scale", type=float, default=30.0,
                   help="ArcFace s (logit scale). 30 suits ~10^3 identities; 64 is the "
                        "large-scale face default and tends to over-sharpen here.")
    p.add_argument("--arcface_margin", type=float, default=20.0,
                   help="ArcFace TARGET angular margin in DEGREES (warmed up from 0)")
    p.add_argument("--arcface_lr", type=float, default=1e-3,
                   help="separate LR for the ArcFace prototype matrix (no weight decay)")
    p.add_argument("--arcface_margin_warmup_epochs", type=int, default=15,
                   help="ramp the ArcFace margin 0 -> target over this many epochs (0 = off)")
    p.add_argument("--sub_centers", type=int, default=3,
                   help="K sub-centers per identity for --loss subcenter (1 = plain ArcFace)")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="linear LR warm-up epochs before cosine anneal (0 = off)")
    p.add_argument("--data_dirs", nargs="+", default=list(DEFAULT_DATA_DIRS))
    p.add_argument("--split_file", default="split_ids.json")
    p.add_argument("--cache_dir", default="./cache_verification")
    p.add_argument("--out", default=None,
                   help="checkpoint path; default ./checkpoints_verification/<model>.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--subjects_per_batch", type=int, default=32, help="P in P×K batch")
    p.add_argument("--windows_per_subject", type=int, default=8, help="K in P×K batch")
    p.add_argument("--batches_per_epoch", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM_DEFAULT)
    p.add_argument("--enroll_ratio", type=float, default=0.7,
                   help="fraction of the session (first, contiguous) used for enroll; rest = verify")
    p.add_argument("--gap_seconds", type=float, default=60.0,
                   help="total guard gap (default 1 min) dropped at the single enroll->verify boundary")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=256, help="inference batch for val eval")
    p.add_argument("--eval_every", type=int, default=10, help="0 disables val EER")
    p.add_argument("--impostors_per_user", type=int, default=2000)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
