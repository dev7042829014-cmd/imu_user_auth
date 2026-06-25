"""
plot_metrics.py
===============
Plotting helpers for the verification pipeline:

  1. Training curves   — training loss vs epoch AND validation EER vs epoch,
                         read from the per-epoch history CSV that
                         train_verification.py writes next to each checkpoint.
  2. Score matrix      — the N x N probe-vs-gallery similarity heatmap
                         (diagonal = genuine), from verification.compute_score_matrix.

Note on "training vs testing loss": in open-set verification the val/test
identities are DISJOINT from train, so there is no comparable supervised loss on
them (ArcFace has no prototypes for unseen people; SupCon would need val
positives). The meaningful "test-side" curve is therefore the validation EER,
which is what we plot against the training loss.

CLI
---
  # curves from a training-history CSV
  python plot_metrics.py curves --history checkpoints_verification/m1a_history.csv \
      --out_dir plots

  # heatmap from a saved score matrix (.npz with arrays 'matrix' and 'sids')
  python plot_metrics.py matrix --npz plots/m1a_test_scores.npz --out plots/m1a_matrix.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless / server-safe backend
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Training-history curves
# ---------------------------------------------------------------------------

def read_history(history_csv) -> dict:
    """Read a train_verification history CSV into column arrays."""
    epochs, train_loss, val_eer, lr = [], [], [], []
    with open(history_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            lr.append(float(row.get("lr", "nan")))
            v = row.get("val_eer", "")
            val_eer.append(float(v) if v not in ("", "nan", "None") else float("nan"))
    return {"epoch": np.array(epochs), "train_loss": np.array(train_loss),
            "val_eer": np.array(val_eer), "lr": np.array(lr)}


def plot_training_curves(history_csv, out_dir=None, title_prefix: str = "") -> List[Path]:
    """
    Two figures saved as PNG:
      <stem>_loss.png — training loss vs epoch
      <stem>_eer.png  — validation EER (%) vs epoch (NaNs skipped)
    Returns the written paths.
    """
    history_csv = Path(history_csv)
    h = read_history(history_csv)
    out_dir = Path(out_dir) if out_dir else history_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = history_csv.stem.replace("_history", "")
    written: List[Path] = []

    # --- training loss ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["epoch"], h["train_loss"], color="tab:blue", lw=1.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title(f"{title_prefix}Training loss".strip())
    ax.grid(True, alpha=0.3)
    p = out_dir / f"{stem}_loss.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    # --- validation EER ---
    mask = ~np.isnan(h["val_eer"])
    if mask.any():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(h["epoch"][mask], h["val_eer"][mask] * 100.0,
                color="tab:red", marker="o", ms=3, lw=1.6)
        best_i = int(np.nanargmin(h["val_eer"]))
        ax.axhline(h["val_eer"][best_i] * 100.0, color="grey", ls="--", lw=1,
                   label=f"best {h['val_eer'][best_i]*100:.2f}% @ ep{h['epoch'][best_i]}")
        ax.set_xlabel("epoch"); ax.set_ylabel("validation EER (%)")
        ax.set_title(f"{title_prefix}Validation EER".strip())
        ax.grid(True, alpha=0.3); ax.legend()
        p = out_dir / f"{stem}_eer.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    # --- combined (loss + EER on twin axes) ---
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(h["epoch"], h["train_loss"], color="tab:blue", lw=1.6, label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue"); ax1.grid(True, alpha=0.3)
    if mask.any():
        ax2 = ax1.twinx()
        ax2.plot(h["epoch"][mask], h["val_eer"][mask] * 100.0,
                 color="tab:red", marker="o", ms=3, lw=1.6, label="val EER")
        ax2.set_ylabel("validation EER (%)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title(f"{title_prefix}Loss & EER".strip())
    p = out_dir / f"{stem}_loss_eer.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    return written


# ---------------------------------------------------------------------------
# 2. Probe-vs-gallery score matrix heatmap
# ---------------------------------------------------------------------------

def plot_score_matrix(matrix: np.ndarray, sids: Optional[Sequence[str]] = None,
                      out_path="score_matrix.png", title: str = "Probe vs gallery similarity",
                      max_ticks: int = 30) -> Path:
    """
    Render the N x N probe(row) vs gallery(col) score matrix as a heatmap.
    The diagonal (genuine pairs) should stand out bright against impostors.
    """
    matrix = np.asarray(matrix, dtype=float)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="score (cosine similarity)")
    ax.set_xlabel("gallery subject"); ax.set_ylabel("probe subject")

    diag = float(np.mean(np.diag(matrix))) if n else float("nan")
    off = (matrix.sum() - np.trace(matrix)) / max(1, n * n - n) if n else float("nan")
    ax.set_title(f"{title}\n(N={n}, mean diag={diag:.3f}, mean off-diag={off:.3f})")

    if sids is not None and n and n <= max_ticks:
        ax.set_xticks(range(n)); ax.set_xticklabels(sids, rotation=90, fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(sids, fontsize=6)

    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot training curves / score matrix")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("curves", help="training loss + val EER vs epoch")
    pc.add_argument("--history", required=True, help="path to <model>_history.csv")
    pc.add_argument("--out_dir", default=None)
    pc.add_argument("--title_prefix", default="")

    pm = sub.add_parser("matrix", help="N x N probe-vs-gallery heatmap")
    pm.add_argument("--npz", required=True, help=".npz with 'matrix' (and optional 'sids')")
    pm.add_argument("--out", default="score_matrix.png")
    pm.add_argument("--title", default="Probe vs gallery similarity")

    args = p.parse_args()
    if args.cmd == "curves":
        for w in plot_training_curves(args.history, args.out_dir, args.title_prefix):
            print("wrote", w)
    elif args.cmd == "matrix":
        d = np.load(args.npz, allow_pickle=True)
        sids = list(d["sids"]) if "sids" in d.files else None
        print("wrote", plot_score_matrix(d["matrix"], sids, args.out, args.title))


if __name__ == "__main__":
    main()
