"""
continauth.pairs
================
Identity-balanced P x K batch sampling for the in-batch contrastive loss,
mirroring ContinAuth's balanced same/different-subject pairing (each batch has P
subjects x K windows, so positive pairs always exist).

`iter_pk_batches` yields (windows, labels) tensors straight from the in-memory
training array — no giant materialised pair frame.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch


def iter_pk_batches(X: np.ndarray, y: np.ndarray,
                    subjects_per_batch: int, windows_per_subject: int,
                    batches_per_epoch: int, seed: int = 712
                    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Yield `batches_per_epoch` identity-balanced batches.

    Each batch = P subjects (sampled without replacement) x K windows per subject
    (with replacement if a subject has fewer than K). Returns (x, y) as torch
    tensors on CPU; move to device in the training loop.
    """
    rng = np.random.default_rng(seed)
    labels = np.unique(y)
    rows_by_label = {int(l): np.where(y == l)[0] for l in labels}
    P = min(subjects_per_batch, len(labels))
    K = windows_per_subject

    for _ in range(batches_per_epoch):
        chosen = rng.choice(labels, size=P, replace=False)
        idx = []
        lab = []
        for l in chosen:
            rows = rows_by_label[int(l)]
            sel = rng.choice(rows, size=K, replace=len(rows) < K)
            idx.extend(int(r) for r in sel)
            lab.extend([int(l)] * K)
        idx = np.asarray(idx, dtype=np.int64)
        yield (torch.from_numpy(X[idx]), torch.tensor(lab, dtype=torch.long))
