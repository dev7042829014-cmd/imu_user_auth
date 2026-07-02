"""
continauth.pairs
================
Balanced positive/negative pair generation for the Siamese contrastive training,
mirroring ContinAuth's build_pairs (same-subject -> positive/label 1,
different-subject -> negative/label 0, 50/50 balanced).

Rather than materialising a giant pandas frame of pairs (his approach), we sample
pairs on the fly each epoch from the pooled training windows, which is
memory-cheap and equivalent for contrastive learning.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """
    Yields (left_window, right_window, label) triples.

    Each epoch is `n_pairs` samples, exactly half positive (same subject) and
    half negative (different subjects), drawn with a fixed seed for
    reproducibility.
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray,
                 n_pairs: int = 20000, seed: int = 712):
        self.windows = windows                       # (N, C, T) float32
        self.labels = labels                         # (N,) int subject id
        self.n_pairs = int(n_pairs)
        self.rng = np.random.default_rng(seed)
        self.rows_by_label: Dict[int, np.ndarray] = {}
        for lab in np.unique(labels):
            self.rows_by_label[int(lab)] = np.where(labels == lab)[0]
        self.unique_labels = np.array(sorted(self.rows_by_label))
        if len(self.unique_labels) < 2:
            raise ValueError("need >= 2 subjects to form negative pairs")
        self._resample()

    def _resample(self):
        rng = self.rng
        half = self.n_pairs // 2
        left = np.empty(self.n_pairs, dtype=np.int64)
        right = np.empty(self.n_pairs, dtype=np.int64)
        lab = np.empty(self.n_pairs, dtype=np.int64)

        # positive pairs
        for i in range(half):
            s = int(rng.choice(self.unique_labels))
            rows = self.rows_by_label[s]
            a, b = rng.choice(rows, size=2, replace=len(rows) < 2)
            left[i], right[i], lab[i] = a, b, 1

        # negative pairs
        for i in range(half, self.n_pairs):
            s1, s2 = rng.choice(self.unique_labels, size=2, replace=False)
            a = int(rng.choice(self.rows_by_label[int(s1)]))
            b = int(rng.choice(self.rows_by_label[int(s2)]))
            left[i], right[i], lab[i] = a, b, 0

        self._left, self._right, self._lab = left, right, lab

    def new_epoch(self):
        """Re-draw the pair set (call between epochs for fresh pairs)."""
        self._resample()

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx: int):
        l = torch.from_numpy(self.windows[self._left[idx]])
        r = torch.from_numpy(self.windows[self._right[idx]])
        y = torch.tensor(self._lab[idx], dtype=torch.float32)
        return l, r, y
