"""
continauth.model
================
Faithful PyTorch re-implementation of ContinAuth's Siamese networks
(Buech 2019, chapter-5-5-siamese-cnn.ipynb).  His code is Keras/TF; this is the
same architecture in PyTorch so it drops into the user's torch environment.

Original FCN base (VALID-FCN-ROBUST), Keras, input (T, C):
    Conv1D(32, k=8, padding=same) -> BatchNorm -> ReLU -> Dropout(0.1)
    Conv1D(64, k=5, padding=same) -> BatchNorm -> ReLU -> Dropout(0.1)
    Conv1D(32, k=3, padding=same) -> BatchNorm -> ReLU
    GlobalAveragePooling1D
    Dense(32, activation="sigmoid")            # 32-d deep feature (embedding)

Siamese head:
    euclidean distance between the two branch embeddings,
    trained with contrastive loss (Hadsell 2006), margin = 1.

PyTorch conv is channels-first (B, C, T), which matches how `data.py` emits
windows, so no transpose is needed.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNBase(nn.Module):
    """His fully-convolutional base network -> 32-d sigmoid embedding."""

    def __init__(self, n_channels: int, filters: List[int] = (32, 64, 32),
                 embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        f0, f1, f2 = filters
        self.conv1 = nn.Conv1d(n_channels, f0, kernel_size=8, padding="same")
        self.bn1 = nn.BatchNorm1d(f0)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(f0, f1, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(f1)
        self.drop2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(f1, f2, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm1d(f2)
        self.dense = nn.Linear(f2, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, T)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=-1)                       # GlobalAveragePooling1D
        x = torch.sigmoid(self.dense(x))         # his Dense(32, sigmoid)
        return x


class CNN1DBase(nn.Module):
    """His "1d" variant (elu convs + maxpool + flatten). Kept for completeness."""

    def __init__(self, n_channels: int, window: int,
                 filters: List[int] = (32, 64, 128, 64)):
        super().__init__()
        f0, f1, f2, f3 = filters
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, f0, 7, padding="same"), nn.ELU(), nn.MaxPool1d(2),
            nn.Conv1d(f0, f1, 5, padding="same"), nn.ELU(), nn.MaxPool1d(2),
            nn.Conv1d(f1, f2, 3, padding="same"), nn.ELU(), nn.MaxPool1d(2),
            nn.Conv1d(f2, f3, 3, padding="same"), nn.ELU(), nn.MaxPool1d(2),
        )
        self.embed_dim = f3 * (window // 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.net(x), 1)


def build_base(variant: str, n_channels: int, window: int,
               filters: List[int], embed_dim: int) -> nn.Module:
    if variant == "fcn":
        return FCNBase(n_channels, list(filters), embed_dim)
    if variant == "1d":
        return CNN1DBase(n_channels, window, list(filters))
    raise ValueError(f"unknown base variant {variant!r}")


class SiameseNet(nn.Module):
    """Shared-weight twin of `base`; forward returns the pairwise euclidean dist."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.embed_dim = getattr(base, "embed_dim", None)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        el, er = self.base(left), self.base(right)
        # euclidean distance (his k_euclidean_dist), keepdim removed
        return torch.sqrt(torch.sum((el - er) ** 2, dim=-1) + 1e-9)


def contrastive_loss(dist: torch.Tensor, label: torch.Tensor,
                     margin: float = 1.0) -> torch.Tensor:
    """
    Contrastive loss (Hadsell-Chopra-LeCun 2006), his `k_contrastive_loss`:
        L = mean( y * d^2 + (1-y) * max(margin - d, 0)^2 )
    label = 1 for positive (same subject), 0 for negative (different subjects).
    """
    label = label.float()
    pos = label * dist.pow(2)
    neg = (1.0 - label) * torch.clamp(margin - dist, min=0.0).pow(2)
    return (pos + neg).mean()
