"""
continauth_model.py
===================
Faithful PyTorch port of the ContinAuth / Centeno et al. (2018) baseline:
a *Siamese Fully-Convolutional Network* trained with contrastive loss, whose
deep features are then verified per-user with a One-Class SVM.

This is a COMPARISON BASELINE (a competitor method), kept deliberately separate
from our own models (B0/B1/B2/M1/M2/G2). It reuses our data layer and evaluation
harness (verification.py) but nothing here modifies our files.

Reference
---------
  * dynobo/ContinAuth (MSc thesis, 2019) — reproduces:
  * Centeno et al., "Mobile Based Continuous Authentication Using Deep Features"
    (2018): Siamese CNN feature extractor + per-user OCSVM.

Original setting (H-MOG, smartphone): accel+gyro(+mag) @ 25 Hz, 5 s windows
(125 samples), FCN filters [32, 64, 32], contrastive loss (margin=1), Adam 1e-3.

Adaptation to OUR data (Apple Watch): we keep the ARCHITECTURE and LOSS exactly,
but feed OUR windows (300 samples = 15 s @ 20 Hz) so the row is comparable to our
own models. Two channel variants:
  * "accel_gyro_mag" (9 ch: accel XYZ + gyro XYZ + magnetometer XYZ) — the
    faithful "his way" input (served from continauth_data.ContinAuthMagData);
  * "full" (28 ch) — our full derived-feature version.
"""

from __future__ import annotations

from apw_network import N_INPUT_CHANNELS

# Channel presets over the cached 28-channel array (see verification.ALL_CHANNEL_NAMES).
#   0-2 userAccel XYZ | 3-5 gyro XYZ | 6-8 gravity XYZ | 9-10 roll,pitch | 11-27 derived
CHANNEL_PRESETS = {
    # his way (faithful): accel + gyro + MAGNETOMETER = 9 channels. This preset
    # indexes the 9-channel array served by continauth_data.ContinAuthMagData
    # (NOT the 28-ch cache), so it is an identity slice [0..8].
    "accel_gyro_mag": list(range(0, 9)),
    # our way: the full 28 channels (raw + derived) from our standard cache.
    "full": list(range(0, N_INPUT_CHANNELS)),
}


def build_continauth_model(channels: str = "accel_gyro", embed_dim: int = 32,
                           normalize: bool = True):
    """Build the Siamese FCN encoder over a channel subset of our 28-ch windows.

    filters [32, 64, 32], kernels [8, 5, 3] (the Wang-2017 FCN that Centeno used),
    each: Conv1d -> BatchNorm -> ReLU, then global average pool over time -> a
    32-d deep feature. The window is sliced to the chosen channels INSIDE forward,
    so our data layer (which always serves 28-ch windows) is untouched.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if channels not in CHANNEL_PRESETS:
        raise ValueError(f"channels must be one of {list(CHANNEL_PRESETS)}")
    ch_idx = CHANNEL_PRESETS[channels]

    class SiameseFCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.channels = channels
            self.embed_dim = embed_dim
            self.normalize = normalize
            self.register_buffer("ch_idx", torch.tensor(ch_idx, dtype=torch.long))
            c = len(ch_idx)

            def block(cin, cout, k):
                # 'same'-ish padding (k//2); GAP downstream tolerates ±1 length.
                return nn.Sequential(
                    nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(cout), nn.ReLU())

            self.conv = nn.Sequential(
                block(c, 32, 8), block(32, 64, 5), block(64, 32, 3))
            # FCN outputs 32-d after global average pool; optional linear to embed_dim.
            self.head = nn.Identity() if embed_dim == 32 else nn.Linear(32, embed_dim)

        def forward(self, x):                       # x : (B, 28, 300)
            x = x.index_select(1, self.ch_idx)      # -> (B, c, 300)
            h = self.conv(x)                        # (B, 32, ~300)
            h = h.mean(dim=-1)                      # global average pool -> (B, 32)
            z = self.head(h)
            return F.normalize(z, dim=-1) if self.normalize else z

    return SiameseFCN()


def contrastive_loss(embeddings, labels, margin: float = 1.0):
    """Contrastive loss (Hadsell/Chopra) over in-batch pairs.

    For a pair (i, j): positive (same identity) -> pull together (d^2);
    negative (different) -> push apart until distance >= margin (relu(margin-d)^2).
    Our P×K identity-balanced batches supply the pairs that ContinAuth formed
    within sessions. Embeddings are L2-normalised, so Euclidean distance lies in
    [0, 2] and margin=1 (their value) is meaningful.

    The positive and negative terms are averaged SEPARATELY (each over its own
    pair count) and summed. This mirrors ContinAuth's *balanced* positive/negative
    pair sampling: otherwise the ~P·K·(K-1) positives are swamped by the ~(P·K)^2
    easy negatives (mostly already satisfied → zero gradient), and the encoder
    barely learns.
    """
    import torch

    d = torch.cdist(embeddings, embeddings)                 # (B, B) Euclidean
    same = (labels.view(-1, 1) == labels.view(1, -1)).float()
    off = 1.0 - torch.eye(d.size(0), device=d.device)       # exclude self-pairs
    pos = same * off
    neg = (1.0 - same) * off
    l_pos = (pos * d.pow(2)).sum() / pos.sum().clamp(min=1.0)
    l_neg = (neg * torch.clamp(margin - d, min=0.0).pow(2)).sum() / neg.sum().clamp(min=1.0)
    return l_pos + l_neg
