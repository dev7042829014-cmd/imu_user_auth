"""
m2_model.py
===========
M2 = "slow / fast (tremor) split" verification embedding.

This is a STANDALONE add-on. It does not modify verification.py / apw_network.py /
train_verification.py / eval_verification.py — it only imports from them. The
training and evaluation entry points (train_m2.py, eval_m2.py) register M2 with
the existing pipeline at runtime, so the original files stay byte-for-byte
unchanged.

Idea (plain words): every 15 s window is separated into
  * SLOW  = the smooth, voluntary motion (low frequency), and
  * FAST  = what's left over = the tremor / jitter (high frequency).
Both streams are stacked (28 + 28 = 56 channels) and fed to the SAME SE-CNN
encoder used by M1. The FAST stream is the always-present, activity-independent
identity signal we want the model to lean on.

The split is done with a simple moving-average low-pass (easy to read and to
change later): slow = local average over `smooth_kernel` samples; fast = x - slow.
"""

from __future__ import annotations

from apw_network import N_INPUT_CHANNELS


def _make_m2(hidden_size, embed_dim, smooth_kernel: int = 11, se: bool = True):
    """
    M2 = "slow / fast (tremor) split" model.

    Idea (plain words): every 15 s window is separated into
      * SLOW  = the smooth, voluntary motion (low frequency), and
      * FAST  = what's left over = the tremor / jitter (high frequency).
    Both streams are stacked (28 + 28 = 56 channels) and fed to the SAME SE-CNN
    encoder used by M1. The FAST stream is the always-present, activity-independent
    identity signal we want the model to lean on.

    The split is done with a simple moving-average low-pass (easy to read and to
    change later): slow = local average over `smooth_kernel` samples; fast = x - slow.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from apw_network import APW_Net

    def moving_average(x, k):
        # Average each sample with its k neighbours (centred). Reflect-pad the ends
        # so the output is the same length as the input.
        pad = k // 2
        x_padded = F.pad(x, (pad, pad), mode="reflect")
        return F.avg_pool1d(x_padded, kernel_size=k, stride=1)

    class M2Embedding(APW_Net):
        """SE-CNN over the stacked (slow ‖ fast) streams → 128-d L2 embedding."""

        def __init__(self):
            # 56 input channels = 28 slow + 28 fast (the two streams stacked).
            # NOTE: APW_Net's SE flag is `use_se` (not `se`).
            super().__init__(n_channels=2 * N_INPUT_CHANNELS,
                             hidden_size=hidden_size, use_se=se)
            del self.age_head
            del self.gender_head
            self.embed_dim = embed_dim
            self.smooth_kernel = smooth_kernel
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden_size, embed_dim))

        def split_slow_fast(self, x):
            slow = moving_average(x, self.smooth_kernel)   # voluntary motion (low freq)
            fast = x - slow                                # tremor / jitter (high freq)
            return torch.cat([slow, fast], dim=1)          # (B, 28, T) -> (B, 56, T)

        def forward(self, x):
            x = self.split_slow_fast(x)
            # Same multi-branch fusion as M1, now over the 56-channel input.
            a = self.enc_A(x)
            b = self.enc_B(a)
            c = self.enc_C(b)
            branches = torch.cat([
                self.pool_fine(a),
                self.pool_mid(b),
                self.pool_coarse(c),
                self.spectral(x),
                self.stats_enc(x),
                self.band_enc(x),
            ], dim=-1)
            fused = self.fusion(branches)
            return F.normalize(self.proj(fused), dim=-1)

    return M2Embedding()


# ---------------------------------------------------------------------------
# Registration helper — wires M2 into the existing pipeline WITHOUT editing it.
# ---------------------------------------------------------------------------

_M2_NAMES = ("m2", "m2a", "m2b")
# m2a = M2 + SupCon, m2b = M2 + ArcFace (matches the m1a/m1b convention).
_M2_DEFAULT_LOSS = {"m2a": "supcon", "m2b": "arcface"}


def register(default_embed_dim: int = 128):
    """
    Monkey-patch `build_model` in the verification / train / eval modules so the
    names m2, m2a, m2b resolve to the M2 encoder, and extend the per-model default
    loss map (m2a->supcon, m2b->arcface). Everything else falls through to the
    user's original build_model untouched.
    """
    import verification

    orig_build = verification.build_model

    def build_model(name, hidden_size: int = 256, embed_dim: int = default_embed_dim):
        if str(name).lower() in _M2_NAMES:
            return _make_m2(hidden_size, embed_dim)
        return orig_build(name, hidden_size=hidden_size, embed_dim=embed_dim)

    verification.build_model = build_model

    # train_verification / eval_verification did `from verification import build_model`,
    # so they hold their own reference — patch those names too if imported.
    import importlib
    for mod_name in ("train_verification", "eval_verification"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, "build_model"):
            mod.build_model = build_model
        if hasattr(mod, "_MODEL_DEFAULT_LOSS"):
            mod._MODEL_DEFAULT_LOSS = {**mod._MODEL_DEFAULT_LOSS, **_M2_DEFAULT_LOSS}

    return build_model


def extend_model_choices(parser):
    """Add m2/m2a/m2b to the argparse `--model` choices on an existing parser."""
    for action in parser._actions:
        if getattr(action, "dest", None) == "model" and action.choices is not None:
            extra = [n for n in _M2_NAMES if n not in action.choices]
            action.choices = list(action.choices) + extra
    return parser
