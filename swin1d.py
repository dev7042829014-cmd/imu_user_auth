"""
swin1d.py
=========
Self-contained Swin Transformer 1-D encoder for Apple-Watch IMU verification.

Why a transformer here
----------------------
B1/M1/M2 are convolutional; B2/G2 are recurrent. This adds the third major
family — hierarchical windowed self-attention — as a drop-in embedding encoder
that honours the SAME contract every other model in verification.py obeys:

        (B, 28, 300) window  ->  128-d L2-normalised embedding

so it trains with the existing SupCon / ArcFace losses (train_verification.py)
and is scored by the existing gallery/probe harness (eval_verification.py) with
zero pipeline changes.

Swin in one paragraph (1-D port of Liu et al. 2021 / yukara-ikemiya's 1-D repo)
-------------------------------------------------------------------------------
1. PATCH EMBED    — a strided Conv1d splits the 300-sample window into P-sample
                    patches and embeds each into a token  (300 -> 60 tokens).
2. W-MSA          — self-attention computed only INSIDE local time windows of
                    `window_size` tokens: O(n) not O(n^2), and encodes locality.
3. SW-MSA         — the next block SHIFTS the window boundaries by window//2 so
                    information crosses between neighbouring windows. Alternating
                    W-MSA / SW-MSA is the core Swin mechanism.
4. PATCH MERGING  — concatenates neighbouring token pairs and halves the token
                    count while doubling the channel width -> a coarse-to-fine
                    hierarchy (60 -> 30 -> 15 tokens), like the enc_A/B/C pyramid.

Default config (locked with the data owner; ~21 M params)
---------------------------------------------------------
    patch_size = 5        window_size = 15      embed_dim = 128
    depths     = [2,2,6]  n_heads     = [4,8,16]  (head-dim 32 at every stage)
    mlp_ratio  = 4        drop_path   = 0.2       dropout = 0.1

    Token / width trace (patch 5 on 300 samples):
        patch embed : 300 -> 60 tokens @ 128
        stage 0 (x2): 60 tokens @ 128   (60/15 = 4 windows)
        merge       : 30 tokens @ 256
        stage 1 (x2): 30 tokens @ 256   (30/15 = 2 windows)
        merge       : 15 tokens @ 512
        stage 2 (x6): 15 tokens @ 512   (15/15 = 1 window -> global attention)
        pool + proj : -> 128-d L2 embedding

Note: at the last stage tokens == window_size, so attention is global and the
window shift is disabled (exactly as the original Swin behaves in its final
stage). The `embed_dim=256` "swin_large" setting is kept for an over-capacity
ablation but is NOT the default (it overfits this dataset).
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from apw_network import N_INPUT_CHANNELS, WINDOW_SAMPLES


# ─── DropPath (stochastic depth) ──────────────────────────────────────────────

class DropPath(nn.Module):
    """Per-sample stochastic depth: randomly zeroes the residual branch of a
    block during training (Huang et al. 2016). The single strongest regulariser
    for small-data transformers, which is exactly our regime."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        # broadcast mask over all dims except batch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * mask.floor_() / keep


# ─── Window partition helpers (1-D) ───────────────────────────────────────────

def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, L, C) -> (num_windows*B, ws, C). Assumes L % ws == 0."""
    B, L, C = x.shape
    x = x.view(B, L // ws, ws, C)
    return x.reshape(-1, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, L: int) -> torch.Tensor:
    """(num_windows*B, ws, C) -> (B, L, C)."""
    C = windows.shape[-1]
    B = windows.shape[0] // (L // ws)
    x = windows.view(B, L // ws, ws, C)
    return x.reshape(B, L, C)


# ─── Windowed multi-head self-attention with relative position bias ───────────

class WindowAttention1D(nn.Module):
    """Multi-head self-attention within a 1-D window, with a learned relative-
    position bias table of size (2*ws-1, n_heads) — the 1-D analogue of Swin's
    2-D relative bias. An optional attention mask implements shifted windows."""

    def __init__(self, dim: int, window_size: int, n_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5

        # relative position bias: one bias per (head, relative offset)
        self.rel_bias = nn.Parameter(torch.zeros(2 * window_size - 1, n_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        coords = torch.arange(window_size)
        rel = coords[:, None] - coords[None, :] + (window_size - 1)  # (ws, ws) in [0, 2ws-2]
        self.register_buffer("rel_index", rel.long(), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x : (nW*B, ws, C)
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                     # (3, Bn, heads, ws, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)        # (Bn, heads, ws, ws)

        bias = self.rel_bias[self.rel_index.view(-1)].view(N, N, -1)  # (ws, ws, heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)     # broadcast over batch

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.n_heads, N, N) + \
                mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        return self.proj_drop(self.proj(out))


# ─── Swin block: (S)W-MSA + MLP, both residual with pre-norm ──────────────────

class SwinBlock1D(nn.Module):
    def __init__(self, dim: int, input_len: int, n_heads: int, window_size: int,
                 shift: int, mlp_ratio: float = 4.0, drop: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        # If the sequence is no longer than one window, attention is global and
        # shifting is a no-op — mirror the reference implementations and disable it.
        if input_len <= window_size:
            window_size = input_len
            shift = 0
        self.input_len = input_len
        self.ws = window_size
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention1D(dim, window_size, n_heads,
                                      attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

        # Precompute the SW-MSA attention mask (blocks attention across the
        # cyclic-shift wrap boundary), if shifting.
        if shift > 0:
            mask = self._build_mask(input_len, window_size, shift)
            self.register_buffer("attn_mask", mask, persistent=False)
        else:
            self.attn_mask = None

    @staticmethod
    def _build_mask(L: int, ws: int, shift: int) -> torch.Tensor:
        # Label tokens by segment created by the cyclic shift; tokens in
        # different segments must not attend to each other within a window.
        img = torch.zeros(1, L, 1)
        slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        for c, s in enumerate(slices):
            img[:, s, :] = c
        win = window_partition(img, ws).squeeze(-1)          # (nW, ws)
        diff = win.unsqueeze(1) - win.unsqueeze(2)           # (nW, ws, ws)
        mask = diff.masked_fill(diff != 0, -100.0).masked_fill(diff == 0, 0.0)
        return mask                                          # (nW, ws, ws)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, C)
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift > 0:
            x = torch.roll(x, shifts=-self.shift, dims=1)
        xw = window_partition(x, self.ws)                    # (nW*B, ws, C)
        xw = self.attn(xw, self.attn_mask)
        x = window_reverse(xw, self.ws, L)
        if self.shift > 0:
            x = torch.roll(x, shifts=self.shift, dims=1)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ─── Patch merging: halve tokens, double channels ─────────────────────────────

class PatchMerging1D(nn.Module):
    """Concatenate every adjacent token pair (C -> 2C) then Linear 2C -> 2C.
    Halves sequence length; the hierarchical downsample between stages."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        if L % 2 == 1:                       # pad odd length by repeating last token
            x = torch.cat([x, x[:, -1:, :]], dim=1)
            L += 1
        # reshape (not view): the preceding shifted block's torch.roll leaves x
        # non-contiguous, and merging neighbouring pairs crosses stride subspaces.
        x = x.reshape(B, L // 2, 2 * C)      # merge neighbouring pairs
        return self.reduction(self.norm(x))


# ─── Attention pooling over the final token sequence ──────────────────────────

class _TokenAttnPool(nn.Module):
    """Learned scalar weight per token, then weighted sum -> one vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, L, C) -> (B, C)
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


# ─── Swin-1D backbone (produces a pooled feature vector) ──────────────────────

class SwinTransformer1D(nn.Module):
    """Hierarchical windowed-attention encoder over an IMU window.

    Returns a pooled feature vector of width `embed_dim * 2**(len(depths)-1)`
    (e.g. 128 -> 512 with 3 stages). The S1/S2 wrappers project it to the 128-d
    L2-normalised embedding the verification harness expects.
    """

    def __init__(self,
                 n_channels: int = N_INPUT_CHANNELS,
                 seq_len: int = WINDOW_SAMPLES,
                 patch_size: int = 5,
                 embed_dim: int = 128,
                 depths: Sequence[int] = (2, 2, 6),
                 n_heads: Sequence[int] = (4, 8, 16),
                 window_size: int = 15,
                 mlp_ratio: float = 4.0,
                 drop: float = 0.1,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.2):
        super().__init__()
        assert len(depths) == len(n_heads), "depths and n_heads must align"
        assert seq_len % patch_size == 0, "patch_size must divide seq_len"
        self.n_stages = len(depths)

        # Patch embedding: strided Conv1d over channels-first input.
        self.patch_embed = nn.Conv1d(n_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(drop)
        n_tokens = seq_len // patch_size

        # Linear drop-path schedule across all blocks (0 -> drop_path).
        total_blocks = sum(depths)
        dpr: List[float] = torch.linspace(0, drop_path, total_blocks).tolist()

        self.stages = nn.ModuleList()
        self.mergers = nn.ModuleList()
        dim = embed_dim
        cur_len = n_tokens
        blk = 0
        for si, depth in enumerate(depths):
            blocks = nn.ModuleList()
            for j in range(depth):
                shift = 0 if (j % 2 == 0) else window_size // 2
                blocks.append(SwinBlock1D(
                    dim=dim, input_len=cur_len, n_heads=n_heads[si],
                    window_size=window_size, shift=shift, mlp_ratio=mlp_ratio,
                    drop=drop, attn_drop=attn_drop, drop_path=dpr[blk]))
                blk += 1
            self.stages.append(blocks)
            # Patch-merge between stages (not after the last one).
            if si < self.n_stages - 1:
                self.mergers.append(PatchMerging1D(dim))
                dim *= 2
                cur_len = (cur_len + 1) // 2
            else:
                self.mergers.append(None)

        self.num_features = dim
        self.norm = nn.LayerNorm(dim)
        self.pool = _TokenAttnPool(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, T) -> patch embed -> (B, L, D)
        x = self.patch_embed(x).transpose(1, 2)              # (B, n_tokens, embed_dim)
        x = self.pos_drop(x)
        for si in range(self.n_stages):
            for block in self.stages[si]:
                x = block(x)
            if self.mergers[si] is not None:
                x = self.mergers[si](x)
        x = self.norm(x)
        return self.pool(x)                                  # (B, num_features)


# ─── S1: pure Swin-1D embedding model ─────────────────────────────────────────

class Swin1DEmbedding(nn.Module):
    """S1 — Swin-1D backbone -> 128-d L2 embedding. The transformer counterpart
    of B1/M1 (pure temporal encoder, no hand-built frequency branches)."""

    def __init__(self, embed_dim_out: int = 128, hidden_size: int = 128,
                 depths: Sequence[int] = (2, 2, 6),
                 n_heads: Sequence[int] = (4, 8, 16),
                 patch_size: int = 5, window_size: int = 15,
                 mlp_ratio: float = 4.0, drop: float = 0.1, drop_path: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim_out
        self.backbone = SwinTransformer1D(
            embed_dim=hidden_size, depths=depths, n_heads=n_heads,
            patch_size=patch_size, window_size=window_size,
            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        f = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(f, f), nn.GELU(), nn.Dropout(drop),
            nn.Linear(f, embed_dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                              # (B, num_features)
        return F.normalize(self.proj(feat), dim=-1)          # (B, 128)


# ─── S2: Swin-1D backbone + frequency-domain fusion ───────────────────────────

class Swin1DFusionEmbedding(nn.Module):
    """S2 — Swin-1D temporal backbone fused with the SAME hand-built frequency
    encoders the CNN/GRU models use (spectral shape + physiological band power).

    Rationale: tremor identity lives in the frequency domain, which a patch-
    embedded time-domain transformer cannot see directly. This mirrors exactly
    how G2 augments the GRU (attention pool + spectral + band-power fusion).
    """

    def __init__(self, embed_dim_out: int = 128, hidden_size: int = 128,
                 depths: Sequence[int] = (2, 2, 6),
                 n_heads: Sequence[int] = (4, 8, 16),
                 patch_size: int = 5, window_size: int = 15,
                 mlp_ratio: float = 4.0, drop: float = 0.1, drop_path: float = 0.2):
        super().__init__()
        from apw_network import _SpectralEncoder, _BandPowerEncoder
        self.embed_dim = embed_dim_out
        self.backbone = SwinTransformer1D(
            embed_dim=hidden_size, depths=depths, n_heads=n_heads,
            patch_size=patch_size, window_size=window_size,
            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        f = self.backbone.num_features
        spec_size = hidden_size // 2
        band_size = 32
        self.spectral = _SpectralEncoder(N_INPUT_CHANNELS, spec_size)
        self.band_enc = _BandPowerEncoder(n_dyn_channels=6, out_size=band_size)
        fused_in = f + spec_size + band_size
        self.proj = nn.Sequential(
            nn.Linear(fused_in, f), nn.GELU(), nn.Dropout(drop),
            nn.Linear(f, embed_dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat(
            [self.backbone(x), self.spectral(x), self.band_enc(x)], dim=-1)
        return F.normalize(self.proj(feat), dim=-1)


# ─── Factory (called from verification.build_model) ───────────────────────────

def make_swin(name: str, hidden_size: int = 128,
              embed_dim: int = 128, large: bool = False):
    """Build an S-family model. `large` swaps the internal width to 256 for the
    over-capacity ablation (kept for completeness; overfits this dataset)."""
    name = name.lower()
    hs = 256 if large else hidden_size
    common = dict(embed_dim_out=embed_dim, hidden_size=hs)
    if name in ("s1", "s1a", "s1b"):
        return Swin1DEmbedding(**common)
    if name in ("s2", "s2a", "s2b"):
        return Swin1DFusionEmbedding(**common)
    raise ValueError(f"make_swin: '{name}' is not an S-family model (use s1/s2).")
