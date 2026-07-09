"""
continauth.ocsvm_eval
=====================
His classifier + evaluation: a per-user one-class SVM on the Siamese deep
features, scored with Equal Error Rate.

  * `utils_eer`  — his exact EER (sklearn roc_curve + scipy brentq).
  * `per_owner_ocsvm_eer` — ContinAuth's protocol: for each test subject (owner),
        fit OneClassSVM(rbf, nu, gamma) on the owner's ENROLL deep features,
        score genuine (owner VERIFY) vs impostor (every other subject's VERIFY),
        take EER; report the mean over owners.
  * `cosine_eer_rank1` — the user's own gallery/probe protocol (mean-enroll
        template, cosine similarity), added so VERSION_B/C are directly
        comparable to the numbers their pipeline reports.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from sklearn.svm import OneClassSVM

log = logging.getLogger("continauth.eval")


def utils_eer(y_true, y_pred) -> float:
    """Equal Error Rate. y_true: 1=genuine/owner, 0=impostor. Higher y_pred=owner."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    try:
        return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))
    except Exception:                                                # noqa: BLE001
        # fallback: nearest FAR/FRR crossing
        fnr = 1 - tpr
        i = int(np.nanargmin(np.abs(fpr - fnr)))
        return float((fpr[i] + fnr[i]) / 2)


def _l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# His per-owner OCSVM EER
# ---------------------------------------------------------------------------

def per_owner_ocsvm_eer(feats: Dict[str, Dict[str, np.ndarray]],
                        nu: float = 0.165, gamma: float = 8.296,
                        impostors_per_owner: int = 2000,
                        seed: int = 712) -> dict:
    """
    feats[sid] = {'enroll': (Ne, D), 'verify': (Nv, D)} deep features.

    Returns {'mean_eer', 'per_owner': {sid: eer}, 'n_owners'}.
    """
    rng = np.random.default_rng(seed)
    sids = [s for s in sorted(feats)
            if len(feats[s]["enroll"]) > 0 and len(feats[s]["verify"]) > 0]
    per_owner = {}
    for owner in sids:
        clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        clf.fit(feats[owner]["enroll"])
        gen = clf.decision_function(feats[owner]["verify"])           # (Nv,)

        others = [s for s in sids if s != owner]
        if not others:
            continue
        imp_windows = np.concatenate([feats[o]["verify"] for o in others], 0)
        if impostors_per_owner and len(imp_windows) > impostors_per_owner:
            sel = rng.choice(len(imp_windows), size=impostors_per_owner, replace=False)
            imp_windows = imp_windows[sel]
        imp = clf.decision_function(imp_windows)

        y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
        y_pred = np.concatenate([gen, imp])
        per_owner[owner] = utils_eer(y_true, y_pred)

    eers = np.array([e for e in per_owner.values() if e == e])
    return {"mean_eer": float(eers.mean()) if len(eers) else float("nan"),
            "per_owner": per_owner, "n_owners": len(per_owner)}


def tune_ocsvm(feats_val: Dict[str, Dict[str, np.ndarray]],
               nu_grid, gamma_grid, impostors_per_owner: int = 2000,
               seed: int = 712):
    """
    Grid-search (nu, gamma) on VAL deep features, minimising the mean per-owner
    EER — ContinAuth's own OCSVM tuning objective (their fixed H-MOG nu/gamma do
    not transfer to this feature space). Returns (best_nu, best_gamma, best_eer).
    """
    best = (float("inf"), nu_grid[0], gamma_grid[0])
    for nu in nu_grid:
        for g in gamma_grid:
            try:
                r = per_owner_ocsvm_eer(feats_val, nu=nu, gamma=g,
                                        impostors_per_owner=impostors_per_owner, seed=seed)
            except Exception as e:                                    # noqa: BLE001
                log.warning("OCSVM nu=%s gamma=%s failed: %s", nu, g, e)
                continue
            e = r["mean_eer"]
            if e == e and e < best[0]:
                best = (e, nu, g)
    return best[1], best[2], best[0]


# ---------------------------------------------------------------------------
# Cosine gallery/probe (the user's protocol) — for direct comparability
# ---------------------------------------------------------------------------

def cosine_eer_rank1(feats: Dict[str, Dict[str, np.ndarray]],
                     impostors_per_owner: int = 2000, seed: int = 712) -> dict:
    """
    Gallery = L2-normalised mean ENROLL embedding; probe = mean VERIFY embedding.
    Genuine = cos(probe_i, gallery_i); impostor = cos(probe_i, gallery_j!=i).
    Returns {'eer', 'rank1', 'n_owners'}.
    """
    rng = np.random.default_rng(seed)
    sids = [s for s in sorted(feats)
            if len(feats[s]["enroll"]) > 0 and len(feats[s]["verify"]) > 0]
    if len(sids) < 2:
        return {"eer": float("nan"), "rank1": float("nan"), "n_owners": len(sids)}

    gallery = {s: _l2norm(feats[s]["enroll"].mean(0)) for s in sids}
    probe = {s: _l2norm(feats[s]["verify"].mean(0)) for s in sids}

    scores, is_gen = [], []
    for s in sids:
        scores.append(float(probe[s] @ gallery[s])); is_gen.append(1)
        others = [o for o in sids if o != s]
        if impostors_per_owner and len(others) > impostors_per_owner:
            others = list(rng.choice(others, size=impostors_per_owner, replace=False))
        for o in others:
            scores.append(float(probe[s] @ gallery[o])); is_gen.append(0)
    eer = utils_eer(np.array(is_gen), np.array(scores))

    # rank-1: is each probe nearest to its own gallery?
    G = np.stack([gallery[s] for s in sids])          # (N, D)
    P = np.stack([probe[s] for s in sids])            # (N, D)
    S = P @ G.T                                        # (N, N)
    rank1 = float((S.argmax(1) == np.arange(len(sids))).mean())
    return {"eer": eer, "rank1": rank1, "n_owners": len(sids)}
