"""Metrics, implemented directly -- scikit-learn is not in the dependency set.

Every function takes 1-D numpy arrays and ignores NaN targets, which is how the
label record encodes "unknown" (a search that hit its budget).
"""

from __future__ import annotations

import numpy as np


def _clean(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = ~np.isnan(y)
    return y[m], p[m]


def accuracy(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> float:
    y, p = _clean(y, p)
    if not len(y):
        return float("nan")
    return float(((p >= thr) == (y >= 0.5)).mean())


def balanced_accuracy(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> float:
    """Mean per-class recall: the number a majority-class predictor cannot game."""
    y, p = _clean(y, p)
    if not len(y):
        return float("nan")
    hat = p >= thr
    recalls = []
    for cls in (0, 1):
        m = (y >= 0.5) == bool(cls)
        if m.sum():
            recalls.append(float((hat[m] == bool(cls)).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U form, with ties handled by average ranks."""
    y, p = _clean(y, p)
    pos = y >= 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_f1(y: np.ndarray, hat: np.ndarray, n_classes: int) -> float:
    y, hat = _clean(y.astype(float), hat.astype(float))
    y, hat = y.astype(int), hat.astype(int)
    f1s = []
    for c in range(n_classes):
        tp = int(((hat == c) & (y == c)).sum())
        fp = int(((hat == c) & (y != c)).sum())
        fn = int(((hat != c) & (y == c)).sum())
        if tp + fn == 0:
            continue                      # class absent from this split
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def multiclass_accuracy(y: np.ndarray, hat: np.ndarray) -> float:
    y, hat = _clean(y.astype(float), hat.astype(float))
    return float((y == hat).mean()) if len(y) else float("nan")


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _clean(y, p)
    if len(y) < 3:
        return float("nan")
    ry, rp = _rankdata(y), _rankdata(p)
    ry = ry - ry.mean()
    rp = rp - rp.mean()
    den = float(np.sqrt((ry**2).sum() * (rp**2).sum()))
    return float((ry * rp).sum() / den) if den else float("nan")


def mae(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _clean(y, p)
    return float(np.abs(y - p).mean()) if len(y) else float("nan")
