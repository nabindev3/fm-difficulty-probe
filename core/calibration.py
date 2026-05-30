"""Calibration metrics + Platt/isotonic recalibration — modality-agnostic.

Two responsibilities, both promoted from duplicated code in the two repos:

  1. compute_calibration / ECE / Brier / reliability points  (eval/calibration.py)
  2. recalibrate_oof: 5-fold OUT-OF-FOLD Platt + isotonic on the TRAIN split,
     applied to the test split, so there is zero test leakage in the calibrator
     fit.                                                     (eval/recalibrate.py)

All inputs are numpy arrays; plotting is intentionally left to the caller so the
core stays import-light and unit-testable.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


def compute_ece_brier(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(ece), float(np.mean((p - y) ** 2))


def reliability_points(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """(mean_pred, mean_actual) per occupied bin — feed straight to a plot."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
    return xs, ys


def recalibrate_oof(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Fit Platt + isotonic on K-fold OOF train predictions, evaluate on test.

    The probe coefficients are scale-fixed by `C` (the caller passes the C the
    ladder already chose for this rung), so this measures recalibration only.
    Returns raw/platt/isotonic ECE, Brier, AUROC plus the recalibrated test-set
    probabilities for downstream cascade routing.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)

    # 5-fold OOF predictions on the train split.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p_oof = np.zeros(len(y_train), dtype=float)
    for tr_idx, te_idx in kf.split(X_train):
        clf = LogisticRegression(
            penalty="l1", solver="liblinear",
            class_weight="balanced", max_iter=2000, C=C,
        )
        clf.fit(X_train[tr_idx], y_train[tr_idx])
        p_oof[te_idx] = clf.predict_proba(X_train[te_idx])[:, 1]

    # Final probe on full train -> raw test predictions.
    base = LogisticRegression(
        penalty="l1", solver="liblinear",
        class_weight="balanced", max_iter=2000, C=C,
    )
    base.fit(X_train, y_train)
    p_te_raw = base.predict_proba(X_test)[:, 1]

    # Isotonic (non-monotone freedom) + Platt (monotone, AUROC-preserving).
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_oof, y_train)
    p_te_iso = iso.transform(p_te_raw)

    platt = LogisticRegression(C=1e6)
    platt.fit(p_oof.reshape(-1, 1), y_train)
    p_te_platt = platt.predict_proba(p_te_raw.reshape(-1, 1))[:, 1]

    has_both = len(np.unique(y_test)) > 1
    out = {}
    for tag, p in (("raw", p_te_raw), ("platt", p_te_platt), ("isotonic", p_te_iso)):
        ece, brier = compute_ece_brier(y_test, p)
        out[tag] = {
            "ece": ece,
            "brier": brier,
            "auroc": float(roc_auc_score(y_test, p)) if has_both else None,
        }
    out["_test_probs"] = {
        "raw": p_te_raw,
        "platt": p_te_platt,
        "isotonic": p_te_iso,
    }
    return out
