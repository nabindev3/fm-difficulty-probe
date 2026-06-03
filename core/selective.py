"""Risk-coverage / selective-prediction analysis — modality-agnostic.

Given per-test-item difficulty scores and binary error labels, sweep coverage,
abstaining on the highest-predicted-difficulty items. Reports AURC (area under
the risk-coverage curve, lower is better) against an oracle and a random
baseline. The headline "captures X% of oracle AURC" number lives here:

    fraction_of_oracle = (random_aurc - probe_aurc) / (random_aurc - oracle_aurc)

(i.e. how far the probe closes the gap from random down to the oracle).
"""
from __future__ import annotations

import numpy as np

# np.trapezoid is the numpy>=2.0 name; np.trapz is the <2.0 name. Support both so
# the AURC computation doesn't depend on the numpy major version.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def risk_coverage_curve(
    scores: np.ndarray,
    errors: np.ndarray,
    coverages: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Risk (mean error on retained items) vs coverage for one score vector.

    Items are answered in ascending predicted difficulty; at coverage c the
    easiest-predicted fraction c is retained.
    """
    scores = np.asarray(scores, dtype=float)
    errors = np.asarray(errors, dtype=float)
    n = len(errors)
    rng = np.random.default_rng(seed)

    order = np.argsort(scores)            # ascending predicted P(hard)
    sorted_errors = errors[order]
    curve, lo, hi = [], [], []
    for c in coverages:
        k = max(1, int(round(c * n)))
        kept = sorted_errors[:k]
        curve.append(float(kept.mean()))
        boots = [kept[rng.integers(0, k, k)].mean() for _ in range(n_bootstrap)]
        lo.append(float(np.percentile(boots, 2.5)))
        hi.append(float(np.percentile(boots, 97.5)))
    return {
        "curve": curve,
        "ci95_lower": lo,
        "ci95_upper": hi,
        "aurc": float(_trapz(curve, coverages)),
    }


def oracle_curve(errors: np.ndarray, coverages: np.ndarray) -> np.ndarray:
    """Best achievable: retain the truly-easiest (correct) items first."""
    errors = np.asarray(errors, dtype=float)
    n = len(errors)
    sorted_truth = np.sort(errors)
    return np.array(
        [sorted_truth[: max(1, int(round(c * n)))].mean() for c in coverages]
    )


def random_curve(
    errors: np.ndarray, coverages: np.ndarray, n_bootstrap: int = 2000, seed: int = 42
) -> np.ndarray:
    errors = np.asarray(errors, dtype=float)
    n = len(errors)
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_bootstrap):
        perm = errors[rng.permutation(n)]
        curves.append([perm[: max(1, int(round(c * n)))].mean() for c in coverages])
    return np.array(curves).mean(axis=0)


def selective_prediction(
    score_dict: dict[str, np.ndarray],
    errors: np.ndarray,
    coverages: np.ndarray | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Full risk-coverage analysis for a set of named score vectors.

    Returns oracle/random AURC, per-probe AURC, and the headline
    'fraction of oracle AURC captured' for each probe.
    """
    errors = np.asarray(errors, dtype=float)
    if coverages is None:
        coverages = np.round(np.arange(0.10, 1.001, 0.05), 4)
    n = len(errors)
    mean_err = float(errors.mean())

    o_curve = oracle_curve(errors, coverages)
    r_curve = random_curve(errors, coverages, n_bootstrap=n_bootstrap, seed=seed)
    oracle_aurc = float(_trapz(o_curve, coverages))
    random_aurc = float(_trapz(r_curve, coverages))

    probes = {}
    for name, scores in score_dict.items():
        rc = risk_coverage_curve(scores, errors, coverages, n_bootstrap, seed)
        gap = (random_aurc - oracle_aurc) or 1e-12
        rc["fraction_of_oracle_aurc"] = (random_aurc - rc["aurc"]) / gap
        probes[name] = rc

    return {
        "n_test": n,
        "mean_error_no_abstention": mean_err,
        "coverages": coverages.tolist(),
        "oracle_curve": o_curve.tolist(),
        "oracle_aurc": oracle_aurc,
        "random_curve": r_curve.tolist(),
        "random_aurc": random_aurc,
        "probes": probes,
    }
