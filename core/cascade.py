"""Difficulty-routed cost-quality cascade — modality-agnostic.

Route each item between a cheap and an expensive model using a difficulty score
as the routing signal; sweep the threshold to trace a (mean cost, mean error)
Pareto curve. Compared against always-cheap, always-expensive, random routing,
and an oracle. Reports how many probe-driven points strictly dominate the
cheap<->expensive interpolation line.

LLM:  cheap = Pythia-410M, expensive = Pythia-2.8B, error = 0/1 correctness.
TSFM: cheap = Chronos-small, expensive = Chronos-base, error = (normalized) CRPS.
Both reduce to: per-item err_cheap, err_expensive, a routing score, and two costs.
"""
from __future__ import annotations

import numpy as np


def _route_threshold(scores, tau, err_cheap, err_exp, cost_cheap, cost_exp):
    to_exp = scores >= tau
    final = np.where(to_exp, err_exp, err_cheap)
    cost = np.where(to_exp, cost_exp, cost_cheap)
    return float(final.mean()), float(cost.mean()), float(to_exp.mean())


def _route_random(err_cheap, err_exp, cost_cheap, cost_exp, n_trials=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(err_cheap)
    fractions = np.linspace(0.0, 1.0, 21)
    curve = []
    for f in fractions:
        k = int(round(f * n))
        errors, costs = [], []
        for _ in range(n_trials):
            idx = rng.choice(n, size=k, replace=False) if k > 0 else np.array([], int)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            errors.append(np.where(mask, err_exp, err_cheap).mean())
            costs.append(np.where(mask, cost_exp, cost_cheap).mean())
        curve.append((float(np.mean(costs)), float(np.mean(errors)), float(f)))
    return curve


def _route_oracle(err_cheap, err_exp, cost_cheap, cost_exp):
    # Route to the expensive model where it helps most (largest err_cheap - err_exp).
    gap = err_cheap - err_exp
    order = np.argsort(-gap)
    n = len(gap)
    fractions = np.linspace(0.0, 1.0, 21)
    curve = []
    for f in fractions:
        k = int(round(f * n))
        mask = np.zeros(n, dtype=bool)
        mask[order[:k]] = True
        err = np.where(mask, err_exp, err_cheap).mean()
        cost = np.where(mask, cost_exp, cost_cheap).mean()
        curve.append((float(cost), float(err), float(f)))
    return curve


def _probe_curve(scores, err_cheap, err_exp, cost_cheap, cost_exp, n_taus=41):
    taus = np.linspace(0.0, 1.0, n_taus)
    return [
        {"tau": float(t),
         **dict(zip(("mean_error", "mean_cost", "frac_to_exp"),
                    _route_threshold(scores, t, err_cheap, err_exp, cost_cheap, cost_exp)))}
        for t in taus
    ]


def _dominating_points(pts, cheap_anchor, exp_anchor):
    c0, y0 = cheap_anchor
    c1, y1 = exp_anchor
    dom = []
    for p in pts:
        c = p["mean_cost"]
        if not (c0 < c < c1):
            continue
        t = (c - c0) / (c1 - c0 + 1e-12)
        y_line = y0 + t * (y1 - y0)
        if p["mean_error"] < y_line - 1e-9:
            dom.append(p)
    return dom


def cascade(
    score_dict: dict[str, np.ndarray],
    err_cheap: np.ndarray,
    err_exp: np.ndarray,
    cost_cheap: float = 1.0,
    cost_exp: float = 5.0,
    n_random_trials: int = 500,
    seed: int = 42,
) -> dict:
    """Full cascade analysis for a set of named routing-score vectors."""
    err_cheap = np.asarray(err_cheap, dtype=float)
    err_exp = np.asarray(err_exp, dtype=float)
    n = len(err_cheap)

    cheap_anchor = (cost_cheap, float(err_cheap.mean()))
    exp_anchor = (cost_exp, float(err_exp.mean()))

    summary = {
        "n_windows": n,
        "always_cheap": {"mean_error": cheap_anchor[1], "cost": cheap_anchor[0]},
        "always_expensive": {"mean_error": exp_anchor[1], "cost": exp_anchor[0]},
        "win_rate_expensive": float((err_exp < err_cheap).mean()),
        "random_curve": _route_random(err_cheap, err_exp, cost_cheap, cost_exp,
                                       n_random_trials, seed),
        "oracle_curve": _route_oracle(err_cheap, err_exp, cost_cheap, cost_exp),
        "probes": {},
    }
    for name, scores in score_dict.items():
        pts = _probe_curve(np.asarray(scores, float), err_cheap, err_exp,
                           cost_cheap, cost_exp)
        dom = _dominating_points(pts, cheap_anchor, exp_anchor)
        summary["probes"][name] = {
            "frontier": pts,
            "n_dominating_points": len(dom),
            "best_dominating": min(dom, key=lambda p: p["mean_error"]) if dom else None,
        }
    return summary
