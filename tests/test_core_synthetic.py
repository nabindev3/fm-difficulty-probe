"""Phase-1 acceptance: every core function runs on synthetic arrays, no model.

Run with:  pytest tests/ -q
These tests don't assert research conclusions — they assert the shared core is
modality-agnostic (numpy in, results out) and internally consistent.
"""
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from core import probe as P
from core import selective as S
from core import calibration as C
from core import cascade as K
from core import stats as ST


def _synthetic(n=400, d_cheap=8, d_act=24, seed=0):
    """Cheap features carry signal; raw acts carry a bit more; SAE adds nothing
    over raw — mirrors the project's expected qualitative shape."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)                       # latent difficulty
    y = (z + 0.3 * rng.normal(size=n) > 0).astype(int)
    cheap = np.column_stack([z + rng.normal(size=n) for _ in range(d_cheap)])
    raw = np.column_stack([z + rng.normal(size=n) for _ in range(d_act)])
    sae = raw + 0.01 * rng.normal(size=raw.shape)   # ~redundant with raw
    return cheap, raw, sae, y


def test_probe_ladder_runs_and_shapes():
    cheap, raw, sae, y = _synthetic()
    feats = P.build_ladder(cheap, raw, sae)
    assert set(feats) == {P.P1, P.P2, P.P3, P.P4, P.P5}
    n = len(y)
    train_mask = np.zeros(n, bool); train_mask[: n // 2] = True
    test_mask = ~train_mask
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0)
                 .split(np.zeros((train_mask.sum(), 1)), y[train_mask]))

    result, preds = P.run_probe_ladder(feats, y, train_mask, test_mask, folds, n_boot=200)
    assert result.n_test == int(test_mask.sum())
    for name, pr in result.probes.items():
        assert 0.0 <= pr.auroc <= 1.0
        assert pr.ci_low <= pr.auroc <= pr.ci_high + 1e-6
    assert f"{P.P3}-{P.P2}" in result.deltas
    assert set(preds) == set(feats)


def test_selective_prediction_bounds():
    rng = np.random.default_rng(1)
    n = 300
    errors = (rng.random(n) < 0.4).astype(float)
    good = errors + 0.5 * rng.random(n)       # informative score
    rand = rng.random(n)                       # uninformative score
    out = S.selective_prediction({"good": good, "rand": rand}, errors, n_bootstrap=200)
    assert out["oracle_aurc"] <= out["random_aurc"] + 1e-9
    # An informative score should capture more oracle AURC than a random one.
    assert (out["probes"]["good"]["fraction_of_oracle_aurc"]
            >= out["probes"]["rand"]["fraction_of_oracle_aurc"] - 0.1)


def test_calibration_recalibrate_oof():
    cheap, raw, sae, y = _synthetic(n=300)
    n = len(y)
    tr = np.zeros(n, bool); tr[: n // 2] = True
    te = ~tr
    out = C.recalibrate_oof(cheap[tr], y[tr], cheap[te], y[te], C=1.0, n_splits=3)
    for tag in ("raw", "platt", "isotonic"):
        assert 0.0 <= out[tag]["ece"] <= 1.0
    assert out["_test_probs"]["platt"].shape == (te.sum(),)


def test_cascade_dominating_points():
    rng = np.random.default_rng(2)
    n = 300
    err_cheap = (rng.random(n) < 0.5).astype(float)
    # expensive fixes ~half the cheap errors
    err_exp = err_cheap.copy()
    flip = (err_cheap == 1) & (rng.random(n) < 0.5)
    err_exp[flip] = 0.0
    score = err_cheap - err_exp + 0.1 * rng.random(n)   # good routing signal
    out = K.cascade({"good": score}, err_cheap, err_exp, 1.0, 5.0, n_random_trials=50)
    assert out["probes"]["good"]["n_dominating_points"] >= 0
    assert out["always_expensive"]["mean_error"] <= out["always_cheap"]["mean_error"] + 1e-9


def test_permutation_test_symmetry():
    rng = np.random.default_rng(3)
    n = 200
    y = (rng.random(n) < 0.5).astype(int)
    a = y + rng.normal(scale=0.5, size=n)     # informative
    b = rng.normal(size=n)                     # noise
    res = ST.label_permutation_test(y, a, b, n_perm=500)
    assert 0.0 <= res["p_one_sided"] <= 1.0
    assert res["auroc_a"] >= res["auroc_b"] - 0.2
