"""Synthetic tests for the causal-ablation core — the path behind the paper's
headline causal result. No real model: a tiny TopKSAE + hand-built tensors are
enough to pin the *semantics* of the reconstruction hook and the ranking /
aggregation, which is what a reviewer's "did the intervention do what you say?"
question really targets.

Covers:
  - core.patching.make_recon_hook   (ablation semantics, position slicing, I/O shape)
  - core.patching.rank_top_features (recovers a planted predictive latent)
  - core.patching.aggregate_ablation(significance flag logic)
  - core.stats.label_permutation_test (one-sided directionality)
"""
import numpy as np
import torch

from core.sae import TopKSAE
from core.patching import make_recon_hook, rank_top_features, aggregate_ablation
from core import stats as ST


def _tiny_sae(seed=0, d_model=4, d_hidden=16, k=2):
    torch.manual_seed(seed)
    return TopKSAE(d_model=d_model, d_hidden=d_hidden, k=k).eval()


def _active_latents(sae, hidden):
    """Mirror the encoder's TopK routing to find which latents fire anywhere in
    the slice — lets the test pick a known-active and a known-inactive latent."""
    flat = hidden.reshape(-1, sae.d_model).float() - sae.b_dec.detach()
    pre = flat @ sae.W_enc.detach() + sae.b_enc.detach()
    _, idx = torch.topk(pre, sae.k, dim=-1)
    return set(int(i) for i in idx.reshape(-1).tolist())


# --------------------------------------------------------------------------- #
# make_recon_hook — the intervention that defines the causal claim.
# --------------------------------------------------------------------------- #
def test_recon_hook_ablation_semantics():
    """Zeroing an ACTIVE latent must change the reconstruction; zeroing an
    INACTIVE latent must be a no-op. This is exactly the property the per-feature
    ablation relies on to attribute a ΔCRPS / Δnats to a specific feature."""
    sae = _tiny_sae()
    torch.manual_seed(1)
    hidden = torch.randn(1, 3, sae.d_model)
    active = _active_latents(sae, hidden)
    inactive = sorted(set(range(sae.d_hidden)) - active)
    assert active and inactive, "test needs both an active and an inactive latent"

    recon = make_recon_hook(sae, None, positions="all")(None, None, hidden)
    f_active = sorted(active)[0]
    f_inactive = inactive[0]
    abl_active = make_recon_hook(sae, [f_active], positions="all")(None, None, hidden)
    abl_inactive = make_recon_hook(sae, [f_inactive], positions="all")(None, None, hidden)

    assert recon.shape == hidden.shape
    assert not torch.allclose(recon, abl_active), "ablating an active latent should change recon"
    assert torch.allclose(recon, abl_inactive), "ablating an inactive latent should be a no-op"


def test_recon_hook_last_position_only():
    """positions='last' patches ONLY the final position — the single-position
    'coverage' condition. Earlier positions must pass through untouched."""
    sae = _tiny_sae(seed=3)
    torch.manual_seed(2)
    hidden = torch.randn(1, 5, sae.d_model)
    patched = make_recon_hook(sae, None, positions="last")(None, None, hidden)

    assert patched.shape == hidden.shape
    assert torch.allclose(patched[:, :-1, :], hidden[:, :-1, :]), "non-final positions changed"
    assert not torch.allclose(patched[:, -1, :], hidden[:, -1, :]), "final position not patched"


def test_recon_hook_index_fn_slice():
    """When index_fn returns (start, end), only that span is patched (the LLM
    'all-position over the prompt' path)."""
    sae = _tiny_sae(seed=4)
    torch.manual_seed(5)
    hidden = torch.randn(1, 6, sae.d_model)
    hook = make_recon_hook(sae, None, positions="all", index_fn=lambda: (2, 4))
    patched = hook(None, None, hidden)

    assert torch.allclose(patched[:, :2, :], hidden[:, :2, :])
    assert torch.allclose(patched[:, 4:, :], hidden[:, 4:, :])
    assert not torch.allclose(patched[:, 2:4, :], hidden[:, 2:4, :])


def test_recon_hook_preserves_tuple_output():
    """HF modules often return (hidden, *rest); the hook must patch hidden and
    pass the rest through unchanged."""
    sae = _tiny_sae(seed=6)
    hidden = torch.randn(1, 3, sae.d_model)
    extra = torch.tensor([7.0])
    out = make_recon_hook(sae, None, positions="all")(None, None, (hidden, extra))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == hidden.shape
    assert out[1] is extra


# --------------------------------------------------------------------------- #
# rank_top_features — picks the difficulty-predictive latents to ablate.
# --------------------------------------------------------------------------- #
def test_rank_top_features_recovers_planted_latent():
    rng = np.random.default_rng(0)
    n, d = 300, 12
    y = (rng.random(n) < 0.5).astype(int)
    pooled = rng.normal(size=(n, d))
    pooled[:, 7] += 3.0 * y            # latent 7 strongly predicts the label
    train = np.zeros(n, bool); train[: n // 2] = True

    top, importance = rank_top_features(pooled, y, train, k_features=3, C=0.5)
    assert 7 in top
    assert importance.shape == (d,)
    assert importance[7] == importance.max()


# --------------------------------------------------------------------------- #
# aggregate_ablation — turns paired metrics into the significance flag.
# --------------------------------------------------------------------------- #
def test_aggregate_ablation_significance_flag():
    rng = np.random.default_rng(1)
    n = 200
    natural = rng.normal(size=n)
    sae_recon = natural + 0.001 * rng.normal(size=n)             # recon ≈ natural
    abl_sig = sae_recon + 0.5 + 0.01 * rng.normal(size=n)        # consistent shift
    abl_null = sae_recon + 0.01 * rng.normal(size=n)             # no shift

    out = aggregate_ablation(natural, sae_recon, {1: abl_sig, 2: abl_null}, n_boot=500)
    assert out["n"] == n
    assert out["per_feature_ablation_delta"][1]["significant"] is True
    assert out["per_feature_ablation_delta"][1]["point"] > 0
    assert out["per_feature_ablation_delta"][2]["significant"] is False


# --------------------------------------------------------------------------- #
# label_permutation_test — the predictive-null half of the thesis.
# --------------------------------------------------------------------------- #
def test_label_permutation_directionality():
    rng = np.random.default_rng(5)
    n = 300
    y = (rng.random(n) < 0.5).astype(int)
    strong = y + rng.normal(scale=0.3, size=n)    # very informative
    weak = rng.normal(size=n)                      # uninformative

    res = ST.label_permutation_test(y, strong, weak, n_perm=1000)
    assert res["observed_delta"] > 0
    assert res["p_one_sided"] < 0.05

    # Identical scores -> exactly zero delta and a non-significant one-sided p.
    same = ST.label_permutation_test(y, strong, strong, n_perm=500)
    assert abs(same["observed_delta"]) < 1e-12
    assert same["p_one_sided"] > 0.2
