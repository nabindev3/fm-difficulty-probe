"""Reconstruction-patching causal ablation — modality-agnostic core.

The protocol (Mishra-style), shared across both modalities:

  natural    : no intervention.
  sae_recon  : a forward hook on the chosen layer replaces its output with the
               SAE's reconstruction. Isolates the reconstruction-loss cost of
               inserting the SAE into the forward pass.
  ablate(f)  : same hook, but latent f is zeroed before decoding. Isolates the
               causal contribution of feature f.

A feature is causally tied to difficulty if Δmetric(ablate - sae_recon) is
significantly non-zero. Metric is modality-specific (Δnats for the LLM,
ΔCRPS for the TSFM) but the bootstrap aggregation is identical.

KEY UNIFICATION (the roadmap's "second cross-modal replication"):
`positions` is a first-class argument here. The LLM repo discovered that
single-position ('boundary') patching under-detects feature effects while
all-position patching reveals them — "coverage, not fidelity". This module
exposes the same knob so the TSFM side can run BOTH and test whether the same
coverage-not-fidelity story replicates.

What's modality-agnostic (here):
  - rank_top_features: pick top-k difficulty-predictive latents (L1 logistic
    on pooled SAE codes, train split only).
  - make_recon_hook:   the SAE-reconstruction forward hook, with optional
    feature ablation and single-/all-position slicing.
  - aggregate_ablation: paired-bootstrap Δ aggregation into a summary dict.

What's modality-specific (lives in the adapter):
  - which module to hook,
  - the per-item forward/eval that produces the scalar metric.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .stats import paired_bootstrap_mean_delta


def rank_top_features(
    sae_codes_pooled: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    k_features: int = 5,
    C: float = 0.1,
    n_blocks: int = 1,
    seed: int = 42,
) -> tuple[list[int], np.ndarray]:
    """Rank latents by |L1-logistic coefficient| on pooled SAE codes (train only).

    `sae_codes_pooled` is (N, n_blocks * d_hidden). With the default n_blocks=1 it
    is a single d_hidden-wide pooling (e.g. max-pool); pass n_blocks=3 if the
    caller concatenated mean/max/last poolings, in which case per-latent
    importance is summed across the blocks. Returns the top-k latent indices into
    [0, d_hidden) and the folded per-latent importance vector.
    """
    sae_codes_pooled = np.asarray(sae_codes_pooled, dtype=float)
    y = np.asarray(y).astype(int)
    width = sae_codes_pooled.shape[1]
    if width % n_blocks != 0:
        raise ValueError(f"feature width {width} not divisible by n_blocks={n_blocks}")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(sae_codes_pooled[train_mask])
    clf = LogisticRegression(
        penalty="l1", solver="liblinear",
        class_weight="balanced", max_iter=2000, C=C, random_state=seed,
    )
    clf.fit(X_tr, y[train_mask])
    coefs = np.abs(clf.coef_[0])

    folded = coefs.reshape(n_blocks, width // n_blocks).sum(axis=0) if n_blocks > 1 else coefs
    top = np.argsort(-folded)[:k_features].tolist()
    return [int(i) for i in top], folded


def make_recon_hook(
    sae,
    ablated_features: list[int] | None = None,
    positions: str = "all",
    index_fn: Callable[[], tuple[int, int]] | None = None,
):
    """Build a forward hook that patches a module's output with the SAE recon.

    Parameters
    ----------
    sae        : a TopKSAE (weights read once, detached).
    ablated_features : latents to zero before decoding (None = pure recon).
    positions  : "all"   -> patch every position in the slice returned by index_fn
                            (or the whole sequence if index_fn is None);
                 "last"  -> patch ONLY the final position [s-1, s] (the analogue
                            of the LLM "boundary" token; coverage = 1 position);
                 "single"-> patch one position; index_fn must return (idx, idx+1).
    index_fn   : zero-arg callable returning (start, end) token positions to patch,
                 read at hook-call time (lets the adapter set per-item boundaries).
                 For "all" with index_fn=None the entire sequence is patched.

    The `positions` knob is the cross-modal replication lever: "all" vs "last"
    tests whether feature effects are detectable only under broad intervention
    coverage (the LLM's coverage-not-fidelity finding).
    """
    W_enc = sae.W_enc.detach()
    b_enc = sae.b_enc.detach()
    W_dec = sae.W_dec.detach()
    b_dec = sae.b_dec.detach()
    k = sae.k
    d_model = sae.d_model
    abl = None if not ablated_features else torch.as_tensor(ablated_features, dtype=torch.long)

    def hook(module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        dtype = hidden.dtype
        b, s, d = hidden.shape

        if positions == "last":
            start, end = s - 1, s
        elif index_fn is not None:
            start, end = index_fn()
            start = max(0, start)
            end = min(end, s)
            if end <= start:
                return output
        else:
            start, end = 0, s

        sliced = hidden[:, start:end, :]
        n_tok = sliced.shape[1]
        flat = sliced.to(torch.float32).reshape(-1, d_model)

        x_centered = flat - b_dec
        pre = x_centered @ W_enc + b_enc
        top_acts, top_idx = torch.topk(pre, k, dim=-1)
        codes = torch.zeros_like(pre)
        codes.scatter_(-1, top_idx, F.relu(top_acts))
        if abl is not None:
            codes[:, abl] = 0.0
        recon = (codes @ W_dec + b_dec).reshape(b, n_tok, d_model).to(dtype)

        patched = hidden.clone()
        patched[:, start:end, :] = recon
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    return hook


def aggregate_ablation(
    natural: np.ndarray,
    sae_recon: np.ndarray,
    per_feature: dict[int, np.ndarray],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired-bootstrap aggregation of the three-condition ablation outcome.

    Reports Δ(sae_recon - natural) (the reconstruction-loss baseline) and, for
    each feature, Δ(ablate_f - sae_recon) with a 95% CI and a significance flag.
    Metric units (nats / CRPS) are recorded by the caller.
    """
    natural = np.asarray(natural, dtype=float)
    sae_recon = np.asarray(sae_recon, dtype=float)

    d_recon, ci_recon = paired_bootstrap_mean_delta(sae_recon, natural, n_boot, seed)
    summary = {
        "n": int(len(natural)),
        "mean_natural": float(natural.mean()),
        "mean_sae_recon": float(sae_recon.mean()),
        "delta_sae_recon": {"point": d_recon, "ci95": list(ci_recon)},
        "per_feature_ablation_delta": {},
    }
    for f, abl in per_feature.items():
        abl = np.asarray(abl, dtype=float)
        d, ci = paired_bootstrap_mean_delta(abl, sae_recon, n_boot, seed)
        summary["per_feature_ablation_delta"][int(f)] = {
            "point": d,
            "ci95": list(ci),
            "significant": bool(ci[0] > 0 or ci[1] < 0),
        }
    return summary
