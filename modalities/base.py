"""The Modality interface every backend must satisfy.

The whole point of the merged repo: `experiments/run.py` drives the shared core
through this interface and never imports a model directly. Adding a third
modality means writing one adapter, not touching the core.

An adapter owns everything model- and dataset-specific:
  - producing the three feature blocks (cheap baseline, raw activations, SAE codes),
  - the difficulty labels and the train/test split masks,
  - the inner-CV folds (stratified for i.i.d. data, time-series for sequential),
  - the causal metric and which layer to hook for patching,
  - the cheap/expensive model pair and their relative cost for the cascade.

The core consumes only numpy arrays + a TopKSAE, so adapters are free to read
activations from whatever on-disk format the legacy extraction produced.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Modality(Protocol):
    name: str

    # --- feature blocks (all aligned, shape (N, d_*)) ----------------------- #
    def cheap_baseline_features(self) -> np.ndarray:
        """Cheap, no-forward-pass features: lexical stats | classical TS stats."""
        ...

    def raw_activations(self, layer: str) -> np.ndarray:
        """Pooled raw model activations at a layer ('mid' | 'late').

        The crucial MIDDLE rung of the ladder — the comparator both modalities
        must share for the cross-modal claim to be apples-to-apples.
        """
        ...

    def sae_codes(self, layer: str) -> np.ndarray:
        """Pooled SAE codes at a layer, using the train-fit SAE for this modality."""
        ...

    # --- labels & splits ---------------------------------------------------- #
    def difficulty_labels(self) -> np.ndarray:
        """Binary difficulty labels (1 = hard), aligned with the feature blocks."""
        ...

    def split_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """(train_mask, test_mask) boolean arrays. Leakage controls live here:
        purge gap, pretraining-contamination filter, etc."""
        ...

    def cv_folds(self, train_mask: np.ndarray, y: np.ndarray):
        """Inner-CV folds over the TRAIN rows: StratifiedKFold for the LLM,
        TimeSeriesSplit for the TSFM. Returned materialized (list of index pairs)."""
        ...

    # --- causal side -------------------------------------------------------- #
    def layers(self) -> dict[str, object]:
        """Named layer handles, at minimum {'mid': ..., 'late': ...}."""
        ...

    def causal_metric_name(self) -> str:
        """e.g. 'nats' (LLM) or 'CRPS' (TSFM) — recorded in the ablation summary."""
        ...

    # --- cascade models ----------------------------------------------------- #
    def cascade_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """(err_cheap, err_expensive) per test item, on the cascade's error scale."""
        ...

    def cascade_costs(self) -> tuple[float, float]:
        """(cost_cheap, cost_expensive) relative inference costs."""
        ...
