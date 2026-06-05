"""TSFM modality adapter — Chronos-T5 + forecast windows (ETTh1).

Wraps the legacy `tsfm-sae-routing` extraction outputs through the same
Modality interface the LLM adapter implements, so the shared core treats them
identically.

Cheap baseline  : 8 classical time-series statistics (variance, volatility,
                  lag-1 & seasonal autocorr, ADF p-value, spectral entropy,
                  trend slope, range).
Raw / SAE       : concat(mean, max, last) pooling of the encoder block output /
                  SAE codes (windows are fixed-length, so no padding mask needed).
Labels          : top-quartile normalized CRPS = "hard".
CV              : TimeSeriesSplit (consecutive overlapping windows must not leak).
Causal metric   : CRPS.

RECONCILIATION NOTE (per the roadmap): the headline TSFM comparison in the
legacy repo was SAE vs. classical stats; the P2 (cheap+raw encoder activations)
middle rung existed only as a 'diagnostic'. Routing through the shared
`run_probe_ladder` PROMOTES P2 to a first-class rung, so the TSFM side now
reports the identical three-rung ladder as the LLM side. That is the single
most important fix called out in the unification plan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


# --------------------------------------------------------------------------- #
# Cheap classical TS statistics (ported from legacy probing/probe.py).
# --------------------------------------------------------------------------- #
TS_STAT_NAMES = [
    "variance", "volatility", "lag1_acf", "seasonal_acf",
    "adf_pvalue", "spectral_entropy", "trend_slope", "range",
]


def _spectral_entropy(ts):
    import scipy.signal
    import scipy.stats
    _, Pxx = scipy.signal.welch(ts)
    if np.sum(Pxx) == 0:
        return 0.0
    Pxx = Pxx / np.sum(Pxx)
    return float(scipy.stats.entropy(Pxx))


def compute_ts_stats(df_meta: pd.DataFrame, series: np.ndarray,
                     context_length: int = 512, season_length: int = 24) -> np.ndarray:
    """8 classical features per window, computed on the context preceding it."""
    from statsmodels.tsa.stattools import acf, adfuller
    stats = []
    for _, row in df_meta.iterrows():
        s = int(row["start_ts"])
        x = series[s:s + context_length]
        n = len(x)
        var = float(np.var(x))
        volatility = float(np.mean(np.abs(np.diff(x)))) if n > 1 else 0.0
        acf_vals = (acf(x, nlags=max(1, season_length), fft=False)
                    if n > season_length else np.zeros(season_length + 1))
        lag1 = float(acf_vals[1]) if len(acf_vals) > 1 else 0.0
        seasonal = float(acf_vals[season_length]) if len(acf_vals) > season_length else 0.0
        try:
            adf_p = float(adfuller(x, autolag="AIC")[1])
        except Exception:
            adf_p = 1.0
        ent = _spectral_entropy(x)
        slope = float(np.polyfit(np.arange(n), x, 1)[0]) if n > 1 else 0.0
        rng = float(x.max() - x.min())
        stats.append([var, volatility, lag1, seasonal, adf_p, ent, slope, rng])
    return np.array(stats)


def _pool(seq_tensor: np.ndarray) -> np.ndarray:
    """concat(mean, max, last) over the (fixed-length) window dimension."""
    mean = seq_tensor.mean(axis=1)
    mx = seq_tensor.max(axis=1)
    last = seq_tensor[:, -1, :]
    return np.concatenate([mean, mx, last], axis=1)


# --------------------------------------------------------------------------- #
# Adapter.
# --------------------------------------------------------------------------- #
class TSFMModality:
    """Chronos-T5 adapter. Construct from a config dict (see configs/tsfm_*.yaml)."""
    name = "tsfm"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.meta = pd.read_parquet(cfg["metadata"])
        self._series = None
        self._raw_cache: dict[str, np.ndarray] = {}
        self._sae_cache: dict[str, np.ndarray] = {}

    @property
    def series(self) -> np.ndarray:
        if self._series is None:
            df = pd.read_csv(self.cfg["series_csv"])
            self._series = df[self.cfg.get("target_col", "OT")].values.astype(np.float64)
        return self._series

    def _load_raw(self, layer: str) -> np.ndarray:
        from safetensors.torch import load_file
        return load_file(self.cfg["activations"][layer])["encoder_embeddings"].numpy()

    def _load_sae(self, layer: str):
        import torch
        from core.sae import TopKSAE
        state = torch.load(self.cfg["sae_ckpt"][layer], map_location="cpu", weights_only=True)
        return TopKSAE.from_checkpoint(state, k=self.cfg.get("k", 32))

    # --- Modality interface ------------------------------------------------- #
    def cheap_baseline_features(self) -> np.ndarray:
        return compute_ts_stats(self.meta, self.series,
                                context_length=self.cfg.get("context_length", 512))

    def raw_activations(self, layer: str = "mid") -> np.ndarray:
        if layer not in self._raw_cache:
            self._raw_cache[layer] = _pool(self._load_raw(layer))
        return self._raw_cache[layer]

    def sae_codes(self, layer: str = "mid") -> np.ndarray:
        if layer not in self._sae_cache:
            import torch
            raw = self._load_raw(layer)
            sae = self._load_sae(layer)
            codes = []
            with torch.no_grad():
                for i in range(raw.shape[0]):
                    w = torch.tensor(raw[i:i + 1], dtype=torch.float32)
                    acts, _, _ = sae(w.reshape(-1, w.shape[-1]))
                    codes.append(acts.reshape(w.shape[1], -1).numpy())
            self._sae_cache[layer] = _pool(np.stack(codes))
        return self._sae_cache[layer]

    def difficulty_labels(self) -> np.ndarray:
        col = "crps_norm" if "crps_norm" in self.meta.columns else "crps_raw"
        q = self.cfg.get("hard_quantile", 0.75)
        # label_threshold_split controls where the hard-quantile threshold is computed:
        #   "all"   -> whole-dataset quantile  (REPRODUCES the legacy TSFM number;
        #              the Phase-2 regression gate uses this).
        #   "train" -> train-only quantile     (leakage fix; documented Phase-3 improvement).
        split = self.cfg.get("label_threshold_split", "all")
        if split == "train":
            tr = (self.meta["split"] == "train").values
            thr = np.quantile(self.meta.loc[tr, col].values, q)
        else:
            thr = np.quantile(self.meta[col].values, q)
        return (self.meta[col].values >= thr).astype(int)

    def selective_error(self, test_mask: np.ndarray) -> np.ndarray:
        # Natural error scale = continuous CRPS (risk = mean CRPS on retained
        # windows), matching the legacy TSFM selective-prediction report.
        return self.meta["crps_raw"].values.astype(float)[test_mask]

    def split_masks(self):
        return ((self.meta["split"] == "train").values,
                (self.meta["split"] == "test").values)

    def cv_folds(self, train_mask: np.ndarray, y: np.ndarray):
        n_tr = int(train_mask.sum())
        y_tr = y[train_mask]
        n_splits = max(2, min(5, int(np.bincount(y_tr).min()) - 1, n_tr // 3))
        return list(TimeSeriesSplit(n_splits=n_splits).split(np.zeros((n_tr, 1))))

    def layers(self) -> dict:
        # Encoder block indices; filled by config (legacy used mid=3, late=5).
        return self.cfg.get("layer_modules", {"mid": 3, "late": 5})

    def causal_metric_name(self) -> str:
        return "CRPS"

    def cascade_errors(self):
        small = pd.read_parquet(self.cfg["cascade"]["cheap_metadata"])
        base = pd.read_parquet(self.cfg["cascade"]["expensive_metadata"])
        # Use the column present in BOTH (base extraction only stores crps_raw),
        # so cheap and expensive errors are on the same scale.
        col = "crps_raw"
        small = small[["window_id", col]].rename(columns={col: "err_cheap"})
        base = base[["window_id", col]].rename(columns={col: "err_exp"})
        df = small.merge(base, on="window_id")
        return df["err_cheap"].values.astype(float), df["err_exp"].values.astype(float)

    def cascade_costs(self):
        c = self.cfg.get("cascade", {})
        return float(c.get("cost_cheap", 1.0)), float(c.get("cost_expensive", 4.0))
