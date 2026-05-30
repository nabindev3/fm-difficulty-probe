"""LLM modality adapter — Pythia + HellaSwag / SQuAD.

Wraps the legacy `llm-sae-difficulty` extraction outputs (a safetensors of
padded per-token activations + a metadata parquet + a trained TopK SAE) and
exposes them through the Modality interface.

Cheap baseline  : 8 lexical prompt statistics (char/token len, TTR, prompt ppl,
                  char-to-token, capitalization, category id, bigram novelty).
Raw / SAE       : padding-aware mean+max+last pooling of the residual stream /
                  SAE codes.
Labels          : meta['difficulty'] (1 = incorrect / high gold-perplexity).
CV              : StratifiedKFold (i.i.d. prompts).
Causal metric   : nats (neg-log-prob of true ending | cross-entropy of gold answer).

This adapter assumes the legacy extraction has already been run. Porting the
cheap-stat and pooling code here (rather than importing from _legacy) is the
Phase-1 promotion: the core never sees it, only the adapter does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# --------------------------------------------------------------------------- #
# Cheap lexical statistics (ported from legacy probing/features.py).
# --------------------------------------------------------------------------- #
INPUT_STAT_NAMES = [
    "char_length", "token_count", "lexical_diversity", "perplexity",
    "char_to_token", "capitalization_ratio", "category_id", "bigram_novelty",
]


def compute_prompt_stats(df_meta: pd.DataFrame) -> np.ndarray:
    """8 prompt-level cheap statistics. Prompt-only perplexity is used (never the
    gold-target ppl that the SQuAD label is derived from — that would be leakage)."""
    train_mask = (df_meta["split"] == "train").values
    train_prompts = (df_meta.loc[train_mask, "prompt"].values
                     if train_mask.sum() else df_meta["prompt"].values)
    global_bigrams: set = set()
    for prompt in train_prompts:
        words = prompt.lower().split()
        if len(words) >= 2:
            global_bigrams.update(zip(words[:-1], words[1:]))

    categories = df_meta["activity_label"].unique()
    cat_to_id = {c: i for i, c in enumerate(categories)}

    prompts = df_meta["prompt"].to_numpy()
    words_lower = [[w.lower() for w in p.split()] for p in prompts]

    char_len = np.array([len(p) for p in prompts], dtype=np.float64)
    token_count = df_meta["seq_len"].to_numpy(dtype=np.float64)
    char_to_token = char_len / (token_count + 1e-8)
    lexical_diversity = np.array(
        [len(set(ws)) / (len(ws) + 1e-8) for ws in words_lower], dtype=np.float64)

    ppl_col = "prompt_perplexity" if "prompt_perplexity" in df_meta.columns else "perplexity"
    perplexity = df_meta[ppl_col].to_numpy(dtype=np.float64)

    def _cap_ratio(p: str) -> float:
        alpha = upper = 0
        for c in p:
            if c.isalpha():
                alpha += 1
                if c.isupper():
                    upper += 1
        return upper / (alpha + 1e-8)
    cap = np.fromiter((_cap_ratio(p) for p in prompts), dtype=np.float64, count=len(prompts))
    cat_id = df_meta["activity_label"].map(cat_to_id).fillna(0.0).to_numpy(dtype=np.float64)

    def _novelty(ws):
        if len(ws) < 2:
            return 1.0
        bg = set(zip(ws[:-1], ws[1:]))
        return sum(1 for b in bg if b not in global_bigrams) / len(bg)
    novelty = np.array([_novelty(ws) for ws in words_lower], dtype=np.float64)

    return np.column_stack([char_len, token_count, lexical_diversity, perplexity,
                            char_to_token, cap, cat_id, novelty])


def aggregate_sequence(seq_tensor: np.ndarray, meta_df: pd.DataFrame) -> np.ndarray:
    """Padding-aware concat(mean, max, last) pooling over valid prompt tokens."""
    if seq_tensor.ndim != 3:
        raise ValueError(f"Expected (N, seq, d), got {seq_tensor.shape}")
    N, max_seq, d = seq_tensor.shape
    if max_seq == 1:
        return seq_tensor[:, 0, :]
    mean_l, max_l, last_l = [], [], []
    for i in range(N):
        seq_len = min(int(meta_df.iloc[i]["seq_len"]), max_seq)
        valid = seq_tensor[i, :seq_len, :]
        mean_l.append(valid.mean(axis=0))
        max_l.append(valid.max(axis=0))
        last_l.append(valid[-1, :])
    return np.concatenate([np.array(mean_l), np.array(max_l), np.array(last_l)], axis=1)


# --------------------------------------------------------------------------- #
# Adapter.
# --------------------------------------------------------------------------- #
class LLMModality:
    """Pythia adapter. Construct from a config dict (see configs/llm_*.yaml)."""
    name = "llm"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.meta = pd.read_parquet(cfg["metadata"])
        self._raw_cache: dict[str, np.ndarray] = {}
        self._sae_cache: dict[str, np.ndarray] = {}

    # --- lazy loaders ------------------------------------------------------- #
    def _load_raw(self, layer: str) -> np.ndarray:
        from safetensors.torch import load_file
        path = self.cfg["activations"][layer]
        return load_file(path)["encoder_embeddings"].numpy()

    def _load_sae(self, layer: str):
        import torch
        from core.sae import TopKSAE
        state = torch.load(self.cfg["sae_ckpt"][layer], map_location="cpu")
        return TopKSAE.from_checkpoint(state, k=self.cfg.get("k", 32))

    # --- Modality interface ------------------------------------------------- #
    def cheap_baseline_features(self) -> np.ndarray:
        return compute_prompt_stats(self.meta)

    def raw_activations(self, layer: str = "mid") -> np.ndarray:
        if layer not in self._raw_cache:
            raw = self._load_raw(layer)
            self._raw_cache[layer] = aggregate_sequence(raw, self.meta)
        return self._raw_cache[layer]

    def sae_codes(self, layer: str = "mid") -> np.ndarray:
        if layer not in self._sae_cache:
            import torch
            raw = self._load_raw(layer)
            sae = self._load_sae(layer)
            N, max_seq, d = raw.shape
            x = torch.tensor(raw.reshape(-1, d), dtype=torch.float32)
            codes = []
            with torch.no_grad():
                for i in range(0, x.shape[0], 8192):
                    acts, _, _ = sae(x[i:i + 8192])
                    codes.append(acts.cpu().numpy())
            codes = np.concatenate(codes).reshape(N, max_seq, sae.d_hidden)
            self._sae_cache[layer] = aggregate_sequence(codes, self.meta)
        return self._sae_cache[layer]

    def difficulty_labels(self) -> np.ndarray:
        return self.meta["difficulty"].values.astype(int)

    def split_masks(self):
        return ((self.meta["split"] == "train").values,
                (self.meta["split"] == "test").values)

    def cv_folds(self, train_mask: np.ndarray, y: np.ndarray):
        y_tr = y[train_mask]
        n_splits = max(2, min(5, int(np.bincount(y_tr).min()) - 1, int(train_mask.sum()) // 10))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(skf.split(np.zeros((len(y_tr), 1)), y_tr))

    def layers(self) -> dict:
        # Residual block indices (0-based) the SAEs were trained on; e.g. 11 -> L12.
        return self.cfg.get("layer_modules", {"mid": 11, "late": 17})

    def causal_metric_name(self) -> str:
        return "nats"

    def cascade_errors(self):
        small = pd.read_parquet(self.cfg["cascade"]["cheap_metadata"])
        base = pd.read_parquet(self.cfg["cascade"]["expensive_metadata"])
        small = small[["window_id", "difficulty"]].rename(columns={"difficulty": "err_cheap"})
        base = base[["window_id", "difficulty"]].rename(columns={"difficulty": "err_exp"})
        df = small.merge(base, on="window_id")
        return df["err_cheap"].values.astype(float), df["err_exp"].values.astype(float)

    def cascade_costs(self):
        c = self.cfg.get("cascade", {})
        return float(c.get("cost_cheap", 1.0)), float(c.get("cost_expensive", 5.0))
