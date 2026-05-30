"""TSFM causal ablation — single-position vs all-position reconstruction patching.

This is the Phase-3 "second cross-modal replication": the LLM side discovered
that single-position ('boundary') patching under-detects SAE-feature effects
while all-position patching reveals them (coverage, not fidelity). Here we run
the SAME comparison on Chronos-T5 to test whether the story replicates.

Pipeline (reuses core.patching for everything modality-agnostic):
  1. rank top-k difficulty-predictive latents (L1 logistic on pooled SAE codes).
  2. hook encoder.block[mid].layer[-1] with the SAE reconstruction hook.
  3. for positions in {all, last}: measure CRPS under natural / sae_recon /
     ablate(feature) per window; aggregate with paired-bootstrap Δ.

    python experiments/causal_tsfm.py --config configs/tsfm_etth1.yaml \
        --positions all  --max_windows 80
    python experiments/causal_tsfm.py --config configs/tsfm_etth1.yaml \
        --positions last --max_windows 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from core.sae import TopKSAE
from core.patching import rank_top_features, make_recon_hook, aggregate_ablation


def compute_crps(samples, truth):
    """Empirical CRPS (same estimator as the legacy extraction)."""
    crps_vals = []
    for i in range(len(truth)):
        s = samples[:, i]
        t = truth[i]
        mae = np.mean(np.abs(s - t))
        mean_diff = np.mean(np.abs(s[:, None] - s[None, :]))
        crps_vals.append(mae - 0.5 * mean_diff)
    return float(np.mean(crps_vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--positions", choices=["all", "last"], default="all")
    ap.add_argument("--k_features", type=int, default=5)
    ap.add_argument("--hard_quantile", type=float, default=0.85)
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--prediction_length", type=int, default=96)
    ap.add_argument("--max_windows", type=int, default=None)
    ap.add_argument("--layer", default="mid")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg.get("out_dir", "results/tsfm_etth1")
    os.makedirs(out_dir, exist_ok=True)
    ctx_len = cfg.get("context_length", 512)

    # --- SAE + activations for feature ranking ----------------------------- #
    state = torch.load(cfg["sae_ckpt"][args.layer], map_location="cpu")
    sae = TopKSAE.from_checkpoint(state, k=cfg.get("k", 32))

    acts = load_file(cfg["activations"][args.layer])["encoder_embeddings"].numpy()
    meta = pd.read_parquet(cfg["metadata"])

    # max-pool codes per window -> (N, d_hidden) for feature ranking.
    pooled = []
    with torch.no_grad():
        for i in range(acts.shape[0]):
            w = torch.tensor(acts[i:i + 1], dtype=torch.float32)
            c, _, _ = sae(w.reshape(-1, w.shape[-1]))
            pooled.append(c.reshape(w.shape[1], -1).numpy().max(axis=0))
    pooled = np.stack(pooled)

    col = "crps_norm" if "crps_norm" in meta.columns else "crps_raw"
    tr = (meta["split"] == "train").values
    thr = np.quantile(meta.loc[tr, col].values, args.hard_quantile)  # train-only threshold
    y = (meta[col].values >= thr).astype(int)
    top, _ = rank_top_features(pooled, y, tr, k_features=args.k_features, C=0.3)
    print(f"[{args.positions}] top-{args.k_features} features: {top}")

    # --- Chronos model ------------------------------------------------------ #
    from chronos import ChronosPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_name = cfg.get("causal_model", "amazon/chronos-t5-small")
    pipeline = ChronosPipeline.from_pretrained(model_name, device_map=device, dtype=dtype)
    n_layers = pipeline.model.model.config.num_layers
    layer_idx = cfg.get("layer_modules", {"mid": n_layers // 2}).get(args.layer, n_layers // 2)
    hook_module = pipeline.model.model.encoder.block[layer_idx].layer[-1]
    print(f"Hooking encoder.block[{layer_idx}].layer[-1] (num_layers={n_layers}), positions={args.positions}")

    series = pd.read_csv(cfg["series_csv"])[cfg.get("target_col", "OT")].values.astype(np.float64)
    test = meta[meta["split"] == "test"].copy().reset_index(drop=True)
    if args.max_windows:
        test = test.iloc[:args.max_windows].copy()
    print(f"Running ablation on {len(test)} test windows.")

    recon_hook = make_recon_hook(sae, ablated_features=None, positions=args.positions)
    feat_hooks = {f: make_recon_hook(sae, ablated_features=[f], positions=args.positions)
                  for f in top}

    def predict_crps(hook_fn, context, truth):
        h = hook_module.register_forward_hook(hook_fn) if hook_fn else None
        try:
            with torch.no_grad():
                fc = pipeline.predict(context, prediction_length=args.prediction_length,
                                      num_samples=args.num_samples)
        finally:
            if h:
                h.remove()
        fc = fc.cpu().numpy() if torch.is_tensor(fc) else np.asarray(fc)
        return compute_crps(fc[0], truth)

    rows = []
    for _, row in tqdm(list(test.iterrows()), total=len(test)):
        s = int(row["start_ts"])
        context = torch.tensor(series[s:s + ctx_len], dtype=torch.float32)
        truth = series[s + ctx_len:s + ctx_len + args.prediction_length]
        out = {"window_id": int(row["window_id"]),
               "crps_natural": float(row["crps_raw"]),
               "crps_sae_recon": predict_crps(recon_hook, context, truth)}
        for f in top:
            out[f"crps_ablate_{f}"] = predict_crps(feat_hooks[f], context, truth)
        rows.append(out)

    df = pd.DataFrame(rows)
    summary = aggregate_ablation(
        natural=df["crps_natural"].values,
        sae_recon=df["crps_sae_recon"].values,
        per_feature={f: df[f"crps_ablate_{f}"].values for f in top},
        seed=args.seed,
    )
    summary["metric"] = "CRPS"
    summary["positions"] = args.positions
    summary["top_features"] = top
    summary["model"] = model_name

    tag = f"causal_ablation_{args.positions}"
    df.to_parquet(os.path.join(out_dir, f"{tag}.parquet"))
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nΔ(sae_recon − natural) = {summary['delta_sae_recon']['point']:+.4f} "
          f"CI {summary['delta_sae_recon']['ci95']}")
    n_sig = 0
    for f, d in summary["per_feature_ablation_delta"].items():
        sig = " *" if d["significant"] else ""
        n_sig += d["significant"]
        print(f"  feat {f:5d}: Δ(ablate−recon) {d['point']:+.4f}  CI {d['ci95']}{sig}")
    print(f"[{args.positions}] {n_sig}/{len(top)} features significant. "
          f"Saved {out_dir}/{tag}.json")


if __name__ == "__main__":
    main()
