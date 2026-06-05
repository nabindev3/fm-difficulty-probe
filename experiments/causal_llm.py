"""LLM causal ablation — boundary (single) vs all-position reconstruction patching.

Port of the legacy hook-based ablation onto core.patching, so the LLM and TSFM
causal experiments share the hook + aggregation code. Pythia residual-stream
patching at the SAE's training layer; continuous metric (neg-log-prob of the true
HellaSwag ending | cross-entropy of the SQuAD gold answer), which is the
sub-threshold-sensitive metric used in the cross-modal comparison.

Use either the in-process API (`run_causal`, called by experiments/run.py) or the
CLI:

    USE_TF=0 python experiments/causal_llm.py --config configs/llm_squad.yaml --positions all
    USE_TF=0 python experiments/causal_llm.py --config configs/llm_squad.yaml --positions boundary
"""
from __future__ import annotations

import argparse
import json
import os

import core._repro  # noqa: F401  — pins single-thread BLAS before numpy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from core.sae import TopKSAE
from core.patching import rank_top_features, make_recon_hook, aggregate_ablation
from modalities.llm import aggregate_sequence  # noqa: F401  (kept for parity / external callers)


def run_causal(
    cfg: dict,
    positions: str = "all",
    layer: str = "mid",
    k_features: int = 5,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict:
    """Run the Pythia reconstruction-patching ablation for one position mode.

    Returns the aggregated summary dict (also written to <out_dir>/causal_ablation_
    <positions>.json/.parquet). Raises on failure so callers get real error
    propagation. `positions` is 'all' (patch the whole prompt) or 'boundary'
    (patch only the final prompt token — the single-position coverage condition).
    """
    if positions not in ("all", "boundary"):
        raise ValueError(f"positions must be 'all' or 'boundary', got {positions!r}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = cfg.get("experiment", "squad")
    out_dir = cfg.get("out_dir", "results/llm")
    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_parquet(cfg["metadata"])
    raw = load_file(cfg["activations"][layer])["encoder_embeddings"]
    sae = TopKSAE.from_checkpoint(
        torch.load(cfg["sae_ckpt"][layer], map_location="cpu", weights_only=True),
        k=cfg.get("k", 32))
    d_model, d_hidden = sae.W_enc.shape

    y = meta["difficulty"].values.astype(int)
    tr = (meta["split"] == "train").values

    # Rank top-k features on train-split SAE codes (max-pool to d_hidden).
    N, S, _ = raw.shape
    with torch.no_grad():
        codes = sae(raw.reshape(-1, d_model).float())[0].reshape(N, S, d_hidden).numpy()
    pooled = codes.max(axis=1)  # (N, d_hidden)
    top, _ = rank_top_features(pooled, y, tr, k_features=k_features, C=0.1)
    print(f"[{positions}] top-{k_features} features: {top}")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    sae = sae.to(device)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = cfg.get("causal_model", "EleutherAI/pythia-410m")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32 if device != "cuda" else torch.float16).to(device).eval()
    layer_idx = cfg.get("layer_modules", {"mid": 11, "late": 17})[layer]

    # Hooks via core.patching; index_fn reads per-sample state set in the loop.
    state = {"prompt_len": None, "boundary": None}
    if positions == "all":
        idx_fn = lambda: (0, state["prompt_len"])
        pos_mode = "all"
    else:
        idx_fn = lambda: (state["boundary"], state["boundary"] + 1)
        pos_mode = "single"
    recon_hook = make_recon_hook(sae, None, positions=pos_mode, index_fn=idx_fn)
    feat_hooks = {f: make_recon_hook(sae, [f], positions=pos_mode, index_fn=idx_fn) for f in top}
    hook_target = model.gpt_neox.layers[layer_idx]

    from datasets import load_dataset
    ds = load_dataset(dataset, split="validation")
    test = meta[meta["split"] == "test"].copy().reset_index(drop=True)
    if max_samples:
        test = test.iloc[:max_samples].copy()
    print(f"Running {dataset} causal ({positions}) on {len(test)} prompts.")

    def metric_for(window_id):
        """Continuous difficulty metric for one item, evaluated under the active hook."""
        s = ds[int(window_id)]
        if dataset == "hellaswag":
            prompt = s["ctx_a"] + ((" " + s["ctx_b"]) if s["ctx_b"] else "")
            pids = tok.encode(prompt, add_special_tokens=True)[-128:]
            plen = len(pids); state["prompt_len"] = plen; state["boundary"] = plen - 1
            eids = tok.encode(" " + s["endings"][int(s["label"])].strip(), add_special_tokens=False)
            ids = torch.tensor([pids + eids], device=device)
            with torch.no_grad():
                logits = model(ids).logits
            lp = F.log_softmax(logits[0, plen - 1:-1], dim=-1)
            tgt = ids[0, plen:]
            return -lp[torch.arange(len(eids)), tgt].mean().item()
        else:  # squad
            prompt = f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:"
            pids = tok.encode(prompt, add_special_tokens=True)[-200:]
            plen = len(pids); state["prompt_len"] = plen; state["boundary"] = plen - 1
            tids = tok.encode(" " + s["answers"]["text"][0].strip(), add_special_tokens=False)
            ids = torch.tensor([pids + tids], device=device)
            with torch.no_grad():
                logits = model(ids).logits
            return F.cross_entropy(logits[0, plen - 1:-1], ids[0, plen:]).item()

    rows = []
    for _, row in tqdm(list(test.iterrows()), total=len(test)):
        wid = int(row["window_id"])
        nat = metric_for(wid)
        h = hook_target.register_forward_hook(recon_hook)
        rec = metric_for(wid); h.remove()
        out = {"window_id": wid, "m_natural": nat, "m_sae_recon": rec}
        for f in top:
            h = hook_target.register_forward_hook(feat_hooks[f])
            out[f"m_ablate_{f}"] = metric_for(wid); h.remove()
        rows.append(out)

    df = pd.DataFrame(rows)
    summary = aggregate_ablation(df["m_natural"].values, df["m_sae_recon"].values,
                                 {f: df[f"m_ablate_{f}"].values for f in top}, seed=seed)
    summary.update({"metric": "nats", "positions": positions,
                    "top_features": top, "model": model_name})
    tag = f"causal_ablation_{positions}"
    df.to_parquet(os.path.join(out_dir, f"{tag}.parquet"))
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    nsig = sum(v["significant"] for v in summary["per_feature_ablation_delta"].values())
    print(f"[{positions}] {nsig}/{len(top)} significant. Saved {out_dir}/{tag}.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--positions", choices=["all", "boundary"], default="all")
    ap.add_argument("--layer", default="mid")
    ap.add_argument("--k_features", type=int, default=5)
    ap.add_argument("--max_samples", type=int, default=None, help="Cap test prompts.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_causal(cfg, positions=args.positions, layer=args.layer,
               k_features=args.k_features, max_samples=args.max_samples, seed=args.seed)


if __name__ == "__main__":
    main()
