"""Train a TopK SAE on cached activations — modality-agnostic.

Used for the Phase-3 expansion-robustness sweep: retrain at a non-native
expansion (LLM 4x->8x, TSFM 8x->4x) and re-probe to confirm the predictive null
is invariant to SAE width.

Trains on the TRAIN split only (leakage discipline). For sequence activations
with a `seq_len` column (LLM) it masks padding; for fixed-length windows (TSFM)
it uses every position. Faithful port of the legacy training loop (aux
dead-feature revival, decoder renorm, lr warmup).

    python experiments/train_sae.py \
        --activations data/llm/squad/activations/squad_activations.safetensors \
        --metadata    data/llm/squad/activations/squad_metadata.parquet \
        --expansion 8 --out sae_expansion8/sae_topk_32.pt
"""
from __future__ import annotations

import argparse
import logging
import os

import core._repro  # noqa: F401  — pins single-thread BLAS before numpy

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from core.sae import TopKSAE
from core._log import setup_logging, add_logging_args

log = logging.getLogger(__name__)


def load_train_tokens(activations_path, metadata_path, split_filter="train"):
    acts = load_file(activations_path)["encoder_embeddings"]
    d_model = int(acts.shape[-1])
    meta = pd.read_parquet(metadata_path)
    assert len(meta) == acts.shape[0], "metadata/activation row mismatch"
    keep = (meta["split"] == split_filter).values
    acts = acts[keep]
    meta_f = meta[keep].reset_index(drop=True)
    if "seq_len" in meta_f.columns and acts.dim() == 3 and acts.shape[1] > 1:
        # LLM: mask padding via per-row seq_len.
        valid = [acts[i, :int(r["seq_len"]), :] for i, r in meta_f.iterrows()]
        tokens = torch.cat(valid, dim=0)
    else:
        # TSFM (fixed-length windows) or single-position: use all positions.
        tokens = acts.reshape(-1, d_model)
    return tokens.to(torch.float32), d_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--expansion", type=int, default=None)
    ap.add_argument("--d_hidden", type=int, default=None)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--aux_k", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dead_after_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="Output checkpoint path (.pt)")
    add_logging_args(ap)
    args = ap.parse_args()
    setup_logging(args.verbose, args.quiet)
    torch.manual_seed(args.seed)

    tokens, d_model = load_train_tokens(args.activations, args.metadata)
    if args.d_hidden is None:
        args.d_hidden = (args.expansion or 4) * d_model
    log.info("train tokens=%s  d_model=%d  d_hidden=%d (%dx)",
             tuple(tokens.shape), d_model, args.d_hidden, args.d_hidden // d_model)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    var = tokens.var(dim=0).mean().item()
    loader = DataLoader(TensorDataset(tokens), batch_size=args.batch_size, shuffle=True)

    sae = TopKSAE(d_model=d_model, d_hidden=args.d_hidden, k=args.k, aux_k=args.aux_k).to(device)
    sae.b_dec.data = tokens.mean(dim=0).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min(1.0, s / max(1, args.warmup_steps)))

    steps_since_fired = torch.zeros(args.d_hidden, device=device)
    sae.train()
    for epoch in range(args.epochs):
        tot = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for (batch,) in pbar:
            batch = batch.to(device)
            dead = steps_since_fired > args.dead_after_steps
            opt.zero_grad()
            acts, recon, aux = sae(batch, dead_mask=dead)
            loss = F.mse_loss(recon, batch)
            if isinstance(aux, torch.Tensor):
                loss = loss + aux
            loss.backward()
            opt.step(); sched.step(); sae.normalize_decoder()
            fired = (acts > 0).sum(dim=0) > 0
            steps_since_fired = torch.where(fired, torch.zeros_like(steps_since_fired),
                                            steps_since_fired + 1)
            tot += F.mse_loss(recon, batch).item()
            pbar.set_postfix(nMSE=f"{(loss.item()/(var+1e-8)):.3f}",
                             dead=f"{dead.float().mean().item():.1%}")
        log.info("epoch %d: nMSE=%.3f", epoch + 1, tot / len(loader) / (var + 1e-8))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(sae.state_dict(), args.out)
    log.info("saved %s", args.out)


if __name__ == "__main__":
    main()
