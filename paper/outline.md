# Paper outline — cross-modal SAE difficulty probing

**Working titles**
1. *SAE Features Are Causally Active but Predictively Redundant for Difficulty:
   Evidence Across Language and Time-Series Foundation Models*
2. *Do Foundation Models Know What They Don't Know? Label-Free Difficulty
   Probing Across Modalities*

**Framing:** a rigorous *negative-result-plus-deployable-positive* paper.
Target: interpretability / "I Can't Believe It's Not Better" / negative-results
workshop at NeurIPS/ICLR/ICML; stretch COLM.

## Spine (tape to monitor)

Across an autoregressive LM (Pythia) and an encoder-based TSFM (Chronos-T5),
TopK-SAE features add **no incremental predictive power** for difficulty beyond
the strongest cheap baseline, yet carry a **(near-)significant causal
contribution** under reconstruction patching. Deployable artifact in both:
a Platt-recalibrated selective predictor on the cheap baseline, capturing
**30–41% of oracle AURC**, supporting a Pareto-dominating cost–quality cascade.

## Structure

1. **Intro** — routing/abstention problem; open question (do FM internals encode
   a *self*-difficulty signal beyond cheap baselines?); modality-generalization
   framing.
2. **Method** (one shared pipeline, described once):
   - TopK SAE (`core/sae.py`)
   - three-rung probe ladder + 2 diagnostics (`core/probe.py`)
   - paired-bootstrap ΔAUROC + label-permutation test (`core/stats.py`)
   - reconstruction-patching ablation, single- & all-position (`core/patching.py`)
   - risk-coverage + Platt/isotonic recalibration (`core/selective.py`,
     `core/calibration.py`)
   - **Unified leakage controls** as a contribution: purge gap, pretraining-
     contamination filter, train-only SAE fit, prompt-only (never gold-target)
     perplexity feature.
3. **Results — Modality A (LLM)** and **Modality B (TSFM)**: parallel subsections,
   the same four questions each (predictive ladder / diagnostics / causal
   ablation / deployable selective+cascade).
4. **Cross-modal synthesis** — the money table: predictive null + causal positive
   + deployable selective predictor, side by side for both modalities.
5. **Deployable artifact** — recalibrated cascades, Pareto frontiers, framed
   against RouteLLM / FrugalGPT.

## Seams to close before the synthesis table (reviewer-attack surface)

| seam | issue | fix in this repo |
|------|-------|------------------|
| Baseline mismatch | TSFM headline was SAE-vs-classical-stats; raw-activation **middle rung** was only a diagnostic | both modalities now route through `core.probe.run_probe_ladder` → identical 5-rung ladder; report Δ(P3−P2) on both |
| Causal sample size | TSFM ablation thin (167 windows × top-5; ΔCRPS CI straddles 0) | raise to LLM standard; `core.patching` parametrizes feature count + bootstrap |
| Single vs all-position | LLM found coverage-not-fidelity (all-position reveals effects single-position misses) | `core.patching.make_recon_hook(positions=...)` ports the knob to TSFM → **second cross-modal replication** if it holds |
| SAE expansion | 8× (LLM) vs 4× (TSFM, 512→4096) | `TopKSAE(expansion=...)`; align or show robustness to expansion |
| Layer choice | residual 12/18 vs encoder 3/5 | frame as "mid + late"; `layers()` exposes both |
| Causal metric | Δnats vs ΔCRPS | appropriately modality-specific; frame as parallel, recorded in summary |

## Synthesis table (REAL reproduced numbers — see `results/cross_modal_synthesis.md`)

```
                          | LLM/HellaSwag        | LLM/SQuAD            | TSFM/ETTh1
n test                    | 1499                 | 1500                | 167
P1 cheap AUROC            | 0.509                | 0.590               | 0.694
P2 cheap+raw AUROC        | 0.472                | 0.671               | 0.570
P3 cheap+SAE AUROC        | 0.500                | 0.592               | 0.523
Δ(P3−P2) AUROC [95% CI]   | +0.028[-0.001,+0.058]| -0.079[-0.118,-0.041]| -0.047[-0.192,+0.093]
Δ(P3−P1) AUROC [95% CI]   | -0.009[-0.039,+0.020]| +0.002[-0.044,+0.047]| -0.171[-0.291,-0.047]
causal sig (all-position) | 5/5                  | 5/5                 | (see results/)
causal sig (single-pos)   | 0/5                  | 2/5                 | (see results/)
selective: % oracle AURC  | 2.0% (P1)            | 41.3% (raw)         | 52.6% (P1)
cascade: Pareto-dom pts   | 1 (P1)               | 31 (SAE), 24 (P1)   | 7 (P1), 3 (SAE)
```

Notes:
- LLM rows reproduce the legacy reports **exactly**; TSFM uses a train-only label
  threshold (leakage fix), which shifts its AUROCs slightly from the legacy paper.
- HellaSwag is the weak modality (near-chance probes); SQuAD and ETTh1 carry the
  predictive-null + deployable-positive story.
- The causal coverage contrast (all 5/5 vs single 0–2/5) is the cross-modal
  replication; metric held continuous (Δnats / ΔCRPS) so coverage is not confounded
  with the binary/continuous choice.
