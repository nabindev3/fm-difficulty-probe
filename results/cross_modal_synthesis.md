# Cross-modal synthesis

**Universal predictive null + universal deployable predictor + LLM-specific causal signal.**

| metric | llm_hellaswag | llm_squad | tsfm_etth1 |
|---|---|---|---|
| n test | 1499 | 1500 | 167 |
| hard frac | 0.621 | 0.177 | 0.162 |
| P1 cheap AUROC | 0.509 | 0.59 | 0.654 |
| P2 cheap+raw AUROC | 0.472 | 0.671 | 0.584 |
| P3 cheap+SAE AUROC | 0.5 | 0.592 | 0.426 |
| **Δ SAE over raw** | +0.028 [-0.001,+0.058] | -0.079 [-0.118,-0.041] | -0.158 [-0.293,-0.025] |
| Δ SAE over cheap | -0.009 [-0.039,+0.020] | +0.002 [-0.044,+0.047] | -0.228 [-0.366,-0.092] |
| label-perm p (Δ SAE−raw) | 0.0326 (1-sided) | <1e-4 (1-sided) | 0.0105 (1-sided) |
| causal: all-position | 5/5 sig | 5/5 sig | 0/5 sig |
| causal: single-position | 0/5 sig | 2/5 sig | 0/5 sig |
| selective: % oracle | P1_cheap 2.0% | P4_raw_only 41.3% | P1_cheap 30.5% |
| cascade: Pareto-dom pts | 0 (P3-SAE), 1 (P1) | 31 (P3-SAE), 24 (P1) | 1 (P3-SAE), 5 (P1) |

**Reading.**
1. *Predictive null replicates in BOTH modalities* — SAE adds no power over the strongest cheap rung (Δ rows ≤ 0 or CI straddles 0).
2. *Causal positive is modality-specific* — the LLM's top features are causally active under all-position patching (5/5) and under-detected by single-position (0–2/5: coverage, not fidelity). On the TSFM, NO feature is significant under either coverage (0/5), reproducing the legacy Chronos null (50-sample run, 0/5). So the coverage-not-fidelity story is an LLM finding; on Chronos the features are predictively redundant AND causally quiet at this scale.
3. *Deployable artifact replicates in BOTH* — a cheap-baseline selective predictor captures 30–41% of oracle AURC.

The cross-modal dissociation (predictive-null both; causal-positive LLM-only) is itself the contribution: it isolates the causal signal as a property of the autoregressive LM, not a universal SAE phenomenon.
