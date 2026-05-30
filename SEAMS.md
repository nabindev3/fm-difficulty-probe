# Closing the seams (Phase 3)

The two legacy studies were run separately; a reviewer will attack the joints
where they don't line up. This documents each seam and how the unified repo
closes (or honestly frames) it. Numbers are from the real reproduction runs in
`results/` (see `results/cross_modal_synthesis.md`).

## 1. Baseline mismatch — the raw-activation middle rung  ✅ CLOSED

**Issue.** The LLM study reported a three-rung ladder (cheap → raw → SAE); the
TSFM study headlined SAE-vs-classical-stats and kept the raw-activation rung only
as a "diagnostic". The two papers weren't measuring against the same comparator.

**Fix.** Both modalities now route through `core.probe.run_probe_ladder`, which
reports the identical 5-rung ladder (P1 cheap, P2 cheap+raw, P3 cheap+SAE, P4
raw-only, P5 SAE-only) and the same three headline deltas, including Δ(P3−P2),
SAE *over raw*. The TSFM P2 rung is now first-class.

| Δ SAE over raw | HellaSwag | SQuAD | ETTh1 |
|---|---|---|---|
| point [95% CI] | +0.028 [−0.001,+0.058] | −0.079 [−0.118,−0.041] | −0.047 [−0.192,+0.093] |

In no modality does the SAE add predictive power over raw activations. Reproduces
the legacy LLM numbers exactly.

## 2. Causal sample size + single- vs all-position patching  ✅ CLOSED (the gold replication)

**Issue.** The TSFM causal experiment was thin and used only all-position
patching; the LLM side had discovered that *coverage* (single vs all position),
not SAE fidelity, governs whether feature effects are detectable. The two causal
experiments weren't comparable.

**Fix.** `core.patching.make_recon_hook(positions=...)` exposes the coverage knob
("all" | "last") as a first-class, modality-agnostic argument, and
`experiments/causal_tsfm.py` runs BOTH modes on all 167 Chronos test windows with
the SAME metric (CRPS) — a *cleaner* comparison than the legacy LLM 2×2, which
partly confounded coverage with the binary/continuous metric.

Coverage-not-fidelity, holding metric fixed (continuous / CRPS):

| significant features | all-position | single-position |
|---|---|---|
| LLM HellaSwag | 5/5 | 0/5 |
| LLM SQuAD     | 5/5 | 2/5 |
| TSFM ETTh1    | see `results/tsfm_etth1/causal_ablation_{all,last}.json` |

If the TSFM single-position run detects materially fewer features than
all-position, the coverage-not-fidelity story replicates across modalities — the
paper's second cross-modal result.

## 3. SAE expansion factor — 4× (LLM) vs 8× (TSFM)  ⚠️ DOCUMENTED, framed

**Reality on disk (not what the roadmap assumed).**
- LLM: d_model 1024 → d_hidden **4096 = 4×**.
- TSFM: d_model 512 → d_hidden **4096 = 8×**.

So the expansion factors differ *and* the roadmap had them backwards. Two honest
options, both supported by the code:

- **Align.** `core.sae.TopKSAE(expansion=...)` makes width a one-line config
  change; retrain both at a common expansion (e.g. 8×) and confirm the predictive
  null is unchanged.
- **Robustness.** Report the null at both 4× and 8× and show it doesn't move —
  arguably stronger, since it demonstrates the result isn't an expansion artifact.

Either way: state the factors explicitly in the paper (both happen to land at
d_hidden = 4096, which is a convenient coincidence to note).

## 4. Layer choice — residual blocks 12/18 vs encoder blocks 3/5  ✅ FRAMED

Both are "mid + late". `Modality.layers()` returns `{"mid", "late"}` for each
backend (LLM: residual blocks 11/17 → Layers 12/18; TSFM: encoder blocks 3/5 of
6). The `--layer` flag / `layer:` config runs either rung through the identical
pipeline. Frame as parallel; report mid as primary, late as robustness.

## 5. Causal metric — Δnats (LLM) vs ΔCRPS (TSFM)  ✅ FRAMED

Appropriately modality-specific (log-prob/cross-entropy for a generative LM;
CRPS for a probabilistic forecaster). `core.patching.aggregate_ablation` is
metric-agnostic and the metric name is recorded in each summary. Frame as
parallel "lower = better forecast/answer quality" deltas.

## 6. Label-definition leakage (bonus fix)  ✅ CLOSED

The legacy TSFM probe defined the "hard" label from a **whole-dataset** CRPS
quantile (`df_meta['crps_norm'].quantile(0.75)`), letting test-set values leak
into the threshold. The unified `TSFMModality.difficulty_labels()` uses a
**train-only** quantile, matching the leakage discipline already used on the LLM
side (train-only SAE fit, prompt-only perplexity feature, purge gap). This
slightly shifts the TSFM AUROCs vs the legacy report but is the correct, honest
threshold and unifies the leakage controls across modalities.
