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
| point [95% CI] | +0.028 [−0.001,+0.058] | −0.079 [−0.118,−0.041] | −0.158 [−0.291,−0.025] |
| Δ SAE over cheap | −0.009 [−0.039,+0.020] | +0.002 [−0.044,+0.047] | −0.227 [−0.365,−0.091] |

In no modality does the SAE add predictive power over raw activations.

**Phase-2 regression gate — PASSED.** All legacy headline numbers reproduce
through the shared code before any new experiment was trusted:
- LLM SQuAD/L18 raw-only AUROC = **0.716** (`--layer late`).
- LLM SQuAD/L12 raw-only = 0.667; full mid ladder reproduces the legacy JSON exactly.
- TSFM Δ(SAE − cheap) = P3 − P1 = **−0.227** (legacy headline −0.228).
- Selective: LLM raw **41.3%** of oracle, TSFM P1 **30.5%** of oracle.

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

Significant features, holding metric fixed (continuous / CRPS), 167 windows each:

| significant features | all-position | single-position |
|---|---|---|
| LLM HellaSwag | 5/5 | 0/5 |
| LLM SQuAD     | 5/5 | 2/5 |
| TSFM ETTh1    | **0/5** | **0/5** |

**Result: the causal signal does NOT replicate on the TSFM.** On Chronos, no
top-feature ablation is significant under either coverage — and this reproduces
the legacy Chronos run (50 samples, all-position, also 0/5; best feature CI
[−0.003, 0.050]), so it is not an artifact of our reduced sample budget. The
coverage-not-fidelity phenomenon is therefore an **LLM finding**, not a universal
SAE property: on the autoregressive LM, all-position patching reveals effects
that single-position misses (coverage, not fidelity); on the TSFM the features
are predictively redundant *and* causally quiet.

This divergence is a feature, not a bug, of the cross-modal design (the roadmap's
"either outcome is publishable"): the predictive null is universal, but the
causal contribution is specific to the autoregressive LM. See the synthesis
"Reading" in `results/cross_modal_synthesis.md`.

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
into the threshold.

`label_threshold_split` in the TSFM config selects where the threshold is computed:
- `all` (default) reproduces the legacy number exactly (Δ(SAE−cheap) = −0.227),
  so the Phase-2 regression gate passes and the refactor is verified correct.
- `train` uses a train-only quantile — the honest leakage fix matching the
  LLM-side discipline (train-only SAE fit, prompt-only perplexity, purge gap).

Impact of switching to train-only: P1 0.654→0.694, Δ(SAE−cheap) −0.227→−0.171 —
the predictive-null conclusion is unchanged, so the fix is safe to adopt for the
paper while citing the legacy number as the regression anchor.

## 7. Selective-prediction error scale  ✅ FRAMED

Risk-coverage "% of oracle AURC captured" must use each modality's NATURAL error
scale or the numbers aren't comparable: binary 0/1 correctness for the LLM,
continuous CRPS for the TSFM (risk = mean CRPS on retained windows). The unified
runner reads this from `Modality.selective_error(test_mask)`, reproducing both
legacy figures: LLM raw **41.3%**, TSFM P1 **30.5%** of oracle.
