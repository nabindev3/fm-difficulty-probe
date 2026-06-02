# fm-difficulty-probe

A modality-agnostic study of whether **TopK sparse-autoencoder features encode a
self-difficulty signal** in foundation models — replicated across an
autoregressive LM (Pythia) and an encoder-based time-series FM (Chronos-T5).

> This repo merges [`llm-sae-difficulty`](https://github.com/nabindev3/llm-sae-difficulty)
> (Pythia) and [`tsfm-sae-difficulty`](https://github.com/nabindev3/tsfm-sae-difficulty)
> (Chronos-T5) into one shared pipeline; see those for per-modality detail and the
> original workshop reports.

## Thesis (as tested) and what the data actually showed

**Going-in thesis.** Across an autoregressive LM and an encoder-based TSFM,
TopK-SAE features add **no incremental predictive power** for difficulty beyond
the strongest cheap baseline, yet carry a **(near-)significant causal
contribution** under reconstruction patching; the deployable artifact in both is
a selective predictor on the cheap baseline capturing **30–41% of oracle AURC**.

**What the unified runs showed.** Two of the three legs replicate cleanly; one
does not — and the divergence is the contribution:

> The **predictive null** and the **deployable selective predictor** replicate in
> BOTH modalities. The **causal contribution does not**: SAE features are causally
> active in the LM (5/5 features under all-position patching) but causally quiet
> in the TSFM (0/5 under either coverage, reproducing the legacy Chronos null).
> The causal signal is thus a property of the autoregressive LM, not a universal
> SAE phenomenon.

This is the roadmap's "either outcome is publishable" case: a *universal
predictive null + a modality-specific causal positive* is a sharper, more
falsifiable claim than a forced two-way replication.

## Why this layout

The project lives or dies on whether **both modalities run the same pipeline and
report the same metrics**. So the code is split into:

- **`core/`** — modality-agnostic. Pure numpy/sklearn/torch-SAE. No model code.
  Unit-tested on synthetic arrays (`pytest tests/`).
- **`modalities/`** — thin adapters implementing one `Modality` interface
  (`modalities/base.py`). Each owns its model, dataset, features, and CV scheme.
- **`experiments/run.py`** — one config-driven entrypoint; no per-modality
  branches beyond picking the adapter.
- **`configs/`** — one YAML per (modality × experiment).

```
core/
  sae.py          TopK SAE (aux dead-feature revival; expansion & k configurable)
  probe.py        three-rung (+2 diagnostic) ladder, paired-bootstrap ΔAUROC
  calibration.py  ECE/Brier + Platt/isotonic 5-fold OOF recalibration
  selective.py    risk-coverage curves, AURC, oracle, "% of oracle captured"
  cascade.py      cheap↔expensive routing, Pareto frontier
  patching.py     reconstruction-patch ablation: single- AND all-position
  stats.py        ProbeResult, paired bootstrap, label-permutation test
modalities/
  base.py         the Modality Protocol every backend satisfies
  llm.py          Pythia-410M/2.8B + HellaSwag / SQuAD
  tsfm.py         Chronos-T5 small/base + ETTh1 forecast windows
```

## The probe ladder (the apples-to-apples comparator)

Both modalities report the identical five-rung ladder:

| rung | features | role |
|------|----------|------|
| `P1_cheap`     | cheap baseline (lexical stats \| classical TS stats) | floor |
| `P2_cheap_raw` | cheap + **raw activations** | the crucial middle rung |
| `P3_cheap_sae` | cheap + SAE codes | the claim under test |
| `P4_raw_only`  | raw activations only | diagnostic |
| `P5_sae_only`  | SAE codes only | diagnostic |

Headline number: **Δ(P3 − P2)** — SAE *over raw activations*, with a
paired-bootstrap CI and a label-permutation p-value. The reconciliation work
that makes this one paper is forcing **both** modalities through this exact
ladder (the legacy TSFM repo reported SAE-vs-classical-stats and kept the raw
middle rung only as a diagnostic; routing it through `core.probe` promotes it).

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -q                 # model-free; proves the core is modality-agnostic

# Roadmap interface: pick a backend + analysis stage.
python experiments/run.py --modality llm  --dataset squad --experiment all
python experiments/run.py --modality tsfm --experiment probe
python experiments/run.py --modality tsfm --experiment causal     # single vs all-position
# (or point at an explicit config: --config configs/llm_squad.yaml)

# Regenerate the entire results table end-to-end, unattended:
bash reproduce.sh                       # full table (LLM causal reused from legacy)
RUN_EXPANSION=1 bash reproduce.sh       # + SAE expansion-robustness sweep
REPROBE_LLM_CAUSAL=1 bash reproduce.sh  # + recompute LLM causal on Pythia
FAST=1 bash reproduce.sh                # probe table only (skip heavy causal)
```

Every experiment writes a uniform artifact pair — `<name>.json` (full detail) +
`<name>.parquet` (flat table) — plus PNG figures (risk-coverage, Pareto, reliability)
under `results/<experiment>/`. Guardrails: the runner refuses a missing /
non-TopKSAE / random-init checkpoint and single-class labels, and sets the
threading backend + MPS device for Apple Silicon. (Activation *extraction* and its
`--skip_predict` flag remain in the legacy `extract_activations.py`; this repo
starts from cached activations — see `configs/README.md` for staging.)

## Reproduced results (real runs, in `results/`)

The unified pipeline reproduces the legacy LLM numbers **exactly** and the TSFM
numbers qualitatively (with an improved train-only label threshold). The
cross-modal synthesis lives in `results/cross_modal_synthesis.md`:

All three runs reproduce the legacy headline numbers through the shared code
(the Phase-2 regression gate): LLM SQuAD/L18 raw = **0.716**; TSFM Δ(SAE−cheap)
= **−0.227** (legacy −0.228); selective **41%** (LLM) / **30%** (TSFM).

| | HellaSwag | SQuAD (mid) | ETTh1 |
|---|---|---|---|
| P1 cheap AUROC | 0.509 | 0.590 | 0.654 |
| P2 cheap+raw   | 0.472 | 0.671 | 0.584 |
| P3 cheap+SAE   | 0.500 | 0.592 | 0.426 |
| **Δ SAE over raw** | +0.028 [−0.001,+0.058] | −0.079 [−0.118,−0.041] | −0.158 [−0.291,−0.025] |
| Δ SAE over cheap | −0.009 [−0.039,+0.020] | +0.002 [−0.044,+0.047] | −0.227 [−0.365,−0.091] |
| causal all-pos | 5/5 sig | 5/5 sig | **0/5 sig** |
| causal single-pos | 0/5 sig | 2/5 sig | **0/5 sig** |
| selective % oracle | 2.0% (P1) | 41.3% (raw) | 30.5% (P1) |

(SQuAD raw rung peaks at the late layer: P4 raw-only L18 = 0.716; run with
`--layer late`.)

**What replicates, and what doesn't — the actual cross-modal finding:**
1. **Predictive null — both.** SAE adds no power over the strongest cheap rung in
   either modality.
2. **Causal positive — LLM only.** On Pythia the top features are causally active
   under all-position patching (5/5) and under-detected by single-position (0–2/5:
   coverage, not fidelity). On Chronos **no** feature is significant under either
   coverage (0/5) — reproducing the legacy Chronos null (50-sample run also 0/5),
   so it's not a sample-budget artifact. The causal signal is a property of the
   autoregressive LM, not a universal SAE phenomenon.
3. **Deployable artifact — both.** A cheap-baseline selective predictor captures
   30–41% of oracle AURC (each modality on its natural error scale: binary
   correctness for the LLM, continuous CRPS for the TSFM).

That divergence (universal predictive null, LLM-specific causal contribution) is
the paper's contribution, not a problem — see `SEAMS.md` §2.

```bash
# reproduce everything (after staging data/ — see configs/README.md)
bash reproduce.sh
# causal coverage replication (needs the live models, cached locally):
USE_TF=0 python experiments/causal_tsfm.py --config configs/tsfm_etth1.yaml --positions all
USE_TF=0 python experiments/causal_tsfm.py --config configs/tsfm_etth1.yaml --positions last
python experiments/synthesize.py
```

## Status / roadmap

- [x] **Phase 0** repo architecture (this layout)
- [x] **Phase 1** shared core extracted + unit-tested on synthetic arrays
- [x] **Phase 2** adapters wired against real extraction outputs; legacy LLM
      numbers reproduce exactly through the unified ladder
- [x] **Phase 3** seams closed — see `SEAMS.md`:
      (a) identical three-rung ladder in both modalities at **both** layers (TSFM
      P2 middle rung promoted to headline; mid + late);
      (b) single- AND all-position causal results on Chronos via `core.patching`
      (the coverage experiment — which produced the LLM-vs-TSFM divergence);
      (c) **expansion-robustness sweep run** (LLM 4×↔8×, TSFM 8×↔4×): null holds
      at both widths — `results/expansion_robustness.md`.
      Plus: train-only-threshold leakage fix (opt-in) and natural-scale selective
      error, unifying leakage controls across modalities.
- [x] **Phase 4** unified runner (`--modality`/`--experiment`, incl. `causal` &
      `calibrate`), uniform json+parquet+PNG artifacts, guardrails (SAE/label
      refusal, MPS threading), and an unattended `reproduce.sh` that regenerates
      the full results table
- [ ] **Phase 5/6** cross-modal paper from `paper/outline.md` + `results/cross_modal_synthesis.md`
      (synthesis table auto-generated by `experiments/synthesize.py`)

See `SEAMS.md` for the seam-by-seam reconciliation and `paper/outline.md` for the
manuscript skeleton.
