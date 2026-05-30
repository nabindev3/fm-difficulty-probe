# fm-difficulty-probe

A modality-agnostic study of whether **TopK sparse-autoencoder features encode a
self-difficulty signal** in foundation models — replicated across an
autoregressive LM (Pythia) and an encoder-based time-series FM (Chronos-T5).

## Thesis

> Across an autoregressive LM and an encoder-based TSFM, TopK-SAE features add
> **no incremental predictive power** for difficulty beyond the strongest cheap
> baseline, yet carry a **(near-)significant causal contribution** under
> reconstruction patching. The deployable artifact in both modalities is a
> **Platt-recalibrated selective predictor** on the cheap baseline, capturing
> **30–41% of oracle AURC**.

The result is a *predictive null + causal positive* dissociation that replicates
in two unrelated modalities — much stronger than either single negative result,
which would invite "you just did it wrong."

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

# Stage the legacy extraction outputs under data/ to match configs/*.yaml, then:
python experiments/run.py --config configs/llm_hellaswag.yaml --stage all
python experiments/run.py --config configs/tsfm_etth1.yaml    --stage all
bash reproduce.sh                # runs every model-free stage for all configs
```

## Status / roadmap

- [x] **Phase 0** repo architecture (this layout)
- [x] **Phase 1** shared core extracted + unit-tested on synthetic arrays
- [ ] **Phase 2** wire adapters against real extraction outputs; confirm legacy
      numbers reproduce through the unified ladder
- [ ] **Phase 3** close the seams: TSFM P2 middle rung as headline; bring the
      TSFM causal ablation up to the LLM standard (single- vs all-position
      patching — the second cross-modal replication); align SAE expansion or
      show robustness
- [ ] **Phase 4** cross-modal synthesis table + paper (`paper/outline.md`)

See `paper/outline.md` for the manuscript skeleton.
