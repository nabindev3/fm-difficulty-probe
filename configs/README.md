# Configs

One YAML per (modality × experiment). The runner (`experiments/run.py`) reads
exactly one and drives the shared core through the matching adapter.

| config | modality | model | dataset | label |
|--------|----------|-------|---------|-------|
| `llm_hellaswag.yaml` | llm  | Pythia-410M / 2.8B | HellaSwag | 0/1 correctness |
| `llm_squad.yaml`     | llm  | Pythia-410M / 2.8B | SQuAD     | top-quartile gold perplexity |
| `tsfm_etth1.yaml`    | tsfm | Chronos-T5 small/base | ETTh1  | top-quartile normalized CRPS |

## Staging data

The `metadata` / `activations` / `sae_ckpt` paths point under `data/`. Copy or
symlink the legacy extraction outputs there, e.g.:

```bash
mkdir -p data/llm/hellaswag
cp -r ../llm-sae-difficulty/{activations,activations_late,activations_base,sae} \
      data/llm/hellaswag/
```

(or adjust the YAML paths to point straight at the legacy trees / `_legacy/`).

## Adding a modality/experiment

1. Copy a YAML, set `modality:` to a key registered in `run.py:MODALITIES`.
2. Point the paths at that experiment's extraction outputs.
3. `python experiments/run.py --config configs/your.yaml --stage all`.
