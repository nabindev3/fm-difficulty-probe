# data/ — staged model artifacts (not version-controlled)

Everything under `data/` except this README is **gitignored**: the activation
tensors and SAE checkpoints total ~1.2 GB and live in the two legacy extraction
repos, not here. `reproduce.sh` step 1 populates this directory with symlinks
into sibling clones of those repos — nothing is copied.

## Layout after staging

```
data/
├── llm/
│   ├── hellaswag/                 # both point at the SAME LLM-repo dirs;
│   └── squad/                     # configs select the dataset within them
│       ├── activations/          -> ../llm-sae-difficulty/activations/
│       ├── activations_late/     -> ../llm-sae-difficulty/activations_late/
│       ├── activations_base/     -> ../llm-sae-difficulty/activations_base/
│       └── sae/                  -> ../llm-sae-difficulty/sae/
└── tsfm/
    └── etth1/
        ├── activations/          -> ../tsfm-sae-difficulty/activations/
        ├── activations_late/     -> ../tsfm-sae-difficulty/activations_late/
        ├── activations_base/     -> ../tsfm-sae-difficulty/activations_base/
        └── sae/                  -> ../tsfm-sae-difficulty/sae/
```

Approximate sizes (resolved through the symlinks): LLM activations ~51 MB per
layer dir + SAE checkpoints ~192 MB; TSFM activations ~351 MB per layer dir +
SAE checkpoints ~144 MB.

## How to obtain the artifacts

1. **Clone the legacy repos beside this one** (the staging step looks for them
   at `../llm-sae-difficulty` / `../tsfm-sae-difficulty`, accepting the older
   `…-routing` names too; override with `LLM=/path TSFM=/path bash reproduce.sh`):

   - https://github.com/nabindev3/llm-sae-difficulty   (Pythia-410m, HellaSwag + SQuAD)
   - https://github.com/nabindev3/tsfm-sae-difficulty  (Chronos-T5-small, ETTh1)

2. **LLM artifacts — download.** The large tensors and checkpoints are in a
   public HuggingFace dataset; from inside `llm-sae-difficulty/` run:

   ```bash
   bash download_artifacts.sh   # hf.co/datasets/nabindev3/llm-sae-difficulty-artifacts
   ```

3. **TSFM artifacts — regenerate.** There is no hosted copy; ETTh1 is small and
   public, so from inside `tsfm-sae-difficulty/` follow its README /
   `reproduce.sh` to run `extract_activations.py` (downloads ETTh1 and
   Chronos-T5-small from HF) and `sae/train_sae.py` for the checkpoints.

4. **Stage.** From this repo's root, `bash reproduce.sh` creates the symlinks
   above (idempotent) before running the pipeline. No further data setup is
   needed — the metadata parquets ride along in the same directories.
