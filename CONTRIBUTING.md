# Contributing to fm-difficulty-probe

Thanks for your interest. This is a research repository behind a cross-modal
paper, so the bar is less "ship a feature" and more "keep the results
reproducible and the two modalities symmetric." A few conventions make that easy.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -r requirements.txt && pip install -e .   # editable: no sys.path hacks
pip install -e ".[dev]"                               # adds pytest
pytest -q                                             # model-free core tests
```

For runs that must reproduce the committed CIs bit-for-bit, install the pinned
environment instead of the floors:

```bash
pip install -r requirements.lock && pip install -e . --no-deps
```

## Layout (where things go)

- `core/` — modality-agnostic, numpy/sklearn in → results out. **No model code,
  no `print`.** Anything truly modality-independent lives here and must be
  unit-testable on synthetic arrays (`tests/`).
- `modalities/` — thin adapters implementing the `Modality` protocol
  (`modalities/base.py`). Model/data specifics live here, never in `core/`.
- `experiments/` — entrypoints (the config-driven `run.py`, causal drivers,
  `train_sae.py`, `synthesize.py`).
- `configs/` — one YAML per `(modality × experiment)`; auto-discovered.
- `paper/` — the manuscript (separate license; see below).

## Conventions

- **Logging, not print.** Entrypoints configure logging via
  `core._log.setup_logging` and emit through a module logger
  (`log = logging.getLogger(__name__)`). Use `log.info` for stage results,
  `log.debug` for detail. The only legitimate `print` is a command's actual
  report to stdout (e.g. the synthesis table). Every entrypoint gets `-v`/`-q`
  for free via `core._log.add_logging_args`.
- **Add a modality/experiment by adding a file, not by editing code.** Drop a
  YAML in `configs/` with `modality:` + `experiment:` set (see
  `configs/README.md`); `run.py` discovers it. New backends register their
  adapter class in `run.py:MODALITIES`.
- **Both modalities run the same core.** If you change the probe ladder,
  calibration, selective prediction, cascade, or patching, it must apply to LLM
  *and* TSFM identically — that symmetry is the paper's whole claim.
- **Reproducibility is a gate.** `core/_repro.py` pins single-thread BLAS and
  probes seed liblinear. Don't introduce nondeterminism in the hot path, and if
  a change moves a committed number, regenerate the affected `results/` artifact
  and say so in the PR. Prefer changes that are provably bit-stable (see how the
  selective-bootstrap vectorization was verified against the old code).
- **Security.** Load checkpoints with `torch.load(..., weights_only=True)`.

## Tests

`pytest -q` must stay green. New core functionality needs a synthetic test in
`tests/` (no model, no network). The causal-ablation path is covered by
`tests/test_patching.py` — extend it if you touch `core/patching.py`.

## Pull requests

- Keep PRs focused; describe what moved and why.
- Note any change to committed `results/` numbers and how you regenerated them.
- Run `pytest -q` and, if you touched a stage, the relevant
  `python experiments/run.py --config … --experiment …` before submitting.

## Licensing of contributions

By contributing, you agree that your contributions to the **software** are
licensed under the MIT License (`LICENSE`), and contributions to the
**manuscript** under `paper/` are licensed under CC BY 4.0 (`paper/LICENSE`).
