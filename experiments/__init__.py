"""Entrypoints and modality-specific drivers (config-driven runner, causal
ablations, SAE training, cross-modal synthesis). Made a package so the runner can
`from experiments import causal_tsfm` and call its `run_causal` in-process."""
