#!/usr/bin/env bash
# Reproduce the full cross-modal study from extracted activations.
#
# Prerequisite: the legacy extraction outputs (activations + SAE checkpoints +
# metadata parquets) are staged under data/llm/... and data/tsfm/... to match
# the paths in configs/*.yaml. The extraction + SAE-training scripts themselves
# live in the legacy repos (see _legacy/ if you imported them).
#
# This script runs only the SHARED, model-free analysis stages. The causal
# ablation needs the live model and is launched separately per modality.
set -euo pipefail

echo "== core unit tests (model-free) =="
python -m pytest tests/ -q

for cfg in configs/llm_hellaswag.yaml configs/llm_squad.yaml configs/tsfm_etth1.yaml; do
  echo ""
  echo "== $cfg =="
  python experiments/run.py --config "$cfg" --stage probe
  python experiments/run.py --config "$cfg" --stage selective
  python experiments/run.py --config "$cfg" --stage cascade
done

echo ""
echo "Done. Per-experiment JSON under results/. Build the cross-modal"
echo "synthesis table from results/*/probe_results.json (see paper/outline.md)."
