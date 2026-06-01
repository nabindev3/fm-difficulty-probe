#!/usr/bin/env bash
# Regenerate the full cross-modal results table end-to-end, unattended.
#
#   bash reproduce.sh                 # full table (LLM causal reused from legacy)
#   REPROBE_LLM_CAUSAL=1 bash reproduce.sh   # also recompute LLM causal (Pythia, ~1h/config)
#   RUN_EXPANSION=1 bash reproduce.sh        # also run the SAE expansion-robustness sweep
#   FAST=1 bash reproduce.sh                 # skip the heavy causal stages (probe table only)
#
# Prereq: the two legacy repos sit beside this one (../llm-sae-difficulty,
# ../tsfm-sae-routing) with their activations/ + sae/ checkpoints present. We
# symlink them under data/ (idempotent). No model downloads beyond the HF cache
# the causal stages need (Pythia-410m, Chronos-t5-small).
set -euo pipefail
export USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_ADVISORY_WARNINGS=1
PY=${PY:-python3}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "== [0/6] core unit tests (model-free) =="
$PY -m pytest tests/ -q

echo "== [1/6] stage legacy data under data/ (idempotent) =="
LLM=../llm-sae-difficulty
TSFM=../tsfm-sae-routing
[ -d "$LLM/activations" ] || { echo "missing $LLM/activations — clone/extract the legacy LLM repo"; exit 1; }
[ -d "$TSFM/activations" ] || { echo "missing $TSFM/activations — clone/extract the legacy TSFM repo"; exit 1; }
mkdir -p data/llm/hellaswag data/llm/squad data/tsfm/etth1
for sub in hellaswag squad; do
  for d in activations activations_late activations_base sae; do
    ln -sfn "$(cd "$LLM" && pwd)/$d" "data/llm/$sub/$d"
  done
done
for d in activations activations_late activations_base sae; do
  ln -sfn "$(cd "$TSFM" && pwd)/$d" "data/tsfm/etth1/$d"
done

echo "== [2/6] probe ladder + selective + cascade + calibrate (all configs) =="
for cfg in configs/llm_hellaswag.yaml configs/llm_squad.yaml configs/tsfm_etth1.yaml; do
  echo "--- $cfg ---"
  $PY experiments/run.py --config "$cfg" --experiment all
done
# Late-layer ladders (the 'both layers' requirement).
$PY experiments/run.py --config configs/llm_squad.yaml  --experiment probe --layer late
$PY experiments/run.py --config configs/tsfm_etth1.yaml --experiment probe --layer late

if [ "${FAST:-0}" != "1" ]; then
  echo "== [3/6] TSFM causal: all- vs single-position (Chronos, ~45 min) =="
  $PY experiments/run.py --config configs/tsfm_etth1.yaml --experiment causal

  echo "== [4/6] LLM causal =="
  if [ "${REPROBE_LLM_CAUSAL:-0}" = "1" ]; then
    echo "  recomputing via causal_llm.py (Pythia, ~1h/config)"
    for cfg in configs/llm_hellaswag.yaml configs/llm_squad.yaml; do
      $PY experiments/causal_llm.py --config "$cfg" --positions all
      $PY experiments/causal_llm.py --config "$cfg" --positions boundary
    done
    # boundary -> last for uniform naming
    for r in llm_hellaswag llm_squad; do
      [ -f results/$r/causal_ablation_boundary.json ] && mv results/$r/causal_ablation_boundary.json results/$r/causal_ablation_last.json
    done
  else
    echo "  reusing the validated legacy LLM causal results (identical experiment)"
    L="$LLM/eval/results"
    cp "$L/allpos/hellaswag_causal_ablation.json"        results/llm_hellaswag/causal_ablation_all.json
    cp "$L/disentangle/hellaswag_causal_ablation.json"   results/llm_hellaswag/causal_ablation_last.json
    cp "$L/squad/allpos/squad_causal_ablation.json"      results/llm_squad/causal_ablation_all.json
    cp "$L/squad/disentangle/squad_causal_ablation.json" results/llm_squad/causal_ablation_last.json
  fi
else
  echo "== [3-4/6] FAST=1: skipping causal stages =="
fi

if [ "${RUN_EXPANSION:-0}" = "1" ]; then
  echo "== [5/6] SAE expansion-robustness sweep =="
  $PY experiments/train_sae.py --activations data/llm/squad/activations/squad_activations.safetensors \
      --metadata data/llm/squad/activations/squad_metadata.parquet --expansion 8 --out saes_sweep/llm_squad_exp8.pt
  $PY experiments/train_sae.py --activations data/tsfm/etth1/activations/ETTh1_activations.safetensors \
      --metadata data/tsfm/etth1/activations/ETTh1_metadata.parquet --expansion 4 --out saes_sweep/tsfm_mid_exp4.pt
  $PY experiments/run.py --config configs/llm_squad.yaml  --experiment probe --sae_override saes_sweep/llm_squad_exp8.pt --tag exp8
  $PY experiments/run.py --config configs/tsfm_etth1.yaml --experiment probe --sae_override saes_sweep/tsfm_mid_exp4.pt --tag exp4
else
  echo "== [5/6] expansion sweep skipped (set RUN_EXPANSION=1 to run) =="
fi

echo "== [6/6] cross-modal synthesis table =="
$PY experiments/synthesize.py

echo ""
echo "Done. Full results table: results/cross_modal_synthesis.md"
echo "Per-experiment json+parquet+png under results/<experiment>/."
