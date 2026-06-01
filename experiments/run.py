"""Single config-driven entrypoint for fm-difficulty-probe.

    python experiments/run.py --config configs/llm_hellaswag.yaml --stage probe
    python experiments/run.py --config configs/tsfm_etth1.yaml   --stage all

Every (modality x experiment) is one YAML. The runner builds the adapter, then
drives the SHARED core through the Modality interface — it has no modality-
specific branches beyond instantiating the right adapter class. That is what
makes the cross-modal comparison apples-to-apples: identical code path, identical
metrics object, for Pythia and Chronos alike.

Stages:
  probe      three-rung (+diagnostic) ladder, paired-bootstrap ΔAUROC.
  selective  risk-coverage / AURC vs oracle & random.
  calibrate  Platt + isotonic OOF recalibration of P1 and P3.
  cascade    cheap<->expensive routing Pareto frontier.
  all        every stage above (causal ablation is run separately; it needs the
             live model and is modality-specific — see eval adapters).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from core import probe as probe_core
from core import selective as selective_core
from core import calibration as calib_core
from core import cascade as cascade_core


MODALITIES = {"llm": ("modalities.llm", "LLMModality"),
              "tsfm": ("modalities.tsfm", "TSFMModality")}


def build_modality(cfg: dict):
    import importlib
    kind = cfg["modality"]
    mod_path, cls_name = MODALITIES[kind]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(cfg)


def stage_probe(m, cfg, out_dir) -> tuple[dict, dict]:
    layer = cfg.get("layer", "mid")
    features = probe_core.build_ladder(
        cheap=m.cheap_baseline_features(),
        raw_agg=m.raw_activations(layer),
        sae_agg=m.sae_codes(layer),
    )
    y = m.difficulty_labels()
    train_mask, test_mask = m.split_masks()
    folds = m.cv_folds(train_mask, y)

    result, preds = probe_core.run_probe_ladder(
        features, y, train_mask, test_mask, folds,
        n_boot=cfg.get("n_boot", 2000), seed=cfg.get("seed", 42),
    )
    _dump(result.to_dict(), out_dir, "probe_results.json")
    _print_ladder(result)
    # Persist the per-rung test predictions for downstream stages.
    np.savez(os.path.join(out_dir, "probe_preds.npz"),
             **{k: v for k, v in preds.items()},
             y_test=y[test_mask])
    return result.to_dict(), preds


def stage_selective(m, cfg, out_dir, preds=None):
    y = m.difficulty_labels()
    _, test_mask = m.split_masks()
    # Use the modality's natural error scale (binary correctness | continuous CRPS)
    # so "% of oracle AURC captured" is comparable across modalities.
    if hasattr(m, "selective_error"):
        errors = np.asarray(m.selective_error(test_mask), dtype=float)
    else:
        errors = y[test_mask].astype(float)
    if preds is None:
        preds = _load_preds(out_dir)
    score_dict = {k: v for k, v in preds.items() if k != "y_test"}
    summary = selective_core.selective_prediction(
        score_dict, errors, n_bootstrap=cfg.get("n_boot", 2000), seed=cfg.get("seed", 42))
    _dump(summary, out_dir, "selective_prediction.json")
    print(f"\n[selective] oracle AURC={summary['oracle_aurc']:.4f}  "
          f"random AURC={summary['random_aurc']:.4f}")
    for name, r in summary["probes"].items():
        print(f"  {name:14s} AURC={r['aurc']:.4f}  "
              f"captures {100*r['fraction_of_oracle_aurc']:.1f}% of oracle")
    return summary


def stage_cascade(m, cfg, out_dir, preds=None):
    if preds is None:
        preds = _load_preds(out_dir)
    err_cheap, err_exp = m.cascade_errors()
    cost_cheap, cost_exp = m.cascade_costs()
    score_cols = cfg.get("cascade", {}).get(
        "score_cols", [probe_core.P3, probe_core.P1])
    score_dict = {c: preds[c] for c in score_cols if c in preds}
    summary = cascade_core.cascade(
        score_dict, err_cheap, err_exp, cost_cheap, cost_exp,
        n_random_trials=cfg.get("cascade", {}).get("n_random_trials", 500),
        seed=cfg.get("seed", 42))
    _dump(summary, out_dir, "cascade_results.json")
    for name, r in summary["probes"].items():
        print(f"[cascade] {name}: {r['n_dominating_points']} Pareto-dominating points")
    return summary


def _dump(obj, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {os.path.join(out_dir, fname)}")


def _load_preds(out_dir):
    data = np.load(os.path.join(out_dir, "probe_preds.npz"))
    return {k: data[k] for k in data.files}


def _print_ladder(result):
    print(f"\n=== Probe ladder  (n_test={result.n_test}, "
          f"hard_frac={result.hard_fraction:.3f}) ===")
    for name, pr in result.probes.items():
        print(f"  {name:14s} AUROC={pr.auroc:.3f}  "
              f"95% CI [{pr.ci_low:.3f}, {pr.ci_high:.3f}]  (C={pr.best_C})")
    print("  --- incremental power (paired bootstrap) ---")
    for key, d in result.deltas.items():
        print(f"  Δ {key:24s} {d['point']:+.3f}  "
              f"95% CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all",
                    choices=["probe", "selective", "calibrate", "cascade", "all"])
    ap.add_argument("--layer", default=None, choices=["mid", "late"],
                    help="Override the cfg 'layer' (mid|late) for this run.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.layer:
        cfg["layer"] = args.layer
        cfg["out_dir"] = cfg.get("out_dir", "results/run") + f"_{args.layer}"
    out_dir = cfg.get("out_dir", f"results/{os.path.splitext(os.path.basename(args.config))[0]}")
    m = build_modality(cfg)

    preds = None
    if args.stage in ("probe", "all"):
        _, preds = stage_probe(m, cfg, out_dir)
    if args.stage in ("selective", "all"):
        stage_selective(m, cfg, out_dir, preds)
    if args.stage in ("cascade", "all"):
        stage_cascade(m, cfg, out_dir, preds)
    # calibrate stage intentionally requires the raw feature matrices; wire it in
    # per-modality once activations are available (see core.calibration.recalibrate_oof).


if __name__ == "__main__":
    main()
