"""Unified, config-driven entrypoint for fm-difficulty-probe.

Two equivalent ways to select what to run:

    # roadmap interface
    python experiments/run.py --modality llm  --dataset squad --experiment probe
    python experiments/run.py --modality tsfm --experiment all

    # explicit config (overrides --modality/--dataset)
    python experiments/run.py --config configs/tsfm_etth1.yaml --experiment selective

--experiment ∈ {probe, selective, calibrate, cascade, causal, all}. The runner
builds the adapter, then drives the SHARED core through the Modality interface —
no modality-specific branches beyond instantiating the right adapter (and the
causal forward, which is necessarily model-specific and lives in causal_*.py).

Every experiment writes a UNIFORM artifact pair: <name>.json (full detail) and
<name>.parquet (a flat table for spreadsheets / cross-run aggregation). Figures
(risk-coverage, Pareto frontier, reliability) are written as PNGs.

Guardrails kept from both legacy repos:
  * refuse to probe on a missing / non-TopKSAE / randomly-initialized checkpoint;
  * refuse on degenerate (single-class) labels;
  * threading backend + MPS device selection for Apple Silicon.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core._repro  # noqa: F401  — pins single-thread BLAS before numpy import

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import probe as probe_core
from core import selective as selective_core
from core import calibration as calib_core
from core import cascade as cascade_core


MODALITIES = {"llm": ("modalities.llm", "LLMModality"),
              "tsfm": ("modalities.tsfm", "TSFMModality")}

# (modality, dataset) -> config path. --dataset defaults per modality.
CONFIG_REGISTRY = {
    ("llm", "hellaswag"): "configs/llm_hellaswag.yaml",
    ("llm", "squad"): "configs/llm_squad.yaml",
    ("tsfm", "etth1"): "configs/tsfm_etth1.yaml",
}
DEFAULT_DATASET = {"llm": "squad", "tsfm": "etth1"}


# --------------------------------------------------------------------------- #
# Guardrails.
# --------------------------------------------------------------------------- #
def _check_sae(cfg: dict, layer: str):
    """Refuse to run on a missing or non-TopKSAE checkpoint (never silently probe
    on random weights — that produces noise that looks identical to a real result)."""
    import torch
    path = cfg["sae_ckpt"][layer]
    if not os.path.exists(path):
        sys.exit(f"[guardrail] SAE checkpoint not found: {path}. Train it first; "
                 f"refusing to probe with random weights.")
    state = torch.load(path, map_location="cpu")
    if "W_enc" not in state:
        sys.exit(f"[guardrail] {path} is not a TopKSAE checkpoint (no W_enc).")


def _check_labels(y: np.ndarray, train_mask, test_mask):
    for split, mask in (("train", train_mask), ("test", test_mask)):
        if int(mask.sum()) == 0:
            sys.exit(f"[guardrail] empty {split} split.")
        if len(np.unique(y[mask])) < 2:
            sys.exit(f"[guardrail] {split} split is single-class — labels look "
                     f"missing/degenerate; refusing to compute AUROC on it.")


def _setup_threads():
    """Threading backend / thread caps that keep Apple-Silicon (MPS) runs stable."""
    import torch
    try:
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    except Exception:
        pass
    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {dev}; torch threads={torch.get_num_threads()}")
    return dev


# --------------------------------------------------------------------------- #
# Artifact helpers (uniform json + parquet).
# --------------------------------------------------------------------------- #
def _dump_json(obj, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {os.path.join(out_dir, fname)}")


def _dump_parquet(df: pd.DataFrame, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    df.to_parquet(path)
    print(f"Saved {path}")


def _test_ids(m, test_mask):
    meta = getattr(m, "meta", None)
    if meta is not None and "window_id" in meta.columns:
        return meta.loc[test_mask, "window_id"].values
    return np.arange(int(test_mask.sum()))


def _load_preds(out_dir):
    data = np.load(os.path.join(out_dir, "probe_preds.npz"))
    return {k: data[k] for k in data.files}


# --------------------------------------------------------------------------- #
# Experiments.
# --------------------------------------------------------------------------- #
def stage_probe(m, cfg, out_dir):
    layer = cfg.get("layer", "mid")
    _check_sae(cfg, layer)
    y = m.difficulty_labels()
    train_mask, test_mask = m.split_masks()
    _check_labels(y, train_mask, test_mask)

    features = probe_core.build_ladder(
        cheap=m.cheap_baseline_features(),
        raw_agg=m.raw_activations(layer),
        sae_agg=m.sae_codes(layer))
    folds = m.cv_folds(train_mask, y)
    result, preds = probe_core.run_probe_ladder(
        features, y, train_mask, test_mask, folds,
        n_boot=cfg.get("n_boot", 2000), seed=cfg.get("seed", 42))

    _dump_json(result.to_dict(), out_dir, "probe_results.json")
    # Uniform parquet: one row per rung.
    rows = [{"rung": n, **pr.to_dict()} for n, pr in result.probes.items()]
    _dump_parquet(pd.DataFrame(rows), out_dir, "probe_results.parquet")
    # Per-item test scores parquet (for selective/cascade + audit).
    ids = _test_ids(m, test_mask)
    scores = pd.DataFrame({"id": ids, "y_test": y[test_mask]})
    for n, p in preds.items():
        scores[f"pred_{n}"] = p
    _dump_parquet(scores, out_dir, "probe_scores.parquet")
    np.savez(os.path.join(out_dir, "probe_preds.npz"),
             **preds, y_test=y[test_mask])
    _print_ladder(result)
    return result.to_dict(), preds


def stage_selective(m, cfg, out_dir, preds=None):
    y = m.difficulty_labels()
    _, test_mask = m.split_masks()
    errors = (np.asarray(m.selective_error(test_mask), dtype=float)
              if hasattr(m, "selective_error") else y[test_mask].astype(float))
    if preds is None:
        preds = _load_preds(out_dir)
    score_dict = {k: v for k, v in preds.items() if k != "y_test"}
    summary = selective_core.selective_prediction(
        score_dict, errors, n_bootstrap=cfg.get("n_boot", 2000), seed=cfg.get("seed", 42))
    _dump_json(summary, out_dir, "selective_prediction.json")

    rows = [{"probe": k, "aurc": v["aurc"],
             "fraction_of_oracle_aurc": v["fraction_of_oracle_aurc"]}
            for k, v in summary["probes"].items()]
    rows += [{"probe": "_oracle", "aurc": summary["oracle_aurc"], "fraction_of_oracle_aurc": 1.0},
             {"probe": "_random", "aurc": summary["random_aurc"], "fraction_of_oracle_aurc": 0.0}]
    _dump_parquet(pd.DataFrame(rows), out_dir, "selective_prediction.parquet")
    _plot_risk_coverage(summary, out_dir, title=cfg.get("experiment", ""))

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
    score_cols = cfg.get("cascade", {}).get("score_cols", [probe_core.P3, probe_core.P1])
    score_dict = {c: preds[c] for c in score_cols if c in preds}
    summary = cascade_core.cascade(
        score_dict, err_cheap, err_exp, cost_cheap, cost_exp,
        n_random_trials=cfg.get("cascade", {}).get("n_random_trials", 500),
        seed=cfg.get("seed", 42))
    _dump_json(summary, out_dir, "cascade_results.json")
    rows = [{"probe": k, "n_dominating_points": v["n_dominating_points"]}
            for k, v in summary["probes"].items()]
    _dump_parquet(pd.DataFrame(rows), out_dir, "cascade_results.parquet")
    _plot_pareto(summary, out_dir, cost_cheap, cost_exp)
    for name, r in summary["probes"].items():
        print(f"[cascade] {name}: {r['n_dominating_points']} Pareto-dominating points")
    return summary


def stage_calibrate(m, cfg, out_dir):
    """Platt + isotonic 5-fold OOF recalibration of P1 (cheap) and P3 (cheap+SAE)."""
    layer = cfg.get("layer", "mid")
    _check_sae(cfg, layer)
    y = m.difficulty_labels()
    train_mask, test_mask = m.split_masks()
    _check_labels(y, train_mask, test_mask)
    cheap = m.cheap_baseline_features()
    sae = m.sae_codes(layer)
    feats = {probe_core.P1: cheap,
             probe_core.P3: np.concatenate([cheap, sae], axis=1)}
    # Use the C the ladder picked if available, else a sensible default per rung.
    chosen = {}
    pr_path = os.path.join(out_dir, "probe_results.json")
    if os.path.exists(pr_path):
        d = json.load(open(pr_path))
        for rung in feats:
            chosen[rung] = d["probes"].get(rung, {}).get("best_C")
    out, fig_pts = {}, {}
    for rung, X in feats.items():
        C = chosen.get(rung) or (1.0 if rung == probe_core.P1 else 0.1)
        res = calib_core.recalibrate_oof(X[train_mask], y[train_mask],
                                         X[test_mask], y[test_mask], C=C)
        fig_pts[rung] = {tag: calib_core.reliability_points(y[test_mask], res["_test_probs"][tag])
                         for tag in ("raw", "platt", "isotonic")}
        out[rung] = {k: v for k, v in res.items() if k != "_test_probs"}
    _dump_json(out, out_dir, "recalibration_results.json")
    rows = [{"rung": rung, "variant": var, **vals}
            for rung, d in out.items() for var, vals in d.items()]
    _dump_parquet(pd.DataFrame(rows), out_dir, "recalibration_results.parquet")
    _plot_reliability(fig_pts, out_dir)
    for rung, d in out.items():
        print(f"[calibrate] {rung}: raw ECE {d['raw']['ece']:.3f} -> "
              f"Platt {d['platt']['ece']:.3f} / isotonic {d['isotonic']['ece']:.3f}")
    return out


def stage_causal(m, cfg, args, out_dir):
    """Dispatch to the modality-specific causal driver (both position modes)."""
    modality = cfg["modality"]
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    script = {"tsfm": "causal_tsfm.py", "llm": "causal_llm.py"}[modality]
    script_path = os.path.join(here, script)
    if not os.path.exists(script_path):
        print(f"[causal] {script} not present; skipping (LLM causal results may be "
              f"reused from legacy — see reproduce.sh).")
        return
    cfg_path = args.config or _resolve_config(args)
    env = dict(os.environ, USE_TF="0", USE_FLAX="0")
    for positions in ("all", "last" if modality == "tsfm" else "boundary"):
        cmd = [sys.executable, script_path, "--config", cfg_path, "--positions", positions]
        print(f"[causal] {' '.join(cmd)}")
        subprocess.run(cmd, env=env, check=False)


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
def _plot_risk_coverage(summary, out_dir, title=""):
    cov = summary["coverages"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cov, summary["oracle_curve"], "k-", lw=2, label=f"oracle (AURC {summary['oracle_aurc']:.3f})")
    ax.plot(cov, summary["random_curve"], "k--", label=f"random (AURC {summary['random_aurc']:.3f})")
    for name, r in summary["probes"].items():
        ax.plot(cov, r["curve"], marker="o", ms=3,
                label=f"{name} (AURC {r['aurc']:.3f}, {100*r['fraction_of_oracle_aurc']:.0f}% oracle)")
    ax.set_xlabel("coverage"); ax.set_ylabel("risk on retained (lower better)")
    ax.set_title(f"Risk-coverage {title}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "risk_coverage.png"), dpi=150)
    plt.close(fig); print(f"Saved {os.path.join(out_dir, 'risk_coverage.png')}")


def _plot_pareto(summary, out_dir, cost_cheap, cost_exp):
    fig, ax = plt.subplots(figsize=(8, 5))
    ca = summary["always_cheap"]; ea = summary["always_expensive"]
    ax.plot([ca["cost"], ea["cost"]], [ca["mean_error"], ea["mean_error"]],
            "gray", ls=":", label="linear interp")
    ox = [c for c, _, _ in summary["oracle_curve"]]; oy = [y for _, y, _ in summary["oracle_curve"]]
    ax.plot(ox, oy, "k-", lw=2, label="oracle")
    for name, r in summary["probes"].items():
        cx = [p["mean_cost"] for p in r["frontier"]]; cy = [p["mean_error"] for p in r["frontier"]]
        ax.plot(cx, cy, marker="o", ms=3, label=f"{name} ({r['n_dominating_points']} dom)")
    ax.scatter([ca["cost"]], [ca["mean_error"]], s=70, edgecolor="k", zorder=6, label="always cheap")
    ax.scatter([ea["cost"]], [ea["mean_error"]], s=70, edgecolor="k", zorder=6, label="always expensive")
    ax.set_xlabel("mean cost"); ax.set_ylabel("mean error (lower better)")
    ax.set_title("Cost–quality cascade"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pareto_frontier.png"), dpi=150)
    plt.close(fig); print(f"Saved {os.path.join(out_dir, 'pareto_frontier.png')}")


def _plot_reliability(fig_pts, out_dir):
    fig, axes = plt.subplots(1, len(fig_pts), figsize=(5.5 * len(fig_pts), 5), squeeze=False)
    for ax, (rung, variants) in zip(axes[0], fig_pts.items()):
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
        for var, (xs, ys) in variants.items():
            ax.plot(xs, ys, marker="o", ms=4, label=var)
        ax.set_title(rung); ax.set_xlabel("predicted P(hard)"); ax.set_ylabel("actual")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "reliability.png"), dpi=150)
    plt.close(fig); print(f"Saved {os.path.join(out_dir, 'reliability.png')}")


def _print_ladder(result):
    print(f"\n=== Probe ladder  (n_test={result.n_test}, "
          f"hard_frac={result.hard_fraction:.3f}) ===")
    for name, pr in result.probes.items():
        print(f"  {name:14s} AUROC={pr.auroc:.3f}  "
              f"95% CI [{pr.ci_low:.3f}, {pr.ci_high:.3f}]  (C={pr.best_C})")
    print("  --- incremental power (paired bootstrap) ---")
    for key, d in result.deltas.items():
        print(f"  Δ {key:24s} {d['point']:+.3f}  95% CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")


# --------------------------------------------------------------------------- #
# Config resolution + main.
# --------------------------------------------------------------------------- #
def _resolve_config(args) -> str:
    if args.config:
        return args.config
    if not args.modality:
        sys.exit("Provide either --config or --modality.")
    ds = args.dataset or DEFAULT_DATASET[args.modality]
    key = (args.modality, ds)
    if key not in CONFIG_REGISTRY:
        sys.exit(f"No config for {key}. Known: {sorted(CONFIG_REGISTRY)}")
    return CONFIG_REGISTRY[key]


def build_modality(cfg: dict):
    import importlib
    mod_path, cls_name = MODALITIES[cfg["modality"]]
    return getattr(importlib.import_module(mod_path), cls_name)(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["llm", "tsfm"])
    ap.add_argument("--dataset", default=None, help="hellaswag|squad|etth1 (default per modality).")
    ap.add_argument("--config", default=None, help="Explicit config path (overrides --modality/--dataset).")
    ap.add_argument("--experiment", "--stage", dest="experiment", default="all",
                    choices=["probe", "selective", "calibrate", "cascade", "causal", "all"])
    ap.add_argument("--layer", default=None, choices=["mid", "late"])
    ap.add_argument("--sae_override", default=None, help="Use this TopKSAE checkpoint (expansion sweep).")
    ap.add_argument("--tag", default=None, help="Suffix for out_dir.")
    args = ap.parse_args()

    args.config = _resolve_config(args)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("experiment", os.path.splitext(os.path.basename(args.config))[0])
    if args.layer:
        cfg["layer"] = args.layer
        cfg["out_dir"] = cfg.get("out_dir", "results/run") + f"_{args.layer}"
    if args.sae_override:
        cfg["sae_ckpt"][cfg.get("layer", "mid")] = args.sae_override
    if args.tag:
        cfg["out_dir"] = cfg.get("out_dir", "results/run") + f"_{args.tag}"
    out_dir = cfg.get("out_dir", f"results/{os.path.splitext(os.path.basename(args.config))[0]}")

    _setup_threads()
    exp = args.experiment

    # Causal is dispatched before building the adapter (it spawns its own process).
    if exp == "causal":
        stage_causal(None, cfg, args, out_dir)
        return

    m = build_modality(cfg)
    preds = None
    if exp in ("probe", "all"):
        _, preds = stage_probe(m, cfg, out_dir)
    if exp in ("selective", "all"):
        stage_selective(m, cfg, out_dir, preds)
    if exp in ("cascade", "all"):
        stage_cascade(m, cfg, out_dir, preds)
    if exp in ("calibrate", "all"):
        stage_calibrate(m, cfg, out_dir)
    if exp == "all":
        print("\n[note] causal ablation runs separately: "
              "`run.py --modality <m> --experiment causal` (needs the live model).")


if __name__ == "__main__":
    main()
