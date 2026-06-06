"""Build the cross-modal synthesis table — the paper's money figure.

Reads each experiment's unified result JSONs and emits one markdown table +
one combined JSON putting the predictive-null / causal-positive / deployable-
selective story side by side for every (modality, experiment).

    python experiments/synthesize.py \
        --runs llm_hellaswag llm_squad tsfm_etth1 --results_dir results

Handles BOTH causal-ablation schemas (the legacy LLM `feature_effects` /
`ci_lower`/`ci_upper`, and the new core.patching `per_feature_ablation_delta` /
`ci95`/`significant`) so LLM and TSFM rows line up.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import core._repro  # noqa: F401  — pins single-thread BLAS before numpy
from core._log import setup_logging, add_logging_args

log = logging.getLogger(__name__)


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _causal_nsig(d):
    """Return (n_significant, n_features, delta_recon) for either schema."""
    if d is None:
        return None
    if "feature_effects" in d:                       # legacy LLM schema
        fe = d["feature_effects"]
        nsig = sum(1 for v in fe.values() if v["ci_lower"] > 0 or v["ci_upper"] < 0)
        return nsig, len(fe), d.get("delta_recon_natural")
    if "per_feature_ablation_delta" in d:            # core.patching schema
        fe = d["per_feature_ablation_delta"]
        nsig = sum(1 for v in fe.values() if v["significant"])
        return nsig, len(fe), d.get("delta_sae_recon", {}).get("point")
    return None


def row_for(run: str, results_dir: str) -> dict:
    base = os.path.join(results_dir, run)
    probe = _load(os.path.join(base, "probe_results.json"))
    sel = _load(os.path.join(base, "selective_prediction.json"))
    casc = _load(os.path.join(base, "cascade_results.json"))
    c_all = _causal_nsig(_load(os.path.join(base, "causal_ablation_all.json")))
    # single-position result is named "last" (TSFM) or "boundary" (LLM).
    c_last = _causal_nsig(_load(os.path.join(base, "causal_ablation_last.json")) or
                          _load(os.path.join(base, "causal_ablation_boundary.json")))

    out = {"run": run}
    if probe:
        d_sr = probe["deltas"].get("P3_cheap_sae-P2_cheap_raw", {})
        d_sc = probe["deltas"].get("P3_cheap_sae-P1_cheap", {})
        out.update({
            "n_test": probe["n_test"],
            "hard_frac": round(probe["hard_fraction"], 3),
            "P1": round(probe["probes"]["P1_cheap"]["auroc"], 3),
            "P2_raw": round(probe["probes"]["P2_cheap_raw"]["auroc"], 3),
            "P3_sae": round(probe["probes"]["P3_cheap_sae"]["auroc"], 3),
            "delta_sae_over_raw": (round(d_sr.get("point"), 3),
                                   round(d_sr.get("ci_low"), 3),
                                   round(d_sr.get("ci_high"), 3)),
            "delta_sae_over_cheap": (round(d_sc.get("point"), 3),
                                     round(d_sc.get("ci_low"), 3),
                                     round(d_sc.get("ci_high"), 3)),
        })
    if sel:
        best = max(sel["probes"].items(), key=lambda kv: kv[1]["fraction_of_oracle_aurc"])
        out["selective_best"] = (best[0], round(100 * best[1]["fraction_of_oracle_aurc"], 1))
    if casc:
        out["cascade_dom_pts"] = {k: v["n_dominating_points"] for k, v in casc["probes"].items()}
    out["causal_all"] = c_all
    out["causal_last"] = c_last
    out["perm_p"] = _perm_p(base)
    return out


def _perm_p(base: str):
    """Label-permutation p-value for Δ(SAE − raw) = AUROC(P3) − AUROC(P2),
    recomputed from the persisted test scores (the roadmap's permutation axis)."""
    path = os.path.join(base, "probe_scores.parquet")
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd
        from core.stats import label_permutation_test
    except ImportError:
        return None  # optional row; never break the whole synthesis on a missing dep
    df = pd.read_parquet(path)
    need = {"y_test", "pred_P3_cheap_sae", "pred_P2_cheap_raw"}
    if not need.issubset(df.columns) or df["y_test"].nunique() < 2:
        return None
    res = label_permutation_test(df["y_test"].values,
                                 df["pred_P3_cheap_sae"].values,
                                 df["pred_P2_cheap_raw"].values, n_perm=10000)
    return {"observed_delta": res["observed_delta"],
            "p_one_sided": res["p_one_sided"], "p_two_sided": res["p_two_sided"]}


def render_markdown(rows: list[dict]) -> str:
    L = ["# Cross-modal synthesis\n",
         "**Universal predictive null + universal deployable predictor + "
         "LLM-specific causal signal.**\n",
         "| metric | " + " | ".join(r["run"] for r in rows) + " |",
         "|---|" + "---|" * len(rows)]

    def line(label, fn):
        return "| " + label + " | " + " | ".join(fn(r) for r in rows) + " |"

    def fmt_ci(t):
        return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]" if t else "—"

    def fmt_causal(c):
        return f"{c[0]}/{c[1]} sig" if c else "—"

    L.append(line("n test", lambda r: str(r.get("n_test", "—"))))
    L.append(line("hard frac", lambda r: str(r.get("hard_frac", "—"))))
    L.append(line("P1 cheap AUROC", lambda r: f"{r.get('P1','—')}"))
    L.append(line("P2 cheap+raw AUROC", lambda r: f"{r.get('P2_raw','—')}"))
    L.append(line("P3 cheap+SAE AUROC", lambda r: f"{r.get('P3_sae','—')}"))
    L.append(line("**Δ SAE over raw**", lambda r: fmt_ci(r.get("delta_sae_over_raw"))))
    L.append(line("Δ SAE over cheap", lambda r: fmt_ci(r.get("delta_sae_over_cheap"))))

    def fmt_perm(p):
        if not p:
            return "—"
        v = p["p_one_sided"]
        return ("<1e-4" if v == 0 else f"{v:.4g}") + " (1-sided)"
    L.append(line("label-perm p (Δ SAE−raw)", lambda r: fmt_perm(r.get("perm_p"))))
    L.append(line("causal: all-position", lambda r: fmt_causal(r.get("causal_all"))))
    L.append(line("causal: single-position", lambda r: fmt_causal(r.get("causal_last"))))
    L.append(line("selective: % oracle",
                  lambda r: f"{r['selective_best'][0]} {r['selective_best'][1]}%"
                  if r.get("selective_best") else "—"))

    def _short(rung):  # P3_cheap_sae -> P3-SAE, P1_cheap -> P1
        return rung.replace("_cheap_sae", "-SAE").replace("_cheap_raw", "-raw").replace("_cheap", "")

    def fmt_casc(d):
        if not d:
            return "—"
        return ", ".join(f"{v} ({_short(k)})" for k, v in d.items())
    L.append(line("cascade: Pareto-dom pts", lambda r: fmt_casc(r.get("cascade_dom_pts"))))
    L.append(
        "\n**Reading.**\n"
        "1. *Predictive null replicates in BOTH modalities* — SAE adds no power "
        "over the strongest cheap rung (Δ rows ≤ 0 or CI straddles 0).\n"
        "2. *Causal positive is modality-specific* — the LLM's top features are "
        "causally active under all-position patching (5/5) and under-detected by "
        "single-position (0–2/5: coverage, not fidelity). On the TSFM, NO feature "
        "is significant under either coverage (0/5), reproducing the legacy "
        "Chronos null (50-sample run, 0/5). So the coverage-not-fidelity story is "
        "an LLM finding; on Chronos the features are predictively redundant AND "
        "causally quiet at this scale.\n"
        "3. *Deployable artifact replicates in BOTH* — a cheap-baseline selective "
        "predictor captures 30–41% of oracle AURC.\n\n"
        "The cross-modal dissociation (predictive-null both; causal-positive LLM-"
        "only) is itself the contribution: it isolates the causal signal as a "
        "property of the autoregressive LM, not a universal SAE phenomenon.\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["llm_hellaswag", "llm_squad", "tsfm_etth1"])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/cross_modal_synthesis.md")
    add_logging_args(ap)
    args = ap.parse_args()
    setup_logging(args.verbose, args.quiet)

    rows = [row_for(r, args.results_dir) for r in args.runs]
    md = render_markdown(rows)
    with open(args.out, "w") as f:
        f.write(md)
    with open(args.out.replace(".md", ".json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(md)                       # the synthesis table is the report -> stdout
    log.info("saved %s", args.out)


if __name__ == "__main__":
    main()
