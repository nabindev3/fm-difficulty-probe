# Submission materials

Ready-to-paste text for a workshop submission form and a PhD application. All
numbers trace to `results/cross_modal_synthesis.md`.

---

## 1. Short abstract (~150 words, for a submission form)

Foundation models ship without a native abstention signal, yet some inputs are
far harder than others. We ask whether a model's internal TopK sparse-autoencoder
(SAE) features encode a *self-difficulty* signal beyond a cheap, no-forward-pass
baseline — and we ask it of two unrelated modalities through one shared pipeline:
an autoregressive language model (Pythia on HellaSwag/SQuAD) and an encoder
time-series foundation model (Chronos-T5 on ETTh1). In both, SAE features add no
predictive power over the strongest cheap baseline (Δ(SAE−raw) ≤ 0; permutation
p < 10⁻⁴ on SQuAD), and the null is invariant to SAE width. A Platt-recalibrated
selective predictor on the cheap baseline is deployable in both (30–41% of oracle
AURC). But the causal picture dissociates: reconstruction-patching ablation finds
the language model's features causally active (5/5 features), while the
time-series model's are silent (0/5). The causal difficulty signal is a property
of the autoregressive LM, not of sparse features in general.

*(Word count ≈ 150. Trim the final sentence for a 120-word cap.)*

---

## 2. One-line / TL;DR

> Same SAE-difficulty pipeline run on a language model and a time-series model:
> the predictive null and the deployable cheap-baseline predictor replicate in
> both, but the causal signal exists only in the language model — localizing it
> as an LM property, not a universal SAE phenomenon.

---

## 3. PhD application paragraph (research-statement / cover letter)

I am drawn to interpretability research that is falsifiable and deployable rather
than merely suggestive. In a recent project I tested whether a foundation model's
sparse-autoencoder features encode a self-difficulty signal — useful for routing
and abstention — beyond what a cheap baseline already captures. Rather than report
one negative result and risk the "you probed it wrong" objection, I built a single
modality-agnostic pipeline and ran the *identical* experiment on two unrelated
foundation models: an autoregressive language model (Pythia) and an encoder
time-series model (Chronos-T5). I gated every new experiment behind a regression
check that reproduced both prior studies' headline numbers through the shared
code, then closed the methodological seams that a reviewer would attack — a
matched three-rung probe ladder, single- versus all-position causal patching, and
an SAE-width robustness sweep. The result is a clean cross-modal dissociation: the
predictive null and a Platt-recalibrated deployable selective predictor replicate
in both modalities, but the causal contribution is specific to the language model
(5/5 features causally active versus 0/5 on the time-series model). That
divergence — not a forced replication — is the contribution, and it localizes the
causal difficulty signal as a property of the autoregressive language model rather
than of sparse features in general. The work taught me to treat a negative result
as a hypothesis to stress-test across conditions, and to build research code
(tested shared core, one-command reproduction) so that a claim rests on the same
pipeline running in every setting by construction. I want to bring this
replication-first, deployment-aware approach to [GROUP]'s work on [TOPIC].

*(Swap [GROUP]/[TOPIC]. ~230 words; cut the last two sentences for a tighter
version.)*

---

## 4. Three bullet points (for a CV "selected projects" entry)

- Built a modality-agnostic interpretability pipeline and ran the identical
  SAE-difficulty experiment on a language model (Pythia) and a time-series
  foundation model (Chronos-T5), gated on reproducing both prior studies' headline
  numbers through shared code.
- Established a cross-modal dissociation: TopK-SAE features are predictively
  redundant for difficulty in both modalities (Δ(SAE−raw) ≤ 0, permutation
  p < 10⁻⁴ on SQuAD; robust to SAE width) yet causally active only in the language
  model (5/5 vs 0/5 features under reconstruction-patching ablation).
- Delivered a deployable artifact — a Platt-recalibrated selective predictor
  capturing 30–41% of oracle AURC — plus a one-command-reproducible repo, unit
  tests on synthetic arrays, and a 6-page workshop manuscript.

---

## 5. Double-blind anonymization checklist

If the target venue is double-blind, submit through an anonymized mirror (e.g.
https://anonymous.4open.science — point it at the GitHub repo and give it the
term list below; it rewrites the strings and hides the git history). Do NOT link
the real repo in the submitted PDF.

Identifying strings currently in tracked files (verified by grep, 2026-07-02):

| file | what to scrub |
|---|---|
| `CITATION.cff` | author name + email; `repository-code`/`url` |
| `LICENSE`, `paper/LICENSE` | copyright holder name |
| `README.md` | badge URLs + links to `github.com/nabindev3/*` legacy repos |
| `data/README.md` | legacy-repo clone URLs; HF dataset `nabindev3/llm-sae-difficulty-artifacts` |
| `paper/main.tex` | `\author{}` block (lines ~20–22); code/data URLs (~lines 295, 305–306) |

Term list for anonymous.4open.science: `Nabin Prasad Dev`, `nabin.dev33`,
`nabindev3`. Rebuild `main.pdf` from the anonymized `main.tex` (the committed
PDF embeds the author block) and check the PDF's metadata carries no author name.

The de-anonymized artifacts (this repo, the HF dataset, the Zenodo DOI) go in
the camera-ready only.
