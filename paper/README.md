# Paper

Cross-modal manuscript (Phase 6). Structure follows the roadmap: shared Intro →
shared Method (described once) → Results-A (LLM) → Results-B (TSFM) → Synthesis →
Deployable artifact → Limitations, with per-modality detail deferred to the
appendix / the two component repos. The main body reads as one method applied
twice; a reader who skips both Results sections still gets the contribution from
the Intro + Table 1 (synthesis).

- `main.tex` — the manuscript (6 pp.).
- `references.bib` — bibliography.
- `outline.md` — the planning outline + the synthesis schema it was built from.

Every number in `main.tex` Table 1 comes from `../results/cross_modal_synthesis.md`
(regenerate with `experiments/synthesize.py`).

## Build

```bash
tectonic main.tex          # self-contained; downloads packages on first run
# or, with a TeX Live install:
latexmk -pdf main.tex
```

Produces `main.pdf`. Builds clean (no undefined references; one benign
underfull-hbox warning).

## Title

> SAE Features Are Causally Active but Predictively Redundant for Difficulty:
> Evidence Across Language and Time-Series Foundation Models

Alternate framing kept in `outline.md`:
*Do Foundation Models Know What They Don't Know? Label-Free Difficulty Probing
Across Modalities.*
