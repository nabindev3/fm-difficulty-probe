# Expansion-robustness sweep

Phase-3 (c): confirm the predictive null (SAE adds no power over raw activations)
is invariant to SAE width. Native expansions differ (LLM 4×, TSFM 8×; both
d_hidden = 4096), so each modality was retrained at the *other* expansion and
re-probed through the identical ladder.

- Train: `experiments/train_sae.py --expansion {4|8}` (train split only, 10 epochs).
- Re-probe: `experiments/run.py --sae_override <ckpt> --tag exp{N}`.

## Full ladder at each expansion

### LLM SQuAD (mid / L12), d_model 1024

| rung | 4× (native, d_hidden 4096) | 8× (sweep, d_hidden 8192) |
|---|---|---|
| P1 cheap         | 0.590 | 0.590 |
| P2 cheap+raw     | 0.671 | 0.671 |
| P3 cheap+SAE     | 0.592 | 0.613 |
| **Δ(SAE−raw)**   | **−0.079 [−0.118, −0.041]** | **−0.058 [−0.097, −0.018]** |
| Δ(SAE−cheap)     | +0.002 [−0.044, +0.047] | +0.023 [−0.021, +0.067] |

### TSFM ETTh1 (mid / encoder block 3), d_model 512

| rung | 8× (native, d_hidden 4096) | 4× (sweep, d_hidden 2048) |
|---|---|---|
| P1 cheap         | 0.654 | 0.654 |
| P2 cheap+raw     | 0.584 | 0.584 |
| P3 cheap+SAE     | 0.426 | 0.479 |
| **Δ(SAE−raw)**   | **−0.158 [−0.293, −0.025]** | **−0.100 [−0.199, −0.001]** |
| Δ(SAE−cheap)     | −0.228 [−0.366, −0.092] | −0.170 [−0.285, −0.050] |

## Verdict

All four Δ(SAE−raw) CIs are strictly below zero: the SAE decomposition adds no
predictive power over raw activations at either 4× or 8× in either modality. The
result is **not** an expansion artifact, and the larger LLM SAE (8×) does not
recover any signal the 4× SAE missed. Note the 8×→ larger width slightly *raises*
P3 toward the raw baseline but never past it (Δ stays significantly negative).
