"""Bit-reproducibility shim — import this BEFORE numpy/sklearn/torch.

Multi-threaded BLAS (OpenBLAS/MKL/Accelerate) sums in a non-deterministic
reduction order, which jitters bootstrap CI tails by ~1e-3 across runs/machines.
Pinning the math libraries to a single thread fixes the reduction order, so the
whole pipeline is bit-reproducible. The probe's cross-validation still
parallelizes across folds via joblib threads (that is a separate knob), so the
speed cost is small.

Values are set with setdefault, so a caller can still override
(e.g. `OMP_NUM_THREADS=8 python ...`) when they want speed over exact bits.
"""
import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",   # Apple Accelerate
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")
