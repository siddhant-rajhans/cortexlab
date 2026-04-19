"""Noise-ceiling estimation for fMRI encoding.

Encoding-model R^2 is bounded above by the reliability of the brain
response itself; no model can predict random measurement noise. Two
standard estimators:

* Inter-subject reliability (this module's ``inter_subject_ceiling``):
  uses multiple subjects' responses to the same stimulus set and asks
  how well one subject's mean response predicts another's. This is the
  ceiling used by Lahner 2024 (BOLD Moments) and Allen 2022 (NSD).

* Split-half reliability (``split_half_ceiling``): uses repeated trials
  of the same stimulus within a single subject, splits them in half,
  and computes the Spearman-Brown-corrected correlation between halves.

Both produce per-voxel ceilings in R^2 space. Normalize model R^2 by
these to get a fair, interpretable score ("fraction of explainable
variance captured"), which is what goes in the slides.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def inter_subject_ceiling(
    responses: np.ndarray,
) -> np.ndarray:
    """Leave-one-subject-out inter-subject correlation, per voxel.

    Parameters
    ----------
    responses
        Array ``(n_subjects, n_stim, n_voxels)`` of responses to the
        same stimulus set across subjects.

    Returns
    -------
    ceiling
        Per-voxel ceiling in R^2, shape ``(n_voxels,)``. The ceiling is
        the mean leave-one-out Pearson correlation squared, which is the
        upper bound on encoding-model R^2 when predicting group-averaged
        responses.
    """
    if responses.ndim != 3:
        raise ValueError(
            f"responses must be (n_subjects, n_stim, n_voxels); got {responses.shape}"
        )
    n_sub, n_stim, n_vox = responses.shape
    if n_sub < 2:
        raise ValueError("need >=2 subjects for inter-subject ceiling")

    r_per_subject = np.empty((n_sub, n_vox), dtype=np.float32)
    for s in range(n_sub):
        left_out = responses[s]
        others_mean = np.mean(np.delete(responses, s, axis=0), axis=0)
        r_per_subject[s] = _pearson_columnwise(left_out, others_mean)
    mean_r = np.nanmean(r_per_subject, axis=0)
    # Convert correlation ceiling to R^2 space.
    return np.clip(mean_r, 0.0, 1.0) ** 2


def split_half_ceiling(
    trials: np.ndarray,
    n_splits: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Spearman-Brown-corrected split-half reliability.

    Parameters
    ----------
    trials
        Array ``(n_stim, n_reps, n_voxels)`` of per-trial responses
        to each stimulus, within a single subject.
    n_splits
        Number of random half-splits to average.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    ceiling
        Per-voxel ceiling in R^2.
    """
    if trials.ndim != 3:
        raise ValueError(f"trials must be (n_stim, n_reps, n_voxels); got {trials.shape}")
    n_stim, n_rep, n_vox = trials.shape
    if n_rep < 2:
        raise ValueError("need >=2 repetitions for split-half")

    rng = np.random.default_rng(seed)
    half = n_rep // 2
    r_acc = np.zeros(n_vox, dtype=np.float32)
    for _ in range(n_splits):
        perm = rng.permutation(n_rep)
        a_idx, b_idx = perm[:half], perm[half:2 * half]
        a = trials[:, a_idx].mean(axis=1)
        b = trials[:, b_idx].mean(axis=1)
        r = _pearson_columnwise(a, b)
        # Spearman-Brown correction from half-half -> full-full reliability.
        r_full = 2 * r / (1 + r)
        r_full = np.where(np.isfinite(r_full), r_full, 0.0)
        r_acc += r_full
    mean_r = r_acc / n_splits
    return np.clip(mean_r, 0.0, 1.0) ** 2


def normalize_by_ceiling(
    r2: np.ndarray,
    ceiling: np.ndarray,
    min_ceiling: float = 0.01,
) -> np.ndarray:
    """Divide model R^2 by the noise ceiling per voxel.

    Voxels with ceiling below ``min_ceiling`` are treated as noise-only
    and assigned 0.0 to avoid division instability.
    """
    if r2.shape != ceiling.shape:
        raise ValueError(f"shape mismatch: r2 {r2.shape} vs ceiling {ceiling.shape}")
    out = np.zeros_like(r2)
    mask = ceiling > min_ceiling
    out[mask] = r2[mask] / ceiling[mask]
    return out


def _pearson_columnwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation between two 2-D arrays with the
    same shape. Returns a 1-D array of length ``n_cols``.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    num = (a_c * b_c).sum(axis=0)
    den = np.sqrt((a_c ** 2).sum(axis=0) * (b_c ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return r.astype(np.float32)
