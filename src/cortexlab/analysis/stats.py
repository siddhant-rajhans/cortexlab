"""General-purpose statistical helpers used across the analysis modules.

Currently homes the Benjamini-Hochberg FDR correction. Kept module-scoped
so it stays reusable: any p-value array (per-voxel, per-ROI, per-channel)
can flow through the same helper rather than each caller re-implementing
the procedure.

Why a separate module?
----------------------

Brain-encoding pipelines compute p-values in many places: the lesion
permutation test (per voxel), bootstrap CIs (per ROI), inter-subject
reliability tests, etc. A single source of truth for multiple-comparison
correction prevents subtle sort-order bugs from spreading.
"""

from __future__ import annotations

import numpy as np


def bh_fdr(
    p_values: np.ndarray,
    *,
    alpha: float | None = None,
) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate correction.

    Returns BH q-values for an arbitrary 1-D array of p-values. The
    returned q-values are in the original input order; voxel ``i``'s
    q-value lives at position ``i`` of the output.

    Parameters
    ----------
    p_values
        1-D array of one-sided or two-sided p-values in ``[0, 1]``.
        NaNs are passed through unchanged so masked voxels do not bias
        the rank-based correction; they are excluded from the BH
        denominator (i.e. ``m = number of finite p_values``).
    alpha
        Optional reject-threshold. When provided, the returned array
        is the BH q-values, but a sibling boolean array of "reject H0"
        decisions can be reconstructed by callers as ``q < alpha``. The
        kwarg exists so callers can pass through their preferred alpha
        for API symmetry; we deliberately do not return a tuple to
        keep the calling convention identical to ``statsmodels``-style
        helpers.

    Returns
    -------
    np.ndarray
        Q-values, same shape and dtype family (``float64``) as the
        input. NaN entries in the input remain NaN in the output.

    Notes
    -----
    The classic Benjamini-Hochberg procedure (1995):

    1. Sort p-values ascending: ``p_(1) <= p_(2) <= ... <= p_(m)``.
    2. Define ``q_(k) = p_(k) * m / k``.
    3. Enforce monotonicity from the right:
       ``q_(k) = min(q_(k), q_(k+1), ..., q_(m))``.
    4. Clip to ``[0, 1]`` and unsort.

    For ``m`` very large (cortex voxels >100k) the implementation is
    O(m log m), dominated by the sort. We keep it numpy-only so the
    function works in environments without scipy.
    """
    if p_values.ndim != 1:
        raise ValueError(f"p_values must be 1-D, got shape {p_values.shape}")
    if alpha is not None and not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    p = np.asarray(p_values, dtype=np.float64).copy()
    out = np.full_like(p, np.nan)

    finite_mask = np.isfinite(p)
    if not finite_mask.any():
        return out

    p_finite = p[finite_mask]
    if (p_finite < 0).any() or (p_finite > 1).any():
        raise ValueError("p_values must lie in [0, 1] (after filtering NaN)")

    m = p_finite.size
    order = np.argsort(p_finite)
    sorted_p = p_finite[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    q_sorted = sorted_p * m / ranks

    # Right-to-left running minimum enforces monotonicity.
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    # Unsort: place each q at the original position.
    q_finite = np.empty_like(q_sorted)
    q_finite[order] = q_sorted

    out[finite_mask] = q_finite
    return out


def fraction_significant(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Convenience wrapper: fraction of finite p-values strictly below ``alpha``.

    NaN entries are excluded from both numerator and denominator.
    Returns 0.0 if there are no finite p-values.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    p = np.asarray(p_values, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.any():
        return 0.0
    return float((p[finite] < alpha).mean())


__all__ = ["bh_fdr", "fraction_significant"]
