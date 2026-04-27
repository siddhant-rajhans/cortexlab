"""Tests for :mod:`cortexlab.analysis.stats`.

The BH-FDR implementation has to behave correctly on three classes of
input: clean, edge cases (empty, all-NaN, all-significant), and
adversarial (already-monotone, reverse-order, ties). Those branches all
have known closed-form expectations, so tests here are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortexlab.analysis.stats import bh_fdr, fraction_significant

# --------------------------------------------------------------------------- #
# bh_fdr — closed-form correctness                                            #
# --------------------------------------------------------------------------- #

def test_bh_fdr_classic_example():
    """Reproduce a small worked example.

    For p = [0.01, 0.04, 0.03, 0.005, 0.20], m=5:
    sorted ascending: [0.005, 0.01, 0.03, 0.04, 0.20]
    raw q = sorted_p * m / rank:
        0.005 * 5/1 = 0.025
        0.01  * 5/2 = 0.025
        0.03  * 5/3 = 0.05
        0.04  * 5/4 = 0.05
        0.20  * 5/5 = 0.20
    Right-running min keeps these monotone (already are).
    """
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.20])
    q = bh_fdr(p)
    # Position-by-position:
    np.testing.assert_allclose(q, [0.025, 0.05, 0.05, 0.025, 0.20], atol=1e-12)


def test_bh_fdr_enforces_monotonicity():
    """BH must enforce that q-values are monotone in p-rank.

    For p = [0.01, 0.02, 0.03, 0.99, 0.99] with m=5:
    raw q = [0.05, 0.05, 0.05, 1.2375, 0.99]
    monotone-from-right: q3=0.99, q4=0.99 (clipped 1.2375 to 1.0 then min with 0.99=0.99),
    q2=0.05, q1=0.05, q0=0.05
    """
    p = np.array([0.01, 0.02, 0.03, 0.99, 0.99])
    q = bh_fdr(p)
    # Monotone non-decreasing in original p order:
    sorted_indices = np.argsort(p)
    sorted_q = q[sorted_indices]
    assert np.all(sorted_q[:-1] <= sorted_q[1:] + 1e-12)
    assert q[0] == pytest.approx(0.05)
    assert q[-1] <= 1.0


def test_bh_fdr_clips_to_unit_interval():
    p = np.array([0.5, 0.6, 0.7, 0.99])
    q = bh_fdr(p)
    assert (q >= 0).all()
    assert (q <= 1.0 + 1e-12).all()


def test_bh_fdr_handles_ties():
    """Tied p-values must produce tied q-values (in input order).

    For p = [0.05, 0.05, 0.05] with m=3:
    sorted = [0.05, 0.05, 0.05]
    raw q = [0.15, 0.075, 0.05]
    monotone-from-right: [0.05, 0.05, 0.05]
    """
    p = np.array([0.05, 0.05, 0.05])
    q = bh_fdr(p)
    assert q[0] == q[1] == q[2]
    assert q[0] == pytest.approx(0.05, abs=1e-12)


def test_bh_fdr_returns_input_order():
    """Ensure unsort step puts q-values back at their original index."""
    p = np.array([0.5, 0.001, 0.4, 0.002])
    q = bh_fdr(p)
    # Smallest p is at index 1, second smallest at index 3 — those should
    # have the smallest q-values.
    smallest_p_idx = np.argmin(p)
    smallest_q_idx = np.argmin(q)
    assert smallest_p_idx == smallest_q_idx


# --------------------------------------------------------------------------- #
# bh_fdr — edge cases                                                         #
# --------------------------------------------------------------------------- #

def test_bh_fdr_passes_through_nan():
    p = np.array([0.01, np.nan, 0.05, 0.5, np.nan])
    q = bh_fdr(p)
    assert np.isnan(q[1])
    assert np.isnan(q[4])
    assert np.isfinite(q[0])
    assert np.isfinite(q[2])
    # NaNs should be excluded from m: with 3 finite values, q at sorted rank 1 is p*3/1.
    # Here p=[0.01, 0.05, 0.5], m=3: q = [0.03, 0.075, 0.5]. (Already monotone.)
    np.testing.assert_allclose(q[[0, 2, 3]], [0.03, 0.075, 0.5], atol=1e-12)


def test_bh_fdr_all_nan_returns_all_nan():
    p = np.full(10, np.nan)
    q = bh_fdr(p)
    assert np.isnan(q).all()


def test_bh_fdr_single_value():
    p = np.array([0.03])
    q = bh_fdr(p)
    np.testing.assert_allclose(q, [0.03])


def test_bh_fdr_uniform_null_is_conservative():
    """Under a pure uniform null with m large, BH should NOT discover
    anything: the smallest q-value should be near 1 since p_(1) * m is
    expected to be O(1). This is the desired calibration property."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, size=10_000)
    q = bh_fdr(p)
    # Under pure null with m=10k, even the smallest q should be >= ~0.5
    # because p_(1) * m ≈ 1 in expectation.
    assert q.min() > 0.3
    # No q can exceed 1.
    assert q.max() <= 1.0 + 1e-12
    # And no q at alpha=0.05 should pass (false discovery rate IS controlled).
    assert (q < 0.05).sum() == 0


# --------------------------------------------------------------------------- #
# bh_fdr — input validation                                                   #
# --------------------------------------------------------------------------- #

def test_bh_fdr_rejects_2d():
    with pytest.raises(ValueError, match="1-D"):
        bh_fdr(np.zeros((3, 3)))


def test_bh_fdr_rejects_p_below_zero():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bh_fdr(np.array([-0.01, 0.5]))


def test_bh_fdr_rejects_p_above_one():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bh_fdr(np.array([0.5, 1.01]))


def test_bh_fdr_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        bh_fdr(np.array([0.1, 0.2]), alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        bh_fdr(np.array([0.1, 0.2]), alpha=1.0)


# --------------------------------------------------------------------------- #
# fraction_significant                                                        #
# --------------------------------------------------------------------------- #

def test_fraction_significant_basic():
    p = np.array([0.01, 0.04, 0.05, 0.5, 0.99])
    # Strictly < 0.05: 0.01 and 0.04 -> 2/5 = 0.4
    assert fraction_significant(p, alpha=0.05) == pytest.approx(0.4)


def test_fraction_significant_excludes_nan():
    p = np.array([0.01, np.nan, 0.5])
    # 1 of 2 finite values < 0.05.
    assert fraction_significant(p, alpha=0.05) == pytest.approx(0.5)


def test_fraction_significant_all_nan_returns_zero():
    p = np.full(5, np.nan)
    assert fraction_significant(p) == 0.0


def test_fraction_significant_rejects_bad_alpha():
    with pytest.raises(ValueError):
        fraction_significant(np.array([0.1]), alpha=0.0)
    with pytest.raises(ValueError):
        fraction_significant(np.array([0.1]), alpha=1.0)


# --------------------------------------------------------------------------- #
# integration with realistic per-voxel p-value distribution                   #
# --------------------------------------------------------------------------- #

def test_bh_fdr_brain_imaging_scale():
    """Sanity: works on a 327k-voxel array with a mix of true signal
    and uniform null. All it needs to do is run, return monotone
    q-values per rank, and clip to [0, 1]."""
    rng = np.random.default_rng(0)
    n_signal = 30_000
    n_null = 297_684
    p = np.concatenate([
        rng.beta(0.5, 5, size=n_signal),     # signal-rich (small p)
        rng.uniform(0, 1, size=n_null),      # uniform null
    ])
    rng.shuffle(p)
    q = bh_fdr(p)
    assert q.shape == p.shape
    # Sorted q-values are non-decreasing along sorted-p order.
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-9)
    # Some signal voxels should survive at q < 0.05.
    assert (q < 0.05).sum() > 0
