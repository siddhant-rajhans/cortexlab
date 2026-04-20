"""Tests for noise-ceiling estimators.

The inter-subject ceiling should be ~1 on perfectly-correlated data and
~0 on independent noise. Split-half should recover the true reliability
of a known signal-plus-noise generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortexlab.analysis.noise_ceiling import (
    _pearson_columnwise,
    inter_subject_ceiling,
    normalize_by_ceiling,
    split_half_ceiling,
)


def test_pearson_columnwise_basic():
    a = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    b = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    r = _pearson_columnwise(a, b)
    assert np.allclose(r, [1.0, -1.0])


def test_pearson_columnwise_zero_variance_column():
    a = np.ones((4, 2))
    b = np.random.randn(4, 2)
    r = _pearson_columnwise(a, b)
    assert np.all(r == 0.0)


def test_inter_subject_ceiling_high_when_signal_is_shared():
    rng = np.random.default_rng(0)
    n_sub, n_stim, n_vox = 6, 200, 50
    signal = rng.standard_normal((n_stim, n_vox)).astype(np.float32)
    # Every subject sees the same signal + small private noise.
    responses = np.stack(
        [signal + 0.1 * rng.standard_normal(signal.shape).astype(np.float32)
         for _ in range(n_sub)],
        axis=0,
    )
    ceil = inter_subject_ceiling(responses)
    assert ceil.shape == (n_vox,)
    assert ceil.mean() > 0.8, f"expected high ceiling, got {ceil.mean():.3f}"


def test_inter_subject_ceiling_low_when_only_noise():
    rng = np.random.default_rng(1)
    responses = rng.standard_normal((5, 200, 30)).astype(np.float32)
    ceil = inter_subject_ceiling(responses)
    assert ceil.mean() < 0.1, f"expected near-zero ceiling, got {ceil.mean():.3f}"


def test_inter_subject_ceiling_rejects_bad_shapes():
    with pytest.raises(ValueError, match="n_subjects"):
        inter_subject_ceiling(np.zeros((10, 20)))
    with pytest.raises(ValueError, match=">=2 subjects"):
        inter_subject_ceiling(np.zeros((1, 20, 5)))


def test_split_half_ceiling_recovers_reliability():
    """With signal variance = 1 and per-trial noise variance = 1,
    the true reliability is 0.5 for single trials; averaging n_rep
    repetitions gives Spearman-Brown-corrected reliability approaching 1
    as n_rep grows. At n_rep=8 it should be well above 0.6.
    """
    rng = np.random.default_rng(42)
    n_stim, n_rep, n_vox = 100, 8, 20
    signal = rng.standard_normal((n_stim, n_vox)).astype(np.float32)
    trials = (signal[:, None, :]
              + rng.standard_normal((n_stim, n_rep, n_vox)).astype(np.float32))
    ceil = split_half_ceiling(trials, n_splits=40, seed=0)
    assert ceil.shape == (n_vox,)
    assert ceil.mean() > 0.6, f"expected >0.6 ceiling, got {ceil.mean():.3f}"
    # With zero signal (pure trial noise), ceiling should drop to ~0.
    noise_only = rng.standard_normal((n_stim, n_rep, n_vox)).astype(np.float32)
    ceil0 = split_half_ceiling(noise_only, n_splits=20, seed=1)
    assert ceil0.mean() < 0.15


def test_split_half_rejects_single_rep():
    with pytest.raises(ValueError, match=">=2 repetitions"):
        split_half_ceiling(np.zeros((10, 1, 5)))


def test_normalize_by_ceiling_respects_min_ceiling():
    r2 = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    ceil = np.array([0.8, 0.005, 0.4], dtype=np.float32)
    norm = normalize_by_ceiling(r2, ceil, min_ceiling=0.01)
    assert np.isclose(norm[0], 0.5)
    assert norm[1] == 0.0  # ceiling below threshold
    assert np.isclose(norm[2], 1.0)


def test_normalize_by_ceiling_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        normalize_by_ceiling(np.zeros(5), np.zeros(3))
