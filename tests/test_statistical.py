"""Tests for statistical testing in brain alignment benchmark."""

import numpy as np
import pytest


class TestPermutationTest:
    def test_p_value_range(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat = np.random.randn(20, 32)
        brain_pred = np.random.randn(20, 50)
        bench = BrainAlignmentBenchmark(brain_pred)

        _, p = bench.permutation_test(model_feat, method="rsa", n_permutations=50, seed=42)
        assert 0.0 <= p <= 1.0

    def test_identical_features_low_p(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        data = np.random.randn(30, 50)
        bench = BrainAlignmentBenchmark(data)

        _, p = bench.permutation_test(data, method="cka", n_permutations=100, seed=42)
        assert p < 0.05

    def test_seed_reproducibility(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat = np.random.randn(15, 32)
        brain_pred = np.random.randn(15, 50)
        bench = BrainAlignmentBenchmark(brain_pred)

        _, p1 = bench.permutation_test(model_feat, method="rsa", n_permutations=50, seed=123)
        _, p2 = bench.permutation_test(model_feat, method="rsa", n_permutations=50, seed=123)
        assert p1 == p2


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        # Use identical features for high alignment so CI is tight around the point estimate
        np.random.seed(42)
        data = np.random.randn(30, 50)
        bench = BrainAlignmentBenchmark(data)

        score, lower, upper = bench.bootstrap_ci(
            data, method="cka", n_bootstrap=500, confidence=0.95, seed=42
        )
        assert lower <= score <= upper

    def test_ci_lower_less_than_upper(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat = np.random.randn(25, 32)
        brain_pred = np.random.randn(25, 50)
        bench = BrainAlignmentBenchmark(brain_pred)

        _, lower, upper = bench.bootstrap_ci(
            model_feat, method="cka", n_bootstrap=100, seed=42
        )
        assert lower <= upper

    def test_seed_reproducibility(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat = np.random.randn(15, 32)
        brain_pred = np.random.randn(15, 50)
        bench = BrainAlignmentBenchmark(brain_pred)

        r1 = bench.bootstrap_ci(model_feat, method="rsa", n_bootstrap=50, seed=99)
        r2 = bench.bootstrap_ci(model_feat, method="rsa", n_bootstrap=50, seed=99)
        assert r1 == r2
