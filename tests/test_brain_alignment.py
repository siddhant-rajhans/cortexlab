"""Tests for the brain-alignment benchmark."""

import numpy as np
import pytest


class TestBrainAlignmentBenchmark:
    def _make_data(self, n_stimuli=20, model_dim=64, n_vertices=100):
        model_features = np.random.randn(n_stimuli, model_dim)
        brain_predictions = np.random.randn(n_stimuli, n_vertices)
        return model_features, brain_predictions

    def test_rsa_returns_score(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat, brain_pred = self._make_data()
        bench = BrainAlignmentBenchmark(brain_pred)
        result = bench.score_model(model_feat, method="rsa")
        assert -1.0 <= result.aggregate_score <= 1.0
        assert result.method == "rsa"
        assert result.n_stimuli == 20

    def test_cka_returns_score(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat, brain_pred = self._make_data()
        bench = BrainAlignmentBenchmark(brain_pred)
        result = bench.score_model(model_feat, method="cka")
        assert isinstance(result.aggregate_score, float)
        assert result.method == "cka"

    def test_procrustes_returns_score(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat, brain_pred = self._make_data()
        bench = BrainAlignmentBenchmark(brain_pred)
        result = bench.score_model(model_feat, method="procrustes")
        assert isinstance(result.aggregate_score, float)

    def test_identical_features_high_score(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        data = np.random.randn(30, 50)
        bench = BrainAlignmentBenchmark(data)
        result = bench.score_model(data, method="cka")
        assert result.aggregate_score > 0.95

    def test_roi_scores(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat, brain_pred = self._make_data()
        roi_indices = {
            "V1": np.array([0, 1, 2, 3, 4]),
            "MT": np.array([10, 11, 12, 13, 14]),
        }
        bench = BrainAlignmentBenchmark(brain_pred, roi_indices=roi_indices)
        result = bench.score_model(model_feat, method="rsa")
        assert "V1" in result.roi_scores
        assert "MT" in result.roi_scores

    def test_roi_filter(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat, brain_pred = self._make_data()
        roi_indices = {
            "V1": np.array([0, 1, 2]),
            "MT": np.array([10, 11]),
            "A1": np.array([20, 21]),
        }
        bench = BrainAlignmentBenchmark(brain_pred, roi_indices=roi_indices)
        result = bench.score_model(model_feat, method="rsa", roi_filter=["V1", "A1"])
        assert "V1" in result.roi_scores
        assert "A1" in result.roi_scores
        assert "MT" not in result.roi_scores

    def test_stimulus_count_mismatch_raises(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat = np.random.randn(10, 32)
        brain_pred = np.random.randn(20, 100)
        bench = BrainAlignmentBenchmark(brain_pred)
        with pytest.raises(ValueError, match="Stimulus count mismatch"):
            bench.score_model(model_feat, method="rsa")

    def test_unknown_method_raises(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat, brain_pred = self._make_data()
        bench = BrainAlignmentBenchmark(brain_pred)
        with pytest.raises(ValueError, match="Unknown method"):
            bench.score_model(model_feat, method="banana")
