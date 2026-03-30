"""Tests for the cognitive load scorer."""

import numpy as np
import pytest


def _make_roi_indices():
    """Create minimal ROI indices for testing."""
    return {
        # Executive
        "46": np.array([0, 1]),
        "FEF": np.array([2, 3]),
        "p32pr": np.array([4]),
        # Visual
        "V1": np.array([10, 11, 12]),
        "V2": np.array([13, 14]),
        "MT": np.array([15, 16]),
        # Auditory
        "A1": np.array([20, 21, 22]),
        "LBelt": np.array([23]),
        # Language
        "44": np.array([30, 31]),
        "45": np.array([32, 33]),
        "TPOJ1": np.array([34]),
    }


class TestCognitiveLoadScorer:
    def test_basic_scoring(self):
        from cortexlab.analysis.cognitive_load import CognitiveLoadScorer

        roi_indices = _make_roi_indices()
        scorer = CognitiveLoadScorer(roi_indices)

        # Create predictions with 50 vertices, 10 timepoints
        predictions = np.random.randn(10, 50)
        result = scorer.score_predictions(predictions)

        assert 0.0 <= result.overall_load <= 1.0
        assert 0.0 <= result.visual_complexity <= 1.0
        assert 0.0 <= result.auditory_demand <= 1.0
        assert 0.0 <= result.language_processing <= 1.0
        assert 0.0 <= result.executive_load <= 1.0

    def test_timeline_length(self):
        from cortexlab.analysis.cognitive_load import CognitiveLoadScorer

        roi_indices = _make_roi_indices()
        scorer = CognitiveLoadScorer(roi_indices)

        predictions = np.random.randn(15, 50)
        result = scorer.score_predictions(predictions, tr_seconds=1.5)

        assert len(result.timeline) == 15
        # Check timeline timestamps
        assert result.timeline[0][0] == 0.0
        assert abs(result.timeline[1][0] - 1.5) < 1e-6

    def test_single_timepoint(self):
        from cortexlab.analysis.cognitive_load import CognitiveLoadScorer

        roi_indices = _make_roi_indices()
        scorer = CognitiveLoadScorer(roi_indices)

        predictions = np.random.randn(50)  # 1D input
        result = scorer.score_predictions(predictions)

        assert len(result.timeline) == 1
        assert 0.0 <= result.overall_load <= 1.0

    def test_high_visual_activation(self):
        from cortexlab.analysis.cognitive_load import CognitiveLoadScorer

        roi_indices = _make_roi_indices()
        scorer = CognitiveLoadScorer(roi_indices, baseline_activation=0.1)

        # Create predictions with very high activation in visual ROIs
        predictions = np.zeros((5, 50))
        for roi in ["V1", "V2", "MT"]:
            predictions[:, roi_indices[roi]] = 10.0

        result = scorer.score_predictions(predictions)
        # Visual complexity should be highest (capped at 1.0 due to normalization)
        assert result.visual_complexity > 0.5

    def test_custom_cognitive_map(self):
        from cortexlab.analysis.cognitive_load import CognitiveLoadScorer

        roi_indices = {"V1": np.array([0, 1, 2]), "A1": np.array([3, 4])}
        custom_map = {
            "visual_complexity": ["V1"],
            "auditory_demand": ["A1"],
        }
        scorer = CognitiveLoadScorer(roi_indices, cognitive_map=custom_map)

        predictions = np.random.randn(5, 10)
        result = scorer.score_predictions(predictions)

        assert result.executive_load == 0.0  # Not in custom map
        assert result.language_processing == 0.0
