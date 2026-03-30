"""Tests for temporal dynamics analysis."""

import numpy as np
import pytest


class TestTemporalDynamicsAnalyzer:
    def test_peak_latency_at_known_time(self, roi_indices):
        from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer

        analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.0)
        predictions = np.zeros((20, 50))
        # Spike at timepoint 10 in V1 vertices
        predictions[10, roi_indices["V1"]] = 5.0

        latency = analyzer.peak_latency(predictions, "V1")
        assert latency == 10.0

    def test_peak_latency_with_tr(self, roi_indices):
        from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer

        analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.5)
        predictions = np.zeros((20, 50))
        predictions[8, roi_indices["A1"]] = 3.0

        latency = analyzer.peak_latency(predictions, "A1")
        assert latency == 8 * 1.5

    def test_temporal_correlation_identical_signals(self, roi_indices):
        from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer

        analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.0)
        # Use abs(sin) since _get_roi_timecourse applies np.abs
        signal = np.abs(np.sin(np.linspace(0.1, 4 * np.pi, 30))) + 0.1
        predictions = np.zeros((30, 50))
        for v in roi_indices["V1"]:
            predictions[:, v] = signal

        corr = analyzer.temporal_correlation(predictions, signal, "V1", max_lag_trs=5)
        # At lag 0, correlation should be high
        assert corr[5] > 0.9  # center of array is lag 0

    def test_temporal_correlation_shape(self, roi_indices):
        from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer

        analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.0)
        predictions = np.random.randn(30, 50)
        features = np.random.randn(30)

        corr = analyzer.temporal_correlation(predictions, features, "V1", max_lag_trs=7)
        assert corr.shape == (15,)  # 2 * 7 + 1

    def test_temporal_correlation_2d_features(self, roi_indices):
        from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer

        analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.0)
        predictions = np.random.randn(30, 50)
        features = np.random.randn(30, 64)  # 2D features

        corr = analyzer.temporal_correlation(predictions, features, "V1", max_lag_trs=5)
        assert corr.shape == (11,)

    def test_decompose_sums_to_original(self, roi_indices):
        from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer

        analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.0)
        predictions = np.random.randn(30, 50)

        sustained, transient = analyzer.decompose_response(predictions, "V1", cutoff_seconds=4.0)
        original = analyzer._get_roi_timecourse(predictions, "V1")
        np.testing.assert_allclose(sustained + transient, original, atol=1e-10)

    def test_analyze_returns_all_fields(self, roi_indices):
        from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer

        analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.0)
        predictions = np.random.randn(30, 50)
        features = np.random.randn(30, 32)

        result = analyzer.analyze(predictions, features)
        assert len(result.peak_latencies) == len(roi_indices)
        assert len(result.temporal_correlations) == len(roi_indices)
        assert len(result.sustained_components) == len(roi_indices)
        assert len(result.transient_components) == len(roi_indices)

    def test_analyze_without_features(self, roi_indices):
        from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer

        analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.0)
        predictions = np.random.randn(20, 50)

        result = analyzer.analyze(predictions)
        assert len(result.peak_latencies) == len(roi_indices)
        assert len(result.temporal_correlations) == 0  # no features provided
        assert len(result.sustained_components) == len(roi_indices)
