"""Tests for ROI connectivity analysis."""

import numpy as np
import pytest


class TestROIConnectivityAnalyzer:
    def test_correlation_matrix_shape(self, roi_indices):
        from cortexlab.analysis.connectivity import ROIConnectivityAnalyzer

        analyzer = ROIConnectivityAnalyzer(roi_indices)
        predictions = np.random.randn(30, 50)
        corr, names = analyzer.compute_correlation_matrix(predictions)

        n_rois = len(roi_indices)
        assert corr.shape == (n_rois, n_rois)
        assert len(names) == n_rois

    def test_correlation_matrix_symmetric(self, roi_indices):
        from cortexlab.analysis.connectivity import ROIConnectivityAnalyzer

        analyzer = ROIConnectivityAnalyzer(roi_indices)
        predictions = np.random.randn(30, 50)
        corr, _ = analyzer.compute_correlation_matrix(predictions)

        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_correlation_matrix_diagonal_ones(self, roi_indices):
        from cortexlab.analysis.connectivity import ROIConnectivityAnalyzer

        analyzer = ROIConnectivityAnalyzer(roi_indices)
        predictions = np.random.randn(30, 50)
        corr, _ = analyzer.compute_correlation_matrix(predictions)

        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_cluster_count(self, roi_indices):
        from cortexlab.analysis.connectivity import ROIConnectivityAnalyzer

        analyzer = ROIConnectivityAnalyzer(roi_indices)
        predictions = np.random.randn(30, 50)
        corr, names = analyzer.compute_correlation_matrix(predictions)
        clusters = analyzer.cluster_networks(corr, names, n_clusters=3)

        assert len(clusters) <= 3

    def test_all_rois_assigned(self, roi_indices):
        from cortexlab.analysis.connectivity import ROIConnectivityAnalyzer

        analyzer = ROIConnectivityAnalyzer(roi_indices)
        predictions = np.random.randn(30, 50)
        corr, names = analyzer.compute_correlation_matrix(predictions)
        clusters = analyzer.cluster_networks(corr, names, n_clusters=3)

        assigned = []
        for rois in clusters.values():
            assigned.extend(rois)
        assert set(assigned) == set(roi_indices.keys())

    def test_graph_metrics_degree_range(self, roi_indices):
        from cortexlab.analysis.connectivity import ROIConnectivityAnalyzer

        analyzer = ROIConnectivityAnalyzer(roi_indices)
        predictions = np.random.randn(30, 50)
        corr, names = analyzer.compute_correlation_matrix(predictions)
        metrics = analyzer.graph_metrics(corr, names, threshold=0.3)

        for deg in metrics["degree_centrality"].values():
            assert 0.0 <= deg <= 1.0
        assert 0.0 <= metrics["mean_degree"] <= 1.0

    def test_analyze_smoke(self, roi_indices):
        from cortexlab.analysis.connectivity import ROIConnectivityAnalyzer

        analyzer = ROIConnectivityAnalyzer(roi_indices)
        predictions = np.random.randn(30, 50)
        result = analyzer.analyze(predictions, n_clusters=3, threshold=0.3)

        assert result.correlation_matrix.shape[0] == len(roi_indices)
        assert len(result.roi_names) == len(roi_indices)
        assert len(result.clusters) > 0
        assert "degree_centrality" in result.graph_metrics

    def test_single_timepoint(self, roi_indices):
        from cortexlab.analysis.connectivity import ROIConnectivityAnalyzer

        analyzer = ROIConnectivityAnalyzer(roi_indices)
        predictions = np.random.randn(1, 50)
        corr, names = analyzer.compute_correlation_matrix(predictions)

        # With 1 timepoint, should return identity
        assert corr.shape == (len(roi_indices), len(roi_indices))
