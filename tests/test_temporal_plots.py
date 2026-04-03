"""Tests for temporal dynamics visualization functions."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_result(n_timepoints=30, roi_names=None, include_correlations=True):
    """Create a TemporalDynamicsResult for testing."""
    from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer

    if roi_names is None:
        roi_names = ["V1", "MT", "A1"]

    roi_indices = {}
    offset = 0
    for name in roi_names:
        roi_indices[name] = np.arange(offset, offset + 10)
        offset += 10

    predictions = np.random.randn(n_timepoints, offset)
    analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.0)

    if include_correlations:
        features = np.random.randn(n_timepoints)
        return analyzer.analyze(predictions, features)
    else:
        return analyzer.analyze(predictions)


class TestPlotPeakLatencies:
    def test_basic(self):
        from cortexlab.viz.temporal_plots import plot_peak_latencies

        result = _make_result()
        fig, ax = plt.subplots()
        plot_peak_latencies(ax, result)
        assert ax.get_title() == "Peak Latency per ROI"
        assert ax.get_ylabel() == "Time (seconds)"
        plt.close()

    def test_custom_title(self):
        from cortexlab.viz.temporal_plots import plot_peak_latencies

        result = _make_result()
        fig, ax = plt.subplots()
        plot_peak_latencies(ax, result, title="Custom Title")
        assert ax.get_title() == "Custom Title"
        plt.close()

    def test_single_roi(self):
        from cortexlab.viz.temporal_plots import plot_peak_latencies

        result = _make_result(roi_names=["V1"])
        fig, ax = plt.subplots()
        plot_peak_latencies(ax, result)
        assert len(ax.patches) == 1  # one bar
        plt.close()

    def test_many_rois(self):
        from cortexlab.viz.temporal_plots import plot_peak_latencies

        result = _make_result(roi_names=["V1", "V2", "V3", "V4", "MT", "A1", "A4", "44"])
        fig, ax = plt.subplots()
        plot_peak_latencies(ax, result)
        assert len(ax.patches) == 8
        plt.close()


class TestPlotResponseCurves:
    def test_basic(self):
        from cortexlab.viz.temporal_plots import plot_response_curves

        result = _make_result()
        fig, ax = plt.subplots()
        plot_response_curves(ax, result, "V1")
        assert "V1" in ax.get_title()
        assert len(ax.lines) == 2  # sustained + transient
        plt.close()

    def test_custom_title(self):
        from cortexlab.viz.temporal_plots import plot_response_curves

        result = _make_result()
        fig, ax = plt.subplots()
        plot_response_curves(ax, result, "V1", title="My Title")
        assert ax.get_title() == "My Title"
        plt.close()

    def test_invalid_roi_raises(self):
        from cortexlab.viz.temporal_plots import plot_response_curves

        result = _make_result()
        fig, ax = plt.subplots()
        with pytest.raises(KeyError, match="not found"):
            plot_response_curves(ax, result, "NONEXISTENT")
        plt.close()

    def test_tr_seconds_scales_axis(self):
        from cortexlab.viz.temporal_plots import plot_response_curves

        result = _make_result(n_timepoints=20)
        fig, ax = plt.subplots()
        plot_response_curves(ax, result, "V1", tr_seconds=2.0)
        # Last x value should be (n_timepoints-1) * tr_seconds
        x_data = ax.lines[0].get_xdata()
        assert x_data[-1] == pytest.approx(19 * 2.0)
        plt.close()


class TestPlotLagCorrelations:
    def test_basic(self):
        from cortexlab.viz.temporal_plots import plot_lag_correlations

        result = _make_result()
        fig, ax = plt.subplots()
        plot_lag_correlations(ax, result)
        assert ax.get_title() == "Lag-Correlation Plot"
        # 3 ROIs = 3 lines + 2 reference lines (axvline + axhline)
        assert len(ax.lines) >= 3
        plt.close()

    def test_no_correlations_raises(self):
        from cortexlab.viz.temporal_plots import plot_lag_correlations

        result = _make_result(include_correlations=False)
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="No lag-correlation data"):
            plot_lag_correlations(ax, result)
        plt.close()

    def test_custom_title(self):
        from cortexlab.viz.temporal_plots import plot_lag_correlations

        result = _make_result()
        fig, ax = plt.subplots()
        plot_lag_correlations(ax, result, title="Custom Lag")
        assert ax.get_title() == "Custom Lag"
        plt.close()

    def test_tr_seconds_scales_lags(self):
        from cortexlab.viz.temporal_plots import plot_lag_correlations

        result = _make_result()
        fig, ax = plt.subplots()
        plot_lag_correlations(ax, result, tr_seconds=1.5)
        # Check that x-axis values are scaled by tr_seconds
        x_data = ax.lines[0].get_xdata()
        # Lags should be multiples of 1.5
        assert all(abs(x % 1.5) < 1e-10 or abs(x % 1.5 - 1.5) < 1e-10 for x in x_data)
        plt.close()


class TestIntegrationWithDemo:
    def test_demo_workflow(self):
        """Replicate the demo script workflow to catch regressions."""
        from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer
        from cortexlab.viz.temporal_plots import (
            plot_lag_correlations,
            plot_peak_latencies,
            plot_response_curves,
        )

        np.random.seed(42)
        n_timepoints, n_vertices = 50, 300
        bold = np.zeros(n_timepoints)
        bold[10] = 1.0
        hrf_t = np.arange(20)
        hrf = hrf_t * np.exp(-hrf_t / 1.5)
        hrf /= hrf.max()
        bold = np.convolve(bold, hrf)[:n_timepoints]

        roi_indices = {"V1": np.arange(0, 100), "MT": np.arange(100, 200)}
        predictions = np.random.randn(n_timepoints, n_vertices) * 0.3
        for v in roi_indices["V1"]:
            predictions[:, v] += bold
        for v in roi_indices["MT"]:
            predictions[:, v] += np.roll(bold, 2) * 0.7

        features = np.sin(np.linspace(0, 4 * np.pi, n_timepoints))
        analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.0)
        result = analyzer.analyze(predictions, features)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_peak_latencies(axes[0], result)
        plot_response_curves(axes[1], result, "V1", tr_seconds=1.0)
        plot_lag_correlations(axes[2], result, tr_seconds=1.0)
        plt.close()

        # V1 should peak before MT (V1 has unshifted bold, MT has +2 shift)
        assert result.peak_latencies["V1"] <= result.peak_latencies["MT"]
