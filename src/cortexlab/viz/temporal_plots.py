"""Visualization utilities for temporal dynamics analysis.

Provides functions to plot peak latencies, response curves (sustained vs
transient), and lag-correlation findings from TemporalDynamicsAnalyzer.
"""

from __future__ import annotations

import numpy as np


def plot_peak_latencies(ax, result, title: str = "Peak Latency per ROI") -> None:
    """Plot peak latencies as a bar chart.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    result : TemporalDynamicsResult
        Result from TemporalDynamicsAnalyzer.analyze().
    title : str
        Plot title.
    """
    roi_names = list(result.peak_latencies.keys())
    latencies = list(result.peak_latencies.values())
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
    colors = colors[: len(roi_names)]

    ax.bar(roi_names, latencies, color=colors)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Time (seconds)")
    ax.grid(axis="y", alpha=0.3)


def plot_response_curves(
    ax,
    result,
    roi_name: str,
    tr_seconds: float = 1.0,
    title: str | None = None,
) -> None:
    """Plot sustained vs transient response components for an ROI.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    result : TemporalDynamicsResult
        Result from TemporalDynamicsAnalyzer.analyze().
    roi_name : str
        Name of the ROI to plot.
    tr_seconds : float
        Repetition time in seconds for time-axis conversion.
    title : str, optional
        Plot title. Defaults to "Response Curves - {roi_name}".
    """
    if title is None:
        title = f"Response Curves - {roi_name}"

    n_timepoints = len(result.sustained_components[roi_name])
    time_axis = np.arange(n_timepoints) * tr_seconds
    sustained = result.sustained_components[roi_name]
    transient = result.transient_components[roi_name]

    ax.plot(time_axis, sustained, label="Sustained", color="#2196F3", linewidth=2)
    ax.plot(time_axis, transient, label="Transient", color="#FF5722", linewidth=2, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Activation")
    ax.legend()
    ax.grid(alpha=0.3)


def plot_lag_correlations(
    ax,
    result,
    tr_seconds: float = 1.0,
    title: str = "Lag-Correlation Plot",
) -> None:
    """Plot lag-shifted correlations for all ROIs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    result : TemporalDynamicsResult
        Result from TemporalDynamicsAnalyzer.analyze().
    tr_seconds : float
        Repetition time in seconds for lag-axis conversion.
    title : str
        Plot title.
    """
    roi_names = list(result.temporal_correlations.keys())
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
    colors = colors[: len(roi_names)]

    # Infer max_lag from first ROI's correlation array
    first_roi = roi_names[0]
    n_lags = len(result.temporal_correlations[first_roi])
    max_lag_trs = (n_lags - 1) // 2
    lags = np.arange(-max_lag_trs, max_lag_trs + 1) * tr_seconds

    for roi_name, color in zip(roi_names, colors):
        corr = result.temporal_correlations[roi_name]
        ax.plot(lags, corr, label=roi_name, color=color, linewidth=2)

    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Correlation")
    ax.legend()
    ax.grid(alpha=0.3)
