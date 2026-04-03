from __future__ import annotations

import matplotlib.cm as cm
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
    colors = [cm.Set2(i / max(len(roi_names) - 1, 1)) for i in range(len(roi_names))]

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

    sustained_rois = getattr(result, "sustained_components", {})
    transient_rois = getattr(result, "transient_components", {})
    if roi_name not in sustained_rois or roi_name not in transient_rois:
        available_rois = sorted(
            set(sustained_rois.keys()) | set(transient_rois.keys())
        )
        raise KeyError(
            f"ROI '{roi_name}' not found in sustained/transient components. "
            f"Available ROIs: {available_rois}"
        )
    sustained = sustained_rois[roi_name]
    transient = transient_rois[roi_name]
    if len(sustained) != len(transient):
        raise ValueError(
            f"Sustained and transient components for ROI '{roi_name}' have "
            f"different lengths (sustained={len(sustained)}, transient={len(transient)})."
        )
    n_timepoints = len(sustained)
    time_axis = np.arange(n_timepoints) * tr_seconds

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
    if not roi_names:
        raise ValueError(
            "No lag-correlation data available in result.temporal_correlations; "
            "cannot plot lag correlations."
        )
    colors = [cm.Set2(i / max(len(roi_names) - 1, 1)) for i in range(len(roi_names))]

    first_roi = roi_names[0]
    first_corr = result.temporal_correlations[first_roi]
    n_lags = len(first_corr)
    if n_lags == 0:
        raise ValueError(
            f"Lag-correlation array for ROI '{first_roi}' is empty; cannot infer "
            "lag axis for plotting."
        )
    max_lag_trs = (n_lags - 1) // 2
    lags = np.arange(-max_lag_trs, max_lag_trs + 1) * tr_seconds

    for idx, roi_name in enumerate(roi_names):
        color = colors[idx % len(colors)]
        corr = result.temporal_correlations[roi_name]
        ax.plot(lags, corr, label=roi_name, color=color, linewidth=2)

    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Correlation")
    ax.legend()
    ax.grid(alpha=0.3)
    
