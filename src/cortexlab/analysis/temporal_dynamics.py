"""Temporal dynamics analysis of predicted brain responses.

Analyzes how brain activation evolves over time, including peak
response latency, lag-shifted correlation with model features,
and decomposition into sustained vs. transient components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TemporalDynamicsResult:
    """Results from temporal dynamics analysis."""

    peak_latencies: dict[str, float] = field(default_factory=dict)
    temporal_correlations: dict[str, np.ndarray] = field(default_factory=dict)
    sustained_components: dict[str, np.ndarray] = field(default_factory=dict)
    transient_components: dict[str, np.ndarray] = field(default_factory=dict)


class TemporalDynamicsAnalyzer:
    """Analyze temporal properties of predicted brain responses.

    Example
    -------
    >>> analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.0)
    >>> result = analyzer.analyze(predictions, model_features)
    >>> print(f"V1 peak latency: {result.peak_latencies['V1']:.1f}s")
    """

    def __init__(
        self,
        roi_indices: dict[str, np.ndarray],
        tr_seconds: float = 1.0,
    ):
        self.roi_indices = roi_indices
        self.tr_seconds = tr_seconds

    def _get_roi_timecourse(
        self, predictions: np.ndarray, roi_name: str
    ) -> np.ndarray:
        """Extract mean timecourse for an ROI from vertex-level predictions."""
        vertices = self.roi_indices[roi_name]
        valid = vertices[vertices < predictions.shape[1]]
        if len(valid) == 0:
            return np.zeros(predictions.shape[0])
        return np.abs(predictions[:, valid]).mean(axis=1)

    def peak_latency(
        self, predictions: np.ndarray, roi_name: str
    ) -> float:
        """Compute time to peak activation for an ROI.

        Parameters
        ----------
        predictions : np.ndarray
            Brain predictions of shape ``(n_timepoints, n_vertices)``.
        roi_name : str
            Name of the ROI to analyze.

        Returns
        -------
        float
            Time in seconds to peak mean absolute activation.
        """
        timecourse = self._get_roi_timecourse(predictions, roi_name)
        if len(timecourse) == 0:
            return 0.0
        return float(np.argmax(timecourse) * self.tr_seconds)

    def temporal_correlation(
        self,
        predictions: np.ndarray,
        model_features: np.ndarray,
        roi_name: str,
        max_lag_trs: int = 10,
    ) -> np.ndarray:
        """Compute lag-shifted Pearson correlation between ROI timecourse and model features.

        Parameters
        ----------
        predictions : np.ndarray
            Brain predictions of shape ``(n_timepoints, n_vertices)``.
        model_features : np.ndarray
            Model feature timecourse of shape ``(n_timepoints,)`` or
            ``(n_timepoints, D)`` (mean across D is used).
        roi_name : str
            Name of the ROI to analyze.
        max_lag_trs : int
            Maximum lag in TRs for the correlation. Output has
            ``2 * max_lag_trs + 1`` entries.

        Returns
        -------
        np.ndarray
            Correlation values at lags ``[-max_lag_trs, ..., +max_lag_trs]``.
        """
        brain_tc = self._get_roi_timecourse(predictions, roi_name)
        if model_features.ndim > 1:
            model_tc = model_features.mean(axis=1)
        else:
            model_tc = model_features

        n = min(len(brain_tc), len(model_tc))
        brain_tc = brain_tc[:n]
        model_tc = model_tc[:n]

        correlations = []
        for lag in range(-max_lag_trs, max_lag_trs + 1):
            if lag >= 0:
                b = brain_tc[lag:]
                m = model_tc[: n - lag]
            else:
                b = brain_tc[: n + lag]
                m = model_tc[-lag:]
            if len(b) < 2:
                correlations.append(0.0)
                continue
            b_z = b - b.mean()
            m_z = m - m.mean()
            denom = np.sqrt((b_z**2).sum() * (m_z**2).sum())
            if denom < 1e-12:
                correlations.append(0.0)
            else:
                correlations.append(float((b_z * m_z).sum() / denom))

        return np.array(correlations)

    def decompose_response(
        self,
        predictions: np.ndarray,
        roi_name: str,
        cutoff_seconds: float = 4.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose ROI response into sustained and transient components.

        Uses a moving-average filter to separate low-frequency (sustained)
        from high-frequency (transient) response components.

        Parameters
        ----------
        predictions : np.ndarray
            Brain predictions of shape ``(n_timepoints, n_vertices)``.
        roi_name : str
            Name of the ROI to analyze.
        cutoff_seconds : float
            Cutoff period in seconds for the moving-average filter.

        Returns
        -------
        sustained : np.ndarray
            Low-frequency component of shape ``(n_timepoints,)``.
        transient : np.ndarray
            High-frequency component of shape ``(n_timepoints,)``.
        """
        timecourse = self._get_roi_timecourse(predictions, roi_name)
        window = max(1, int(cutoff_seconds / self.tr_seconds))

        # Moving average for sustained component
        kernel = np.ones(window) / window
        sustained = np.convolve(timecourse, kernel, mode="same")
        transient = timecourse - sustained

        return sustained, transient

    def analyze(
        self,
        predictions: np.ndarray,
        model_features: np.ndarray | None = None,
    ) -> TemporalDynamicsResult:
        """Run all temporal analyses across all ROIs.

        Parameters
        ----------
        predictions : np.ndarray
            Brain predictions of shape ``(n_timepoints, n_vertices)``.
        model_features : np.ndarray, optional
            Model feature timecourse for correlation analysis.

        Returns
        -------
        TemporalDynamicsResult
        """
        result = TemporalDynamicsResult()

        for roi_name in self.roi_indices:
            result.peak_latencies[roi_name] = self.peak_latency(predictions, roi_name)

            if model_features is not None:
                result.temporal_correlations[roi_name] = self.temporal_correlation(
                    predictions, model_features, roi_name
                )

            sustained, transient = self.decompose_response(predictions, roi_name)
            result.sustained_components[roi_name] = sustained
            result.transient_components[roi_name] = transient

        return result
