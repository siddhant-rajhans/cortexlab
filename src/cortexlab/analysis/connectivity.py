"""ROI-to-ROI functional connectivity analysis.

Computes correlation-based connectivity matrices from predicted brain
responses, clusters ROIs into functional networks, and derives graph
metrics (degree centrality, modularity).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConnectivityResult:
    """Results from ROI connectivity analysis."""

    correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    roi_names: list[str] = field(default_factory=list)
    clusters: dict[int, list[str]] = field(default_factory=dict)
    graph_metrics: dict[str, float | dict[str, float]] = field(default_factory=dict)


class ROIConnectivityAnalyzer:
    """Analyze functional connectivity between brain ROIs.

    Example
    -------
    >>> analyzer = ROIConnectivityAnalyzer(roi_indices)
    >>> result = analyzer.analyze(predictions, n_clusters=4)
    >>> print(result.correlation_matrix.shape)
    (11, 11)
    """

    def __init__(self, roi_indices: dict[str, np.ndarray]):
        self.roi_indices = roi_indices

    def compute_correlation_matrix(
        self, predictions: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Compute pairwise Pearson correlation between ROI timecourses.

        Parameters
        ----------
        predictions : np.ndarray
            Brain predictions of shape ``(n_timepoints, n_vertices)``.

        Returns
        -------
        corr_matrix : np.ndarray
            Correlation matrix of shape ``(n_rois, n_rois)``.
        roi_names : list[str]
            Ordered list of ROI names matching the matrix axes.
        """
        roi_names = list(self.roi_indices.keys())
        n_rois = len(roi_names)
        T = predictions.shape[0]

        timecourses = np.zeros((n_rois, T))
        for i, name in enumerate(roi_names):
            vertices = self.roi_indices[name]
            valid = vertices[vertices < predictions.shape[1]]
            if len(valid) > 0:
                timecourses[i] = predictions[:, valid].mean(axis=1)

        if T < 2:
            return np.eye(n_rois), roi_names

        corr_matrix = np.corrcoef(timecourses)
        # Handle NaN from constant timecourses
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        return corr_matrix, roi_names

    def cluster_networks(
        self,
        correlation_matrix: np.ndarray,
        roi_names: list[str],
        n_clusters: int = 5,
    ) -> dict[int, list[str]]:
        """Cluster ROIs into functional networks via agglomerative clustering.

        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix of shape ``(n_rois, n_rois)``.
        roi_names : list[str]
            ROI names matching matrix axes.
        n_clusters : int
            Number of clusters to form.

        Returns
        -------
        dict[int, list[str]]
            Mapping from cluster ID to list of ROI names.
        """
        from scipy.cluster.hierarchy import fcluster, linkage

        n = correlation_matrix.shape[0]
        n_clusters = min(n_clusters, n)

        # Distance = 1 - |correlation|
        dist = 1.0 - np.abs(correlation_matrix)
        np.fill_diagonal(dist, 0.0)

        # Convert to condensed distance matrix
        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(dist[i, j])
        condensed = np.array(condensed)

        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        clusters: dict[int, list[str]] = {}
        for roi_name, cluster_id in zip(roi_names, labels):
            cid = int(cluster_id)
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(roi_name)

        return clusters

    def graph_metrics(
        self,
        correlation_matrix: np.ndarray,
        roi_names: list[str],
        threshold: float = 0.3,
    ) -> dict[str, float | dict[str, float]]:
        """Compute graph metrics from thresholded connectivity.

        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix of shape ``(n_rois, n_rois)``.
        roi_names : list[str]
            ROI names matching matrix axes.
        threshold : float
            Minimum absolute correlation to form an edge.

        Returns
        -------
        dict
            Contains ``"degree_centrality"`` (per-ROI dict) and
            ``"mean_degree"`` (scalar).
        """
        n = correlation_matrix.shape[0]
        adj = (np.abs(correlation_matrix) > threshold).astype(float)
        np.fill_diagonal(adj, 0.0)

        degree = adj.sum(axis=1)
        max_degree = max(n - 1, 1)
        degree_centrality = {
            name: float(degree[i] / max_degree) for i, name in enumerate(roi_names)
        }
        mean_degree = float(degree.mean() / max_degree)

        return {
            "degree_centrality": degree_centrality,
            "mean_degree": mean_degree,
        }

    def analyze(
        self,
        predictions: np.ndarray,
        n_clusters: int = 5,
        threshold: float = 0.3,
    ) -> ConnectivityResult:
        """Run full connectivity analysis pipeline.

        Parameters
        ----------
        predictions : np.ndarray
            Brain predictions of shape ``(n_timepoints, n_vertices)``.
        n_clusters : int
            Number of functional networks to identify.
        threshold : float
            Correlation threshold for graph metrics.

        Returns
        -------
        ConnectivityResult
        """
        corr, names = self.compute_correlation_matrix(predictions)
        clusters = self.cluster_networks(corr, names, n_clusters)
        metrics = self.graph_metrics(corr, names, threshold)

        return ConnectivityResult(
            correlation_matrix=corr,
            roi_names=names,
            clusters=clusters,
            graph_metrics=metrics,
        )
