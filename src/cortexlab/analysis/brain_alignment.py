"""Brain-alignment benchmark for comparing AI model representations.

Score how "brain-like" any AI model's internal representations are by
comparing them against TRIBE v2's predicted brain responses using
Representational Similarity Analysis (RSA) or Centered Kernel
Alignment (CKA).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Results from a brain-alignment benchmark run."""

    method: str
    aggregate_score: float
    roi_scores: dict[str, float] = field(default_factory=dict)
    n_stimuli: int = 0


def _compute_rdm(features: np.ndarray) -> np.ndarray:
    """Compute a representational dissimilarity matrix (1 - cosine sim)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalised = features / norms
    sim = normalised @ normalised.T
    return 1.0 - sim


def _rsa_score(model_features: np.ndarray, brain_features: np.ndarray) -> float:
    """Representational Similarity Analysis via Spearman correlation of RDMs."""
    from scipy.stats import spearmanr

    rdm_model = _compute_rdm(model_features)
    rdm_brain = _compute_rdm(brain_features)
    # Extract upper triangle (excluding diagonal)
    idx = np.triu_indices(rdm_model.shape[0], k=1)
    corr, _ = spearmanr(rdm_model[idx], rdm_brain[idx])
    return float(corr) if not np.isnan(corr) else 0.0


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear Centered Kernel Alignment between two feature matrices.

    CKA uses Gram matrices (n x n) so it naturally handles different
    feature dimensions without truncation.
    """
    n = X.shape[0]
    # Centre
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    # Gram matrices (n x n, dimension-independent)
    XX = X @ X.T  # (n, n)
    YY = Y @ Y.T  # (n, n)
    # HSIC via Gram matrices - works regardless of feature dimensions
    hsic_xy = np.trace(XX @ YY) / (n - 1) ** 2
    hsic_xx = np.trace(XX @ XX) / (n - 1) ** 2
    hsic_yy = np.trace(YY @ YY) / (n - 1) ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def _procrustes_score(X: np.ndarray, Y: np.ndarray) -> float:
    """Procrustes analysis: rotation-invariant shape comparison.

    Works with different feature dimensions by truncating to the
    smaller dimension.
    """
    # Match dimensions by truncating to min(d_x, d_y)
    min_dim = min(X.shape[1], Y.shape[1])
    X = X[:, :min_dim]
    Y = Y[:, :min_dim]
    # Centre and scale
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    norm_x = np.linalg.norm(X)
    norm_y = np.linalg.norm(Y)
    if norm_x < 1e-12 or norm_y < 1e-12:
        return 0.0
    X = X / norm_x
    Y = Y / norm_y
    # Optimal rotation via SVD
    M = Y.T @ X
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    rotated = Y @ R
    # Score = 1 - normalized Procrustes distance
    dist = np.linalg.norm(X - rotated)
    return float(max(0.0, 1.0 - dist))


_METHODS = {
    "rsa": _rsa_score,
    "cka": _linear_cka,
    "procrustes": _procrustes_score,
}


class BrainAlignmentBenchmark:
    """Benchmark AI model representations against predicted brain responses.

    Example
    -------
    >>> bench = BrainAlignmentBenchmark(brain_predictions)
    >>> result = bench.score_model(clip_features, method="rsa")
    >>> print(result.aggregate_score)
    0.42
    """

    def __init__(
        self,
        brain_predictions: np.ndarray,
        roi_indices: dict[str, np.ndarray] | None = None,
    ):
        """
        Parameters
        ----------
        brain_predictions : np.ndarray
            Array of shape ``(n_stimuli, n_vertices)`` with predicted
            fMRI responses from TRIBE v2 for a set of stimuli.
        roi_indices : dict[str, np.ndarray], optional
            HCP ROI name to vertex index mapping for per-ROI scoring.
        """
        self.brain_predictions = brain_predictions
        self.roi_indices = roi_indices

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str = "facebook/tribev2",
        roi_indices: dict[str, np.ndarray] | None = None,
        **kwargs,
    ) -> BrainAlignmentBenchmark:
        """Create a benchmark instance with a loaded TRIBE v2 model.

        The model is stored so you can call :meth:`score_model_with_stimuli`
        to generate brain predictions on the fly.
        """
        instance = cls(brain_predictions=np.array([]), roi_indices=roi_indices)
        instance._checkpoint_dir = checkpoint_dir
        instance._model_kwargs = kwargs
        instance._model = None
        return instance

    def _ensure_model(self):
        if self._model is None:
            from cortexlab.inference.predictor import TribeModel

            self._model = TribeModel.from_pretrained(
                self._checkpoint_dir, **self._model_kwargs
            )

    def score_model(
        self,
        model_features: np.ndarray,
        method: str = "rsa",
        roi_filter: list[str] | None = None,
        brain_predictions: np.ndarray | None = None,
    ) -> AlignmentResult:
        """Score how brain-aligned a set of model features are.

        Parameters
        ----------
        model_features : np.ndarray
            Feature matrix of shape ``(n_stimuli, D)`` extracted from
            any AI model for the same stimuli used to generate the
            brain predictions.
        method : str
            Comparison method: ``"rsa"``, ``"cka"``, or ``"procrustes"``.
        roi_filter : list[str], optional
            If set, only compute alignment for these ROIs.
        brain_predictions : np.ndarray, optional
            Override the stored brain predictions.

        Returns
        -------
        AlignmentResult
        """
        if method not in _METHODS:
            raise ValueError(f"Unknown method {method!r}. Choose from {list(_METHODS)}")

        brain = brain_predictions if brain_predictions is not None else self.brain_predictions
        score_fn = _METHODS[method]

        if model_features.shape[0] != brain.shape[0]:
            raise ValueError(
                f"Stimulus count mismatch: model has {model_features.shape[0]}, "
                f"brain has {brain.shape[0]}"
            )

        # Aggregate score (full vertex space)
        aggregate = score_fn(model_features, brain)

        # Per-ROI scores
        roi_scores = {}
        if self.roi_indices is not None:
            rois = self.roi_indices
            if roi_filter:
                rois = {k: v for k, v in rois.items() if k in roi_filter}
            for name, vertices in rois.items():
                valid = vertices[vertices < brain.shape[1]]
                if len(valid) < 2:
                    continue
                roi_brain = brain[:, valid]
                roi_scores[name] = score_fn(model_features, roi_brain)

        return AlignmentResult(
            method=method,
            aggregate_score=aggregate,
            roi_scores=roi_scores,
            n_stimuli=model_features.shape[0],
        )

    def permutation_test(
        self,
        model_features: np.ndarray,
        method: str = "rsa",
        n_permutations: int = 1000,
        seed: int | None = None,
        brain_predictions: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Compute significance of alignment via permutation test.

        Shuffles the stimulus order of model_features and recomputes the
        alignment score. The p-value is the fraction of permuted scores
        that meet or exceed the observed score.

        Parameters
        ----------
        model_features : np.ndarray
            Feature matrix of shape ``(n_stimuli, D)``.
        method : str
            Comparison method (``"rsa"``, ``"cka"``, ``"procrustes"``).
        n_permutations : int
            Number of random permutations.
        seed : int, optional
            Random seed for reproducibility.
        brain_predictions : np.ndarray, optional
            Override stored brain predictions.

        Returns
        -------
        observed_score : float
        p_value : float
        """
        rng = np.random.default_rng(seed)
        result = self.score_model(model_features, method=method, brain_predictions=brain_predictions)
        observed = result.aggregate_score

        count = 0
        for _ in range(n_permutations):
            perm_idx = rng.permutation(model_features.shape[0])
            perm_result = self.score_model(
                model_features[perm_idx], method=method, brain_predictions=brain_predictions
            )
            if perm_result.aggregate_score >= observed:
                count += 1

        p_value = (count + 1) / (n_permutations + 1)
        return observed, p_value

    def bootstrap_ci(
        self,
        model_features: np.ndarray,
        method: str = "rsa",
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int | None = None,
        brain_predictions: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """Compute bootstrap confidence interval for alignment score.

        Resamples stimuli with replacement and computes the alignment
        score for each bootstrap sample to estimate the CI.

        Parameters
        ----------
        model_features : np.ndarray
            Feature matrix of shape ``(n_stimuli, D)``.
        method : str
            Comparison method.
        n_bootstrap : int
            Number of bootstrap samples.
        confidence : float
            Confidence level (e.g. 0.95 for 95% CI).
        seed : int, optional
            Random seed for reproducibility.
        brain_predictions : np.ndarray, optional
            Override stored brain predictions.

        Returns
        -------
        point_estimate : float
        ci_lower : float
        ci_upper : float
        """
        rng = np.random.default_rng(seed)
        brain = brain_predictions if brain_predictions is not None else self.brain_predictions
        score_fn = _METHODS[method]
        n = model_features.shape[0]

        result = self.score_model(model_features, method=method, brain_predictions=brain_predictions)
        point_estimate = result.aggregate_score

        scores = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            score = score_fn(model_features[idx], brain[idx])
            scores.append(score)

        scores = np.array(scores)
        alpha = 1 - confidence
        ci_lower = float(np.percentile(scores, 100 * alpha / 2))
        ci_upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        return point_estimate, ci_lower, ci_upper
