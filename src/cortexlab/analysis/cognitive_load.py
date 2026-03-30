"""Cognitive load scoring from predicted brain activation patterns.

Maps TRIBE v2's predicted fMRI responses onto cognitive dimensions
using established HCP MMP1.0 ROI groupings associated with different
cognitive functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# HCP MMP1.0 ROI groupings for cognitive dimensions.
# Each dimension maps to ROIs known to be involved in that function.
COGNITIVE_ROI_MAP: dict[str, list[str]] = {
    "executive_load": [
        # Dorsolateral prefrontal cortex
        "46",
        "9-46d",
        "p9-46v",
        "a9-46v",
        "9a",
        "8Av",
        "8Ad",
        "8BL",
        "8C",
        # Anterior cingulate cortex
        "p32pr",
        "a32pr",
        "d32",
        "p24",
        "a24",
        # Frontal eye fields
        "FEF",
        "PEF",
    ],
    "visual_complexity": [
        # Early visual
        "V1",
        "V2",
        "V3",
        "V4",
        # Ventral stream (object recognition)
        "FFC",
        "VVC",
        "VMV1",
        "VMV2",
        "VMV3",
        # Fusiform
        "PHA1",
        "PHA2",
        "PHA3",
        # Motion / dorsal
        "V3A",
        "V3B",
        "V6",
        "V6A",
        "V7",
        "MT",
        "MST",
        "FST",
        "V4t",
    ],
    "auditory_demand": [
        # Primary auditory
        "A1",
        "LBelt",
        "MBelt",
        "PBelt",
        "RI",
        # Auditory association
        "A4",
        "A5",
        "STSdp",
        "STSda",
        "STSvp",
        "STSva",
        "TA2",
    ],
    "language_processing": [
        # Broca's area (inferior frontal)
        "44",
        "45",
        "IFJa",
        "IFJp",
        "IFSp",
        "IFSa",
        # Wernicke's area (posterior temporal)
        "TPOJ1",
        "TPOJ2",
        "TPOJ3",
        "STV",
        "PSL",
        # Angular gyrus / semantic
        "PGi",
        "PGs",
        "PFm",
        # Temporal pole
        "TGd",
        "TGv",
        "TE1a",
        "TE1p",
        "TE2a",
        "TE2p",
    ],
}


@dataclass
class CognitiveLoadResult:
    """Result of cognitive load scoring."""

    overall_load: float
    visual_complexity: float
    auditory_demand: float
    language_processing: float
    executive_load: float
    timeline: list[tuple[float, dict[str, float]]] = field(default_factory=list)


class CognitiveLoadScorer:
    """Predict cognitive demand of media content from brain activation patterns.

    Uses predicted fMRI responses from TRIBE v2 and maps them onto
    cognitive dimensions via HCP MMP1.0 ROI groupings.

    Example
    -------
    >>> scorer = CognitiveLoadScorer(roi_indices)
    >>> result = scorer.score_predictions(predictions)
    >>> print(f"Overall load: {result.overall_load:.2f}")
    """

    def __init__(
        self,
        roi_indices: dict[str, np.ndarray],
        cognitive_map: dict[str, list[str]] | None = None,
        baseline_activation: float | None = None,
    ):
        """
        Parameters
        ----------
        roi_indices : dict[str, np.ndarray]
            HCP ROI name to vertex index mapping (from ``get_hcp_labels``).
        cognitive_map : dict[str, list[str]], optional
            Override the default cognitive ROI groupings.
        baseline_activation : float, optional
            Baseline activation level for normalisation. If None, uses
            the median activation across all vertices as baseline.
        """
        self.roi_indices = roi_indices
        self.cognitive_map = cognitive_map or COGNITIVE_ROI_MAP
        self.baseline = baseline_activation

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str = "facebook/tribev2",
        **kwargs,
    ) -> CognitiveLoadScorer:
        """Create a scorer with a loaded TRIBE v2 model.

        Lazily loads the model and ROI indices on first use.
        """
        instance = cls.__new__(cls)
        instance._checkpoint_dir = checkpoint_dir
        instance._model_kwargs = kwargs
        instance._model = None
        instance.cognitive_map = COGNITIVE_ROI_MAP
        instance.baseline = None
        instance.roi_indices = None
        return instance

    def _ensure_model(self):
        if self._model is None:
            from cortexlab.data.loader import get_hcp_labels
            from cortexlab.inference.predictor import TribeModel

            self.roi_indices = get_hcp_labels(mesh="fsaverage5")
            self._model = TribeModel.from_pretrained(
                self._checkpoint_dir, **self._model_kwargs
            )

    def _get_dimension_activation(
        self, vertex_data: np.ndarray, dimension: str
    ) -> float:
        """Compute mean activation for a cognitive dimension."""
        roi_names = self.cognitive_map.get(dimension, [])
        activations = []
        for roi in roi_names:
            vertices = self.roi_indices.get(roi)
            if vertices is None:
                continue
            valid = vertices[vertices < len(vertex_data)]
            if len(valid) > 0:
                activations.append(np.abs(vertex_data[valid]).mean())
        if not activations:
            return 0.0
        return float(np.mean(activations))

    def score_predictions(
        self,
        predictions: np.ndarray,
        tr_seconds: float = 1.0,
    ) -> CognitiveLoadResult:
        """Score predicted brain activations for cognitive load.

        Parameters
        ----------
        predictions : np.ndarray
            Predicted brain activations of shape ``(n_timepoints, n_vertices)``.
        tr_seconds : float
            Duration of each TR in seconds (for timeline).

        Returns
        -------
        CognitiveLoadResult
        """
        if predictions.ndim == 1:
            predictions = predictions[np.newaxis, :]

        baseline = self.baseline
        if baseline is None:
            baseline = float(np.median(np.abs(predictions)))
        baseline = max(baseline, 1e-8)

        dimensions = list(self.cognitive_map.keys())
        dim_scores = {d: [] for d in dimensions}
        timeline = []

        for t in range(predictions.shape[0]):
            vertex_data = predictions[t]
            t_scores = {}
            for dim in dimensions:
                raw = self._get_dimension_activation(vertex_data, dim)
                normalised = min(raw / baseline, 1.0) if baseline > 0 else 0.0
                dim_scores[dim].append(normalised)
                t_scores[dim] = normalised
            timeline.append((t * tr_seconds, t_scores))

        avg_scores = {d: float(np.mean(v)) if v else 0.0 for d, v in dim_scores.items()}
        overall = float(np.mean(list(avg_scores.values()))) if avg_scores else 0.0

        return CognitiveLoadResult(
            overall_load=overall,
            visual_complexity=avg_scores.get("visual_complexity", 0.0),
            auditory_demand=avg_scores.get("auditory_demand", 0.0),
            language_processing=avg_scores.get("language_processing", 0.0),
            executive_load=avg_scores.get("executive_load", 0.0),
            timeline=timeline,
        )

    def score(self, video_path: str = None, audio_path: str = None, text_path: str = None) -> CognitiveLoadResult:
        """End-to-end scoring from a media file.

        Loads the model, runs inference, and scores the predictions.
        Exactly one of the path arguments must be provided.
        """
        self._ensure_model()
        kwargs = {}
        if video_path:
            kwargs["video_path"] = video_path
        elif audio_path:
            kwargs["audio_path"] = audio_path
        elif text_path:
            kwargs["text_path"] = text_path
        else:
            raise ValueError("Provide one of video_path, audio_path, or text_path")

        events = self._model.get_events_dataframe(**kwargs)
        predictions, _ = self._model.predict(events, verbose=False)
        return self.score_predictions(predictions)
