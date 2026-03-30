"""Modality importance scoring via ablation.

:class:`ModalityAttributor` measures how much text, audio, and video
each contribute to the predicted brain response at every vertex by
comparing the full prediction against predictions with each modality
zeroed out.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from neuralset.dataloader import SegmentData

logger = logging.getLogger(__name__)


class ModalityAttributor:
    """Score per-vertex importance of each input modality.

    Uses an ablation approach: for each modality, zero out its features
    and measure the change in predicted brain activation.  Larger changes
    mean the modality matters more for that vertex.

    Parameters
    ----------
    model : torch.nn.Module
        A :class:`FmriEncoderModel` instance.
    roi_indices : dict[str, np.ndarray], optional
        If provided, also compute per-ROI summary scores.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        roi_indices: dict[str, np.ndarray] | None = None,
    ):
        self.model = model
        self.roi_indices = roi_indices

    def attribute(self, batch: SegmentData) -> dict[str, np.ndarray]:
        """Compute modality importance scores for a single batch.

        Parameters
        ----------
        batch : SegmentData
            Input batch containing features for all modalities.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are modality names (e.g. ``"text"``, ``"audio"``,
            ``"video"``) mapped to importance arrays of shape
            ``(n_vertices,)``.  If *roi_indices* was provided, additional
            keys like ``"text_roi"`` map ROI names to scalar scores.
        """
        self.model.eval()
        modalities = [m for m in self.model.feature_dims if m in batch.data]

        with torch.inference_mode():
            baseline = self.model(batch).detach()  # (B, V, T)
            baseline_mean = baseline.mean(dim=(0, 2)).cpu().numpy()  # (V,)

        scores: dict[str, np.ndarray] = {}

        for mod in modalities:
            ablated_data = {k: v.clone() for k, v in batch.data.items()}
            ablated_data[mod] = torch.zeros_like(ablated_data[mod])
            ablated_batch = SegmentData(data=ablated_data, segments=batch.segments)

            with torch.inference_mode():
                ablated_pred = self.model(ablated_batch).detach()
                ablated_mean = ablated_pred.mean(dim=(0, 2)).cpu().numpy()

            importance = np.abs(baseline_mean - ablated_mean)
            scores[mod] = importance

            if self.roi_indices is not None:
                roi_scores = {}
                for roi_name, vertices in self.roi_indices.items():
                    valid = vertices[vertices < len(importance)]
                    roi_scores[roi_name] = float(importance[valid].mean()) if len(valid) > 0 else 0.0
                scores[f"{mod}_roi"] = roi_scores

        # Normalise so modality scores sum to 1 at each vertex
        total = sum(scores[m] for m in modalities)
        total = np.where(total > 0, total, 1.0)
        for mod in modalities:
            scores[f"{mod}_normalised"] = scores[mod] / total

        return scores
