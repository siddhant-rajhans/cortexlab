"""ROI attention map extraction from the transformer encoder.

Provides :class:`AttentionExtractor`, a context manager that hooks into
the encoder's attention layers and captures per-head attention weights
during a forward pass.  The raw maps can then be projected onto HCP
MMP1.0 brain ROIs via :func:`attention_to_roi_scores`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


@contextmanager
def AttentionExtractor(encoder: nn.Module):
    """Context manager that captures attention weights from transformer layers.

    Registers forward hooks on every sub-module whose class name contains
    ``"Attention"`` (case-insensitive).  Each hook stores the second element
    of the output tuple (the attention weights) if the layer returns one,
    or falls back to a ``_attn_weights`` attribute when present.

    Yields
    ------
    attn_maps : list[torch.Tensor]
        Mutable list that accumulates attention tensors of shape
        ``(B, heads, T, T)`` as the model runs its forward pass.

    Example
    -------
    >>> with AttentionExtractor(model.encoder) as maps:
    ...     out = model(batch)
    >>> len(maps)  # one tensor per attention layer
    8
    """
    attn_maps: list[torch.Tensor] = []
    hooks = []

    for module in encoder.modules():
        if "attention" in module.__class__.__name__.lower():

            def _hook(mod, inp, out, store=attn_maps):
                if isinstance(out, tuple) and len(out) >= 2:
                    second = out[1]
                    if second is not None:
                        if hasattr(second, "post_softmax_attn") and second.post_softmax_attn is not None:
                            store.append(second.post_softmax_attn.detach())
                        elif hasattr(second, "detach"):
                            store.append(second.detach())
                elif hasattr(mod, "_attn_weights") and mod._attn_weights is not None:
                    store.append(mod._attn_weights.detach())

            hooks.append(module.register_forward_hook(_hook))

    try:
        yield attn_maps
    finally:
        for h in hooks:
            h.remove()


def attention_to_roi_scores(
    attn_maps: list[torch.Tensor],
    roi_indices: dict[str, np.ndarray],
    predictor_weights: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Project raw attention maps onto brain ROIs.

    Parameters
    ----------
    attn_maps : list[torch.Tensor]
        Attention tensors from :class:`AttentionExtractor`, each of shape
        ``(B, heads, T, T)``.
    roi_indices : dict[str, np.ndarray]
        Mapping from ROI name to vertex indices (e.g. from
        :func:`cortexlab.data.loader.get_hcp_labels`).
    predictor_weights : torch.Tensor, optional
        Predictor layer weights of shape ``(n_subjects, hidden, n_vertices)``
        or ``(hidden, n_vertices)``.  When provided, each ROI's temporal
        profile is weighted by the L2 norm of the predictor weights for
        its vertices, emphasising ROIs the model attends to more strongly.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from ROI name to a 1-D temporal importance array of
        length ``T`` (number of time steps in the attention maps).
    """
    if not attn_maps:
        return {name: np.array([]) for name in roi_indices}

    # Stack layers, average over batch and heads -> (T, T)
    stacked = torch.stack(attn_maps)  # (layers, B, heads, T, T)
    avg_attn = stacked.mean(dim=(0, 1, 2))  # (T, T)
    # Per-timestep importance: sum over keys for each query position
    temporal_importance = avg_attn.sum(dim=-1).cpu().numpy()  # (T,)

    roi_scores: dict[str, np.ndarray] = {}
    for name, vertices in roi_indices.items():
        if predictor_weights is not None:
            w = predictor_weights
            if w.ndim == 3:
                w = w.mean(dim=0)  # average across subjects
            # Weight = L2 norm of predictor weights for this ROI's vertices
            roi_weight = w[:, vertices].norm(dim=0).mean().item()
        else:
            roi_weight = 1.0
        roi_scores[name] = temporal_importance * roi_weight

    return roi_scores
