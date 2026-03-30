"""Cross-subject adaptation for the fMRI encoder.

Provides :class:`SubjectAdapter` with two strategies for adapting the
pre-trained model to a new, unseen subject from a small calibration set:

* **Ridge regression** fits a new predictor head from calibration fMRI.
* **Nearest-neighbour** picks the most similar training subject (zero-shot).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SubjectAdapter:
    """Holds new-subject predictor weights ready for injection into a model."""

    def __init__(self, weights: torch.Tensor, bias: torch.Tensor | None = None):
        self._weights = weights  # (1, in_channels, n_outputs)
        self._bias = bias  # (1, n_outputs) or None

    @classmethod
    def from_ridge(
        cls,
        model: nn.Module,
        calibration_loader: DataLoader,
        regularization: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> SubjectAdapter:
        """Fit a ridge-regression predictor head from calibration data.

        Runs the frozen backbone on every batch in *calibration_loader*,
        collects hidden states and fMRI targets, then solves the normal
        equations ``W* = (X^T X + lambda I)^{-1} X^T y``.

        Parameters
        ----------
        model : nn.Module
            The :class:`FmriEncoderModel` instance (must be in eval mode).
        calibration_loader : DataLoader
            Provides ``(batch, ...)`` where ``batch.data["fmri"]`` contains
            ground-truth fMRI for the new subject.
        regularization : float
            Ridge penalty (lambda).
        device : str or torch.device
            Device for computation.
        """
        model.eval()
        all_hidden, all_targets = [], []

        with torch.inference_mode():
            for batch in calibration_loader:
                batch = batch.to(device)
                x = model.aggregate_features(batch)
                if hasattr(model, "temporal_smoothing"):
                    x = model.temporal_smoothing(x.transpose(1, 2)).transpose(1, 2)
                if not model.config.linear_baseline:
                    x = model.transformer_forward(x)
                x = x.transpose(1, 2)  # B, H, T
                if model.config.low_rank_head is not None:
                    x = model.low_rank_head(x.transpose(1, 2)).transpose(1, 2)
                # Pool to match target temporal dimension
                x = model.pooler(x)  # B, H, T'
                target = batch.data["fmri"]  # B, V, T'

                # Flatten time: (B*T', H) and (B*T', V)
                B, H, T = x.shape
                x_flat = x.permute(0, 2, 1).reshape(-1, H)
                t_flat = target.permute(0, 2, 1).reshape(-1, target.shape[1])
                all_hidden.append(x_flat.cpu())
                all_targets.append(t_flat.cpu())

        X = torch.cat(all_hidden, dim=0).float()  # (N, H)
        Y = torch.cat(all_targets, dim=0).float()  # (N, V)

        # Ridge regression: W = (X^T X + lambda I)^{-1} X^T Y
        XtX = X.T @ X
        reg = regularization * torch.eye(XtX.shape[0])
        W = torch.linalg.solve(XtX + reg, X.T @ Y)  # (H, V)
        weights = W.unsqueeze(0)  # (1, H, V)

        logger.info(
            "Ridge adapter fitted: %d samples, hidden=%d, vertices=%d",
            X.shape[0],
            X.shape[1],
            Y.shape[1],
        )
        return cls(weights=weights)

    @classmethod
    def from_nearest_neighbor(
        cls,
        model: nn.Module,
        calibration_loader: DataLoader,
        device: str | torch.device = "cpu",
    ) -> SubjectAdapter:
        """Zero-shot adaptation by finding the closest training subject.

        Computes a mean hidden-state signature for the new subject and
        matches it to each training subject's predictor weight signature
        via cosine similarity.

        Parameters
        ----------
        model : nn.Module
            The :class:`FmriEncoderModel` instance.
        calibration_loader : DataLoader
            Provides calibration batches for the new subject.
        device : str or torch.device
            Device for computation.
        """
        model.eval()
        all_hidden = []

        with torch.inference_mode():
            for batch in calibration_loader:
                batch = batch.to(device)
                x = model.aggregate_features(batch)
                if hasattr(model, "temporal_smoothing"):
                    x = model.temporal_smoothing(x.transpose(1, 2)).transpose(1, 2)
                if not model.config.linear_baseline:
                    x = model.transformer_forward(x)
                all_hidden.append(x.mean(dim=(0, 1)).cpu())

        new_sig = torch.stack(all_hidden).mean(dim=0)  # (H,)

        # Compare against each training subject's predictor weights
        pred_weights = model.predictor.weights  # (n_subjects, in_ch, out_ch)
        n_subjects = pred_weights.shape[0]
        best_sim, best_idx = -1.0, 0
        for i in range(n_subjects):
            w_sig = pred_weights[i].mean(dim=-1).cpu()  # (in_ch,)
            # Truncate/pad to match dimensions if needed
            dim = min(w_sig.shape[0], new_sig.shape[0])
            sim = torch.nn.functional.cosine_similarity(
                w_sig[:dim].unsqueeze(0), new_sig[:dim].unsqueeze(0)
            ).item()
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        logger.info(
            "Nearest-neighbour match: subject %d (cosine sim %.4f)", best_idx, best_sim
        )
        weights = pred_weights[best_idx : best_idx + 1].detach().cpu()
        bias = None
        if hasattr(model.predictor, "bias") and model.predictor.bias is not None:
            bias = model.predictor.bias[best_idx : best_idx + 1].detach().cpu()
        return cls(weights=weights, bias=bias)

    def inject_into_model(self, model: nn.Module) -> int:
        """Append the adapted weights as a new subject in the predictor.

        Returns the integer subject ID assigned to the new subject.
        """
        pred = model.predictor
        old_weights = pred.weights.data  # (n_subjects, in_ch, out_ch)
        new_weights = self._weights.to(old_weights.device)
        pred.weights = nn.Parameter(
            torch.cat([old_weights, new_weights], dim=0)
        )
        new_id = old_weights.shape[0]

        if self._bias is not None and hasattr(pred, "bias") and pred.bias is not None:
            old_bias = pred.bias.data
            new_bias = self._bias.to(old_bias.device)
            pred.bias = nn.Parameter(torch.cat([old_bias, new_bias], dim=0))

        logger.info("Injected new subject as ID %d", new_id)
        return new_id
