"""Causal modality lesion protocol for multimodal fMRI encoders.

The question: which cortical regions causally depend on each input modality
(text / audio / video) of a multimodal foundation-model encoder?

The answer, per voxel ``v`` and per modality ``m``::

    delta_v^(m) = R^2(y_v, y_hat_v)  -  R^2(y_v, y_hat_v^(-m))

where ``y_hat_v`` is the full multimodal prediction and ``y_hat_v^(-m)`` is
the prediction with modality ``m`` masked. A large positive delta means
voxel ``v`` causally depends on modality ``m``.

Two mask strategies are supported:

* ``zero``: replace the modality's input with zeros. Cheapest; small risk
  of an out-of-distribution artefact.
* ``learned``: replace with the per-training-stimulus mean of that
  modality. Rules out "zero input ⇒ OOD" as a trivial explanation.

Inputs are not raw stimuli but *features* from a multimodal encoder,
partitioned by modality (dict mapping modality name to feature array).
Ridge regression fits the full feature set; ablation zeros or replaces
only the columns belonging to the lesioned modality.

This module is infrastructure-agnostic: the heavy ridge solve is
delegated to :class:`cortexlab.gpu.ridge.VoxelRidgeEncoder`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

from cortexlab.gpu.ridge import VoxelRidgeEncoder, _r2_score

logger = logging.getLogger(__name__)


MaskStrategy = str  # "zero" | "learned"


@dataclass
class LesionResult:
    """Outputs of one full lesion study.

    Attributes
    ----------
    full_r2
        Per-voxel R^2 of the full multimodal prediction on the held-out
        test set, shape ``(n_voxels,)``.
    delta_r2
        ``delta_r2[m]`` is per-voxel dR^2 when modality ``m`` is masked,
        shape ``(n_voxels,)`` per modality.
    modality_order
        Iteration order used for modalities (stable dict keys).
    mask_strategy
        Which mask was used (``"zero"`` or ``"learned"``).
    n_train
        Number of training stimuli used to fit the encoder.
    n_test
        Number of held-out test stimuli used to score predictions.
    best_alpha
        Per-voxel selected ridge alpha (diagnostic).
    """

    full_r2: torch.Tensor
    delta_r2: dict[str, torch.Tensor]
    modality_order: list[str]
    mask_strategy: MaskStrategy
    n_train: int
    n_test: int
    best_alpha: torch.Tensor = field(default_factory=lambda: torch.empty(0))


def run_modality_lesion(
    features_train: Mapping[str, np.ndarray | torch.Tensor],
    features_test: Mapping[str, np.ndarray | torch.Tensor],
    y_train: np.ndarray | torch.Tensor,
    y_test: np.ndarray | torch.Tensor,
    alphas: list[float] | None = None,
    cv: int = 5,
    mask_strategy: MaskStrategy = "zero",
    device: str | None = None,
    backend: str = "auto",
) -> LesionResult:
    """Run the causal modality lesion protocol.

    Parameters
    ----------
    features_train, features_test
        Dicts mapping modality name (e.g. ``"text"``, ``"audio"``, ``"video"``)
        to feature arrays of shape ``(n_stim, n_feat_m)``. All modalities
        must have the same ``n_stim`` within each split.
    y_train, y_test
        Voxel responses ``(n_stim, n_voxels)`` for the matching split.
    alphas
        Ridge regularization grid. Defaults to ``[1e-2, 1, 1e2, 1e4, 1e6]``.
    cv
        Number of CV folds for alpha selection.
    mask_strategy
        ``"zero"`` or ``"learned"``.
    device
        Torch device for the ridge solve.
    backend
        Ridge backend: ``"torch"``, ``"triton"``, or ``"auto"``.

    Returns
    -------
    LesionResult
    """
    if mask_strategy not in ("zero", "learned"):
        raise ValueError(f"mask_strategy must be zero|learned, got {mask_strategy!r}")
    if list(features_train.keys()) != list(features_test.keys()):
        raise ValueError("features_train and features_test must share keys in order")
    if len(features_train) < 2:
        raise ValueError("lesion study needs at least 2 modalities")

    modality_order = list(features_train.keys())
    logger.info("Lesion study on modalities: %s (mask=%s)",
                modality_order, mask_strategy)

    # Concatenate modality features along the feature axis, tracking
    # per-modality column slices for later ablation.
    slices: dict[str, slice] = {}
    cur = 0
    X_train_parts = []
    X_test_parts = []
    for m in modality_order:
        tr = _to_tensor(features_train[m])
        te = _to_tensor(features_test[m])
        if tr.shape[0] != next(iter(features_train.values())).shape[0]:
            raise ValueError(f"modality {m!r} train n mismatch")
        if te.shape[0] != next(iter(features_test.values())).shape[0]:
            raise ValueError(f"modality {m!r} test n mismatch")
        if tr.shape[1] != te.shape[1]:
            raise ValueError(f"modality {m!r} feature-dim mismatch between train/test")
        slices[m] = slice(cur, cur + tr.shape[1])
        cur += tr.shape[1]
        X_train_parts.append(tr)
        X_test_parts.append(te)

    X_train = torch.cat(X_train_parts, dim=1)
    X_test = torch.cat(X_test_parts, dim=1)
    Y_train = _to_tensor(y_train)
    Y_test = _to_tensor(y_test)

    # Fit full multimodal encoder once.
    enc = VoxelRidgeEncoder(
        alphas=alphas or [1e-2, 1.0, 1e2, 1e4, 1e6],
        cv=cv, backend=backend, device=device,
    ).fit(X_train, Y_train)
    Y_hat_full = enc.predict(X_test)
    r2_full = _r2_score(Y_test, Y_hat_full)

    # For the "learned" mask we need per-column training means.
    if mask_strategy == "learned":
        mask_values = X_train.mean(dim=0)
    else:
        mask_values = torch.zeros(X_train.shape[1], dtype=X_train.dtype,
                                  device=X_train.device)

    # For each modality, construct the lesioned X_test and re-score.
    # Note we do NOT refit the encoder - that would defeat the lesion:
    # refitting lets the model compensate by reweighting the surviving
    # modalities. We keep the original encoder and ask "what if this
    # modality had been absent at inference time?".
    delta: dict[str, torch.Tensor] = {}
    for m in modality_order:
        X_test_ablated = X_test.clone()
        sl = slices[m]
        X_test_ablated[:, sl] = mask_values[sl]
        Y_hat_m = enc.predict(X_test_ablated)
        r2_m = _r2_score(Y_test, Y_hat_m)
        delta[m] = r2_full - r2_m
        logger.info(
            "  lesion %s: mean dR^2 = %+.4f (top quintile %+.4f)",
            m, delta[m].mean().item(),
            delta[m].quantile(0.8).item(),
        )

    return LesionResult(
        full_r2=r2_full,
        delta_r2=delta,
        modality_order=modality_order,
        mask_strategy=mask_strategy,
        n_train=int(Y_train.shape[0]),
        n_test=int(Y_test.shape[0]),
        best_alpha=enc.best_alpha_,
    )


def roi_summary(
    result: LesionResult,
    roi_indices: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Aggregate a LesionResult over ROIs.

    Parameters
    ----------
    result
        Output of :func:`run_modality_lesion`.
    roi_indices
        Mapping ROI name to numpy array of voxel indices (as in the
        project's ``roi_indices`` fixture).

    Returns
    -------
    dict
        ``{roi: {"full_r2": float, "dR2_<m>": float, ...}}``, values
        are ROI-mean scores.
    """
    out: dict[str, dict[str, float]] = {}
    full = result.full_r2.cpu().numpy()
    dr2 = {m: result.delta_r2[m].cpu().numpy() for m in result.modality_order}
    for roi, idx in roi_indices.items():
        row = {"full_r2": float(np.mean(full[idx]))}
        for m in result.modality_order:
            row[f"dR2_{m}"] = float(np.mean(dr2[m][idx]))
        out[roi] = row
    return out


def _to_tensor(arr, dtype=torch.float32):
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr)
    else:
        t = arr
    return t.to(dtype=dtype).contiguous()
