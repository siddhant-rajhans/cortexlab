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
    p_values
        When :func:`run_modality_lesion` is called with
        ``n_permutations > 0``, ``p_values[m]`` is a ``(n_voxels,)``
        float32 tensor of one-sided p-values for the null that
        modality ``m``'s test-time features are uninformative
        (constructed by row-permuting ``X_test[:, slice(m)]``).
        Smaller values mean the observed ``delta_r2[m]`` is unlikely
        under the null. ``None`` when no permutation test was run.
    n_permutations
        Number of random permutations used; 0 when no permutation test
        was performed.
    """

    full_r2: torch.Tensor
    delta_r2: dict[str, torch.Tensor]
    modality_order: list[str]
    mask_strategy: MaskStrategy
    n_train: int
    n_test: int
    best_alpha: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    p_values: dict[str, torch.Tensor] | None = None
    n_permutations: int = 0


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
    n_permutations: int = 0,
    permutation_seed: int = 0,
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
    n_permutations
        If > 0, run a modality-wise label-permutation test against the
        null that modality ``m``'s test features are uninformative. For
        each of ``n_permutations`` random permutations, the rows of
        ``X_test[:, slice(m)]`` are shuffled across stimuli (the encoder
        is NOT refit) and the resulting ``delta_r2_null`` compared to
        the observed ``delta_r2``. Per-voxel one-sided p-values
        (``(count(null >= observed) + 1) / (n_permutations + 1)``) are
        returned in ``LesionResult.p_values``.
    permutation_seed
        RNG seed for reproducible permutations.

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

    # The encoder moves X/Y to `device` internally during fit, but Y_test
    # was never passed through the encoder, so it is still on the caller's
    # device (typically CPU). Predictions come back on the encoder's
    # device, so align Y_test before scoring to avoid the
    # "tensors on different devices" runtime error.
    target_device = enc.coef_.device
    X_test = X_test.to(target_device)
    Y_test = Y_test.to(target_device)

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
    r2_ablated: dict[str, torch.Tensor] = {}
    for m in modality_order:
        X_test_ablated = X_test.clone()
        sl = slices[m]
        X_test_ablated[:, sl] = mask_values[sl]
        Y_hat_m = enc.predict(X_test_ablated)
        r2_m = _r2_score(Y_test, Y_hat_m)
        delta[m] = r2_full - r2_m
        r2_ablated[m] = r2_m
        logger.info(
            "  lesion %s: mean dR^2 = %+.4f (top quintile %+.4f)",
            m, delta[m].mean().item(),
            delta[m].quantile(0.8).item(),
        )

    # Optional permutation test: shuffle the test-time stimulus order of
    # modality m's feature block only (leaving the other modalities' rows
    # intact) and re-evaluate. Under the null "modality m adds nothing at
    # test time", delta_null should be distributed around delta_observed;
    # under the alternative, delta_observed is much larger.
    p_values: dict[str, torch.Tensor] | None = None
    if n_permutations > 0:
        if n_permutations < 0:
            raise ValueError(f"n_permutations must be >= 0, got {n_permutations}")
        n_test = X_test.shape[0]
        n_vox = int(Y_test.shape[1])
        # Seeded RNG on CPU so permutations are reproducible regardless
        # of target_device (CUDA randperm with a generator is fiddly).
        rng = torch.Generator(device="cpu").manual_seed(int(permutation_seed))
        counts = {
            m: torch.zeros(n_vox, dtype=torch.int32, device=target_device)
            for m in modality_order
        }
        for b in range(n_permutations):
            perm = torch.randperm(n_test, generator=rng).to(target_device)
            for m in modality_order:
                sl = slices[m]
                X_perm = X_test.clone()
                X_perm[:, sl] = X_test[perm][:, sl]
                Y_hat_null = enc.predict(X_perm)
                r2_full_null = _r2_score(Y_test, Y_hat_null)
                # The ablated prediction is invariant under this
                # row-permutation because the slice is replaced by a
                # constant mask; reuse the observed r2_ablated[m].
                delta_null = r2_full_null - r2_ablated[m]
                counts[m] += (delta_null >= delta[m]).to(torch.int32)
            if (b + 1) % max(1, n_permutations // 10) == 0:
                logger.info(
                    "  permutation %d / %d done",
                    b + 1, n_permutations,
                )
        # "+1" smoothing so p is never exactly zero when none of the
        # nulls exceeded observed (classic Phipson & Smyth 2010 fix).
        p_values = {
            m: (counts[m].to(torch.float32) + 1.0) / (n_permutations + 1.0)
            for m in modality_order
        }
        for m in modality_order:
            frac_sig = float((p_values[m] < 0.05).float().mean().item())
            logger.info(
                "  %s: median p = %.3f, fraction p<0.05 = %.3f",
                m, float(p_values[m].median().item()), frac_sig,
            )

    return LesionResult(
        full_r2=r2_full,
        delta_r2=delta,
        modality_order=modality_order,
        mask_strategy=mask_strategy,
        n_train=int(Y_train.shape[0]),
        n_test=int(Y_test.shape[0]),
        best_alpha=enc.best_alpha_,
        p_values=p_values,
        n_permutations=int(n_permutations),
    )


def roi_summary(
    result: LesionResult,
    roi_indices: Mapping[str, np.ndarray],
    ceiling: np.ndarray | None = None,
    min_ceiling: float = 0.01,
) -> dict[str, dict[str, float]]:
    """Aggregate a LesionResult over ROIs.

    Parameters
    ----------
    result
        Output of :func:`run_modality_lesion`.
    roi_indices
        Mapping ROI name to numpy array of voxel indices (as in the
        project's ``roi_indices`` fixture).
    ceiling
        Optional per-voxel noise ceiling in R^2 space, shape
        ``(n_voxels,)``. When provided, each ROI row gains a
        ``full_r2_normalized`` entry (model R^2 divided by ceiling per
        voxel, averaged over the ROI) and a ``ceiling_mean`` entry so
        downstream tables can report both raw and ceiling-normalized
        scores.
    min_ceiling
        Voxels with ceiling below this threshold are dropped from the
        normalized mean to avoid division instability.

    Returns
    -------
    dict
        ``{roi: {"full_r2": float, "dR2_<m>": float, ...}}``, values
        are ROI-mean scores. Adds ``full_r2_normalized`` and
        ``ceiling_mean`` per ROI when ``ceiling`` is provided. Adds
        ``p_<m>_median`` and ``frac_sig_<m>`` (at alpha=0.05) per ROI
        when ``result.p_values`` is populated by
        :func:`run_modality_lesion` with ``n_permutations > 0``.
    """
    out: dict[str, dict[str, float]] = {}
    full = result.full_r2.cpu().numpy()
    dr2 = {m: result.delta_r2[m].cpu().numpy() for m in result.modality_order}
    p_vals: dict[str, np.ndarray] | None = None
    if result.p_values is not None:
        p_vals = {m: result.p_values[m].cpu().numpy() for m in result.modality_order}

    if ceiling is not None:
        ceiling = np.asarray(ceiling)
        if ceiling.shape != full.shape:
            raise ValueError(
                f"ceiling shape {ceiling.shape} does not match full_r2 {full.shape}"
            )

    for roi, idx in roi_indices.items():
        row = {"full_r2": float(np.mean(full[idx]))}
        for m in result.modality_order:
            row[f"dR2_{m}"] = float(np.mean(dr2[m][idx]))
            if p_vals is not None:
                row[f"p_{m}_median"] = float(np.median(p_vals[m][idx]))
                row[f"frac_sig_{m}"] = float(np.mean(p_vals[m][idx] < 0.05))
        if ceiling is not None:
            c_roi = ceiling[idx]
            mask = c_roi > min_ceiling
            if mask.any():
                normalized = full[idx][mask] / c_roi[mask]
                row["full_r2_normalized"] = float(np.mean(normalized))
            else:
                row["full_r2_normalized"] = float("nan")
            row["ceiling_mean"] = float(np.mean(c_roi))
        out[roi] = row
    return out


def _to_tensor(arr, dtype=torch.float32):
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr)
    else:
        t = arr
    return t.to(dtype=dtype).contiguous()
