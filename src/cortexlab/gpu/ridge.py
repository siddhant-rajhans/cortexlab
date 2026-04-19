"""Voxelwise ridge-regression encoder with cross-validated regularization.

The encoder solves, per voxel v and per regularization strength lambda,

    beta_{lambda, v} = argmin_beta ||X beta - y_v||_2^2 + lambda ||beta||_2^2
                    = (X^T X + lambda I)^{-1} X^T y_v

on the training fold, evaluates each lambda on the held-out fold, picks the
best lambda per voxel across folds, and refits on all data with that choice.

Two backends share one API:

* ``torch``: dense closed-form solution via a single SVD of X per fold.
  Works on CPU or CUDA; fine for development and for moderate feature
  dimensions. Used as the correctness reference for the Triton backend.

* ``triton``: fused Triton kernel that batches the per-voxel solve on an
  NVIDIA GPU. Imported lazily so the module is usable on systems without
  a CUDA toolchain. Falls back to ``torch`` when Triton is unavailable.

Both backends match ``sklearn.linear_model.RidgeCV`` to within 1e-5 on
synthetic data (see ``tests/test_gpu_ridge.py``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)

_DEFAULT_ALPHAS: tuple[float, ...] = (1e-2, 1e0, 1e2, 1e4, 1e6)


@dataclass
class RidgeFitResult:
    """Fitted ridge encoder state.

    Attributes
    ----------
    coef_
        Regression weights, shape ``(n_features, n_voxels)``, in the space
        of the (optionally centered) training inputs.
    intercept_
        Per-voxel intercept, shape ``(n_voxels,)``. Zero when ``fit_intercept``
        is False.
    best_alpha_
        Per-voxel selected regularization strength, shape ``(n_voxels,)``.
    cv_scores_
        Per-fold, per-alpha, per-voxel validation R^2, shape
        ``(n_folds, n_alphas, n_voxels)``. Useful for diagnostics.
    x_mean_
        Feature means from the training set when ``fit_intercept`` is True.
    y_mean_
        Target means from the training set when ``fit_intercept`` is True.
    """

    coef_: torch.Tensor
    intercept_: torch.Tensor
    best_alpha_: torch.Tensor
    cv_scores_: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    x_mean_: torch.Tensor | None = None
    y_mean_: torch.Tensor | None = None


class VoxelRidgeEncoder:
    """Cross-validated voxelwise ridge regression.

    Parameters
    ----------
    alphas
        Regularization strengths to search. Must be positive and finite.
    cv
        Number of folds for selecting alpha. ``cv=1`` disables CV and fits
        on all data with ``alphas[0]``.
    fit_intercept
        Center X and y using training-set means before fitting. Recommended.
    backend
        ``"torch"``, ``"triton"``, or ``"auto"``. ``"auto"`` picks Triton
        when a CUDA device is available and the triton package imports,
        otherwise torch.
    device
        Torch device (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ...). When None,
        inputs keep their device.
    dtype
        Working dtype for the solve. Defaults to ``torch.float32``.
    voxel_chunk
        Maximum voxels processed at once to bound memory. None = all at once.
    """

    def __init__(
        self,
        alphas: list[float] | tuple[float, ...] = _DEFAULT_ALPHAS,
        cv: int = 5,
        fit_intercept: bool = True,
        backend: str = "torch",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        voxel_chunk: int | None = None,
    ) -> None:
        if len(alphas) == 0:
            raise ValueError("alphas must be non-empty")
        if any(a <= 0 or not math.isfinite(a) for a in alphas):
            raise ValueError("alphas must be positive and finite")
        if cv < 1:
            raise ValueError("cv must be >= 1")
        if backend not in {"torch", "triton", "auto"}:
            raise ValueError(f"backend must be torch|triton|auto, got {backend!r}")

        self.alphas = tuple(float(a) for a in alphas)
        self.cv = int(cv)
        self.fit_intercept = bool(fit_intercept)
        self.backend_request = backend
        self.device_request = device
        self.dtype = dtype
        self.voxel_chunk = voxel_chunk
        self._fit: RidgeFitResult | None = None

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #

    def fit(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray,
    ) -> VoxelRidgeEncoder:
        """Fit ridge weights per voxel with cross-validated alpha."""
        X_t, Y_t = self._prepare(X, Y)
        backend = self._resolve_backend(X_t.device)
        logger.info(
            "VoxelRidgeEncoder.fit: backend=%s, n=%d, p=%d, v=%d, alphas=%d, cv=%d",
            backend, X_t.shape[0], X_t.shape[1], Y_t.shape[1], len(self.alphas), self.cv,
        )

        if self.fit_intercept:
            x_mean = X_t.mean(dim=0)
            y_mean = Y_t.mean(dim=0)
            Xc = X_t - x_mean
            Yc = Y_t - y_mean
        else:
            x_mean = y_mean = None
            Xc, Yc = X_t, Y_t

        if self.cv == 1 or len(self.alphas) == 1:
            alpha_idx = torch.zeros(Yc.shape[1], dtype=torch.long, device=Xc.device)
            cv_scores = torch.empty(0, device=Xc.device)
        else:
            cv_scores = self._cross_validate(Xc, Yc, backend)
            # (n_alphas, n_voxels) mean score across folds
            mean_scores = cv_scores.mean(dim=0)
            alpha_idx = mean_scores.argmax(dim=0)

        alphas_t = torch.tensor(self.alphas, dtype=self.dtype, device=Xc.device)
        best_alpha = alphas_t[alpha_idx]

        coef = self._refit_per_voxel(Xc, Yc, alpha_idx, backend)
        if self.fit_intercept:
            intercept = y_mean - (x_mean @ coef)
        else:
            intercept = torch.zeros(Yc.shape[1], dtype=self.dtype, device=Xc.device)

        self._fit = RidgeFitResult(
            coef_=coef,
            intercept_=intercept,
            best_alpha_=best_alpha,
            cv_scores_=cv_scores,
            x_mean_=x_mean,
            y_mean_=y_mean,
        )
        return self

    def predict(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Predict voxel responses for new stimuli. Shape ``(n, n_voxels)``."""
        if self._fit is None:
            raise RuntimeError("call fit() before predict()")
        X_t = self._to_tensor(X, self._fit.coef_.device)
        return X_t @ self._fit.coef_ + self._fit.intercept_

    def score(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Per-voxel R^2 on held-out data. Shape ``(n_voxels,)``."""
        if self._fit is None:
            raise RuntimeError("call fit() before score()")
        Y_t = self._to_tensor(Y, self._fit.coef_.device)
        Y_hat = self.predict(X)
        return _r2_score(Y_t, Y_hat)

    @property
    def coef_(self) -> torch.Tensor:
        return self._fit.coef_ if self._fit is not None else None  # type: ignore[return-value]

    @property
    def intercept_(self) -> torch.Tensor:
        return self._fit.intercept_ if self._fit is not None else None  # type: ignore[return-value]

    @property
    def best_alpha_(self) -> torch.Tensor:
        return self._fit.best_alpha_ if self._fit is not None else None  # type: ignore[return-value]

    @property
    def cv_scores_(self) -> torch.Tensor:
        return self._fit.cv_scores_ if self._fit is not None else None  # type: ignore[return-value]

    # --------------------------------------------------------------------- #
    # internals
    # --------------------------------------------------------------------- #

    def _resolve_backend(self, device: torch.device) -> str:
        req = self.backend_request
        if req == "torch":
            return "torch"
        if req == "triton":
            if not _triton_available(device):
                raise RuntimeError(
                    "backend='triton' requested but Triton is unavailable "
                    "(requires CUDA device and a working triton install)."
                )
            return "triton"
        # auto
        return "triton" if _triton_available(device) else "torch"

    def _prepare(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(self.device_request) if self.device_request is not None else None
        X_t = self._to_tensor(X, device)
        Y_t = self._to_tensor(Y, X_t.device)
        if X_t.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {tuple(X_t.shape)}")
        if Y_t.ndim != 2:
            raise ValueError(f"Y must be 2-D, got shape {tuple(Y_t.shape)}")
        if X_t.shape[0] != Y_t.shape[0]:
            raise ValueError(
                f"X and Y must share first axis: got {X_t.shape[0]} vs {Y_t.shape[0]}"
            )
        return X_t, Y_t

    def _to_tensor(
        self,
        arr: torch.Tensor | np.ndarray,
        device: torch.device | None,
    ) -> torch.Tensor:
        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr)
        elif isinstance(arr, torch.Tensor):
            t = arr
        else:
            raise TypeError(f"expected Tensor or ndarray, got {type(arr).__name__}")
        t = t.to(dtype=self.dtype)
        if device is not None:
            t = t.to(device)
        return t.contiguous()

    def _cross_validate(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        backend: str,
    ) -> torch.Tensor:
        """Return per-fold, per-alpha, per-voxel validation R^2."""
        n = X.shape[0]
        if self.cv > n:
            raise ValueError(f"cv={self.cv} exceeds n_samples={n}")
        fold_ids = torch.arange(n) % self.cv
        n_alphas = len(self.alphas)
        scores = torch.empty(self.cv, n_alphas, Y.shape[1], dtype=self.dtype, device=X.device)

        for k in range(self.cv):
            val_mask = fold_ids == k
            train_mask = ~val_mask
            X_tr, Y_tr = X[train_mask], Y[train_mask]
            X_va, Y_va = X[val_mask], Y[val_mask]
            # Pre-center per-fold so the intercept is absorbed into the mean.
            if self.fit_intercept:
                mx = X_tr.mean(dim=0)
                my = Y_tr.mean(dim=0)
                X_tr = X_tr - mx
                Y_tr = Y_tr - my
                X_va = X_va - mx
                Y_va = Y_va - my
            coef_per_alpha = _solve_ridge_all_alphas(X_tr, Y_tr, self.alphas, backend)
            # coef_per_alpha: (n_alphas, p, v)
            for ai in range(n_alphas):
                Y_hat = X_va @ coef_per_alpha[ai]
                scores[k, ai] = _r2_score(Y_va, Y_hat)
        return scores

    def _refit_per_voxel(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        alpha_idx: torch.Tensor,
        backend: str,
    ) -> torch.Tensor:
        """Final refit: for each unique alpha in alpha_idx, solve once."""
        coef = torch.empty(X.shape[1], Y.shape[1], dtype=self.dtype, device=X.device)
        unique_ai = torch.unique(alpha_idx)
        for ai_scalar in unique_ai.tolist():
            mask = alpha_idx == ai_scalar
            if not bool(mask.any()):
                continue
            alpha = self.alphas[int(ai_scalar)]
            Y_sub = Y[:, mask]
            coef_sub = _solve_ridge_single_alpha(X, Y_sub, alpha, backend)
            coef[:, mask] = coef_sub
        return coef


# --------------------------------------------------------------------------- #
# low-level solve routines
# --------------------------------------------------------------------------- #

def _solve_ridge_all_alphas(
    X: torch.Tensor,
    Y: torch.Tensor,
    alphas: tuple[float, ...],
    backend: str,
) -> torch.Tensor:
    """Solve ridge for every alpha in the grid using a single SVD.

    Returns coefficients of shape ``(n_alphas, p, v)``.

    The SVD approach:

        X = U S V^T
        beta(alpha) = V @ diag(s / (s^2 + alpha)) @ U^T @ Y
    """
    if backend == "triton":
        return _solve_ridge_all_alphas_triton(X, Y, alphas)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # U: (n,r) S: (r,) Vh: (r,p)
    UtY = U.T @ Y  # (r, v)
    out = torch.empty(len(alphas), X.shape[1], Y.shape[1], dtype=X.dtype, device=X.device)
    for i, alpha in enumerate(alphas):
        scale = S / (S * S + alpha)  # (r,)
        out[i] = Vh.T @ (scale.unsqueeze(1) * UtY)
    return out


def _solve_ridge_single_alpha(
    X: torch.Tensor,
    Y: torch.Tensor,
    alpha: float,
    backend: str,
) -> torch.Tensor:
    """Solve ridge once. Returns ``(p, v)``."""
    if backend == "triton":
        return _solve_ridge_single_alpha_triton(X, Y, alpha)
    # Cholesky of (X'X + lambda I) is numerically stable and fast for p small.
    p = X.shape[1]
    A = X.T @ X
    A.diagonal().add_(alpha)
    B = X.T @ Y
    try:
        L = torch.linalg.cholesky(A)
        return torch.cholesky_solve(B, L)
    except torch.linalg.LinAlgError:
        logger.warning("Cholesky failed; falling back to generic solve")
        A.diagonal().sub_(alpha)
        A.diagonal().add_(alpha)
        return torch.linalg.solve(A, B)


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Per-column R^2 = 1 - SS_res / SS_tot. Shape ``(n_targets,)``.

    Matches the sklearn convention: when SS_tot == 0 the column is undefined;
    we return 0.0 for those columns to avoid NaN propagation downstream.
    """
    residual = (y_true - y_pred).pow(2).sum(dim=0)
    total = (y_true - y_true.mean(dim=0)).pow(2).sum(dim=0)
    r2 = torch.ones_like(total)
    mask = total > 0
    r2[mask] = 1.0 - residual[mask] / total[mask]
    r2[~mask] = 0.0
    return r2


# --------------------------------------------------------------------------- #
# Triton backend (optional)
# --------------------------------------------------------------------------- #

_TRITON_CHECKED = False
_TRITON_OK = False


def _triton_available(device: torch.device) -> bool:
    global _TRITON_CHECKED, _TRITON_OK
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    if _TRITON_CHECKED:
        return _TRITON_OK
    try:
        import triton  # noqa: F401
        _TRITON_OK = True
    except ImportError:
        _TRITON_OK = False
    _TRITON_CHECKED = True
    return _TRITON_OK


def _solve_ridge_all_alphas_triton(
    X: torch.Tensor,
    Y: torch.Tensor,
    alphas: tuple[float, ...],
) -> torch.Tensor:
    """Triton-accelerated grid solve.

    The outer math is identical to the torch backend; the Triton kernel
    replaces the fused ``V diag(s/(s^2+alpha)) U^T Y`` matmul chain with a
    single pass that tiles over voxels and alphas simultaneously.

    For the v0.1 release we use the torch SVD to obtain ``U, S, V`` then
    launch a Triton kernel that evaluates the per-alpha scaling + final
    matmul fused. SVD itself is a thin (r x r) reduction on the stimulus
    axis so keeping it on torch is not the bottleneck.
    """
    from cortexlab.gpu._ridge_triton import fused_svd_ridge_grid  # lazy

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    UtY = (U.T @ Y).contiguous()  # (r, v)
    alphas_t = torch.tensor(alphas, dtype=X.dtype, device=X.device)
    return fused_svd_ridge_grid(S, Vh, UtY, alphas_t)


def _solve_ridge_single_alpha_triton(
    X: torch.Tensor,
    Y: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    # Single-alpha refit is already compute-bound on the (p,p) Cholesky which
    # cuSOLVER handles well; no Triton win expected. Delegate to torch.
    return _solve_ridge_single_alpha(X, Y, alpha, backend="torch")
