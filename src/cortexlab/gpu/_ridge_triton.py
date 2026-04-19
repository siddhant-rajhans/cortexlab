"""Triton kernel for the fused ``V diag(s/(s^2+alpha)) U^T Y`` ridge step.

Only imported when Triton is installed (see
:func:`cortexlab.gpu.ridge._triton_available`). The CPU/torch backend in
:mod:`cortexlab.gpu.ridge` is the correctness reference; this module exists
to provide the speedup path on NVIDIA GPUs.

Math recap. With ``X = U S V^T`` and ``M = U^T Y`` precomputed (both
comparatively cheap), the solution for regularization strength ``a`` is

    beta(a) = V @ diag(s / (s^2 + a)) @ M                  (1)

Equivalently, with ``Z(a) = diag(s / (s^2 + a)) @ M``,

    beta(a)_{p, v} = sum_r V^T_{p, r} * Z(a)_{r, v}        (2)

The kernel below tiles along ``p`` (output features) and ``v`` (voxels),
with each thread block computing one ``(BLOCK_P, BLOCK_V)`` output tile
per alpha. Reduction is along ``r`` (SVD rank, typically ~= n_samples).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Triton is required for this module. Install with `pip install triton`."
    ) from e


@triton.jit
def _fused_ridge_grid_kernel(
    S_ptr, V_ptr, M_ptr, A_ptr, OUT_ptr,
    r, p, v, n_alpha,
    stride_v_r, stride_v_p,
    stride_m_r, stride_m_v,
    stride_o_a, stride_o_p, stride_o_v,
    BLOCK_P: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid_a = tl.program_id(0)          # alpha index
    pid_p = tl.program_id(1)          # tile over output rows (features)
    pid_v = tl.program_id(2)          # tile over voxels

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_p = offs_p < p
    mask_v = offs_v < v

    alpha = tl.load(A_ptr + pid_a)

    acc = tl.zeros((BLOCK_P, BLOCK_V), dtype=tl.float32)

    for rr in range(0, r, BLOCK_R):
        offs_r = rr + tl.arange(0, BLOCK_R)
        mask_r = offs_r < r

        # s / (s^2 + alpha) -> (BLOCK_R,)
        s = tl.load(S_ptr + offs_r, mask=mask_r, other=0.0)
        scale = s / (s * s + alpha)

        # V[:, offs_r] -> (BLOCK_P, BLOCK_R). V is stored (r, p); V^T is (p, r).
        v_ptr = V_ptr + offs_p[:, None] * stride_v_p + offs_r[None, :] * stride_v_r
        V_tile = tl.load(v_ptr, mask=mask_p[:, None] & mask_r[None, :], other=0.0)

        # M[offs_r, offs_v] -> (BLOCK_R, BLOCK_V)
        m_ptr = M_ptr + offs_r[:, None] * stride_m_r + offs_v[None, :] * stride_m_v
        M_tile = tl.load(m_ptr, mask=mask_r[:, None] & mask_v[None, :], other=0.0)

        # Scaled multiply and accumulate: V * scale[None, :] @ M
        M_scaled = M_tile * scale[:, None]
        acc += tl.dot(V_tile, M_scaled)

    out_ptr = (
        OUT_ptr
        + pid_a * stride_o_a
        + offs_p[:, None] * stride_o_p
        + offs_v[None, :] * stride_o_v
    )
    tl.store(out_ptr, acc, mask=mask_p[:, None] & mask_v[None, :])


def fused_svd_ridge_grid(
    S: torch.Tensor,   # (r,)
    Vh: torch.Tensor,  # (r, p). note: already V^T-like row layout
    M: torch.Tensor,   # (r, v) = U^T @ Y
    alphas: torch.Tensor,  # (n_alpha,)
) -> torch.Tensor:
    """Evaluate equation (1) for every alpha in the grid.

    Returns a tensor of shape ``(n_alpha, p, v)``.

    Implementation notes:

    * ``Vh`` is the SVD's ``V^T``. Indexing ``Vh[r_idx, p_idx]`` corresponds
      to ``V[p_idx, r_idx]`` in the math. The kernel reads with stride
      ``(stride_v_r, stride_v_p)`` so it computes ``V @ ...`` correctly.
    * Block sizes are conservative defaults. For production use autotune
      once hardware is known.
    """
    assert S.ndim == 1 and Vh.ndim == 2 and M.ndim == 2 and alphas.ndim == 1
    r = S.shape[0]
    assert Vh.shape[0] == r, f"Vh.shape[0] ({Vh.shape[0]}) must equal r ({r})"
    assert M.shape[0] == r, f"M.shape[0] ({M.shape[0]}) must equal r ({r})"
    p = Vh.shape[1]
    v = M.shape[1]
    n_alpha = alphas.shape[0]

    out = torch.empty(n_alpha, p, v, dtype=S.dtype, device=S.device)

    BLOCK_P = 32
    BLOCK_V = 64
    BLOCK_R = min(32, triton.next_power_of_2(r)) if r > 1 else 1
    BLOCK_R = max(BLOCK_R, 1)

    grid = (
        n_alpha,
        triton.cdiv(p, BLOCK_P),
        triton.cdiv(v, BLOCK_V),
    )

    _fused_ridge_grid_kernel[grid](
        S, Vh, M, alphas, out,
        r, p, v, n_alpha,
        Vh.stride(0), Vh.stride(1),
        M.stride(0), M.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_P=BLOCK_P, BLOCK_V=BLOCK_V, BLOCK_R=BLOCK_R,
    )
    return out
