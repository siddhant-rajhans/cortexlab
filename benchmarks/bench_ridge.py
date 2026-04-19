"""Benchmark ``VoxelRidgeEncoder`` across backends.

Runs four configurations on matched synthetic data shaped like the
BOLD Moments + TRIBE v2 regression problem, reports wall times,
throughput, and speedups, and writes a JSON record.

Usage
-----

  python -m benchmarks.bench_ridge \
      --n-stim 1000 --n-features 512 --n-voxels 200000 \
      --alphas 0.01,1,100,10000,1000000 --cv 5 \
      --output benchmarks/results/ridge_$(hostname).json

Add ``--backend torch,triton`` etc. to restrict which backends run.
Without ``--backend`` the script probes every backend available on the
current machine (sklearn on CPU, torch on CPU and CUDA, triton on CUDA).

The numbers that appear in the class-project slides come from running
this on one Stevens Jarvis node.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-stim", type=int, default=1000)
    ap.add_argument("--n-features", type=int, default=512)
    ap.add_argument("--n-voxels", type=int, default=50_000)
    ap.add_argument("--alphas", type=str,
                    default="0.01,1,100,10000,1000000")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument(
        "--backend", type=str, default="all",
        help="comma-sep subset of {sklearn, torch-cpu, torch-cuda, triton}",
    )
    ap.add_argument("--warmup", type=int, default=1,
                    help="warmup iterations before timing")
    ap.add_argument("--repeats", type=int, default=1,
                    help="timed iterations per backend; reports min")
    return ap.parse_args()


def _make_data(n: int, p: int, v: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    true_beta = rng.standard_normal((p, v)).astype(np.float32) * 0.3
    Y = X @ true_beta + 0.5 * rng.standard_normal((n, v)).astype(np.float32)
    return X, Y


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_torch(X, Y, alphas, cv, device_str, warmup, repeats, backend_name):
    from cortexlab.gpu.ridge import VoxelRidgeEncoder
    device = torch.device(device_str)
    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)
    times = []
    for i in range(warmup + repeats):
        _sync(device)
        t0 = time.perf_counter()
        enc = VoxelRidgeEncoder(
            alphas=alphas, cv=cv, backend=backend_name, device=device_str,
        ).fit(X_t, Y_t)
        _sync(device)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)
        del enc
    return min(times)


def _time_sklearn(X, Y, alphas, cv, warmup, repeats):
    from sklearn.linear_model import RidgeCV
    # RidgeCV with GridSearch over alphas is single-output; for our
    # multi-output Y we run it column-by-column, which is sklearn's
    # standard multi-output idiom for voxelwise encoding.
    times = []
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        for vi in range(Y.shape[1]):
            RidgeCV(alphas=alphas, fit_intercept=True, cv=cv).fit(X, Y[:, vi])
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)
    return min(times)


def _select(backends_arg: str, cuda_available: bool) -> list[str]:
    if backends_arg == "all":
        picks = ["sklearn", "torch-cpu"]
        if cuda_available:
            picks.extend(["torch-cuda", "triton"])
        return picks
    return [b.strip() for b in backends_arg.split(",") if b.strip()]


def main() -> None:
    args = _parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    X, Y = _make_data(args.n_stim, args.n_features, args.n_voxels, args.seed)
    cuda_ok = torch.cuda.is_available()
    backends = _select(args.backend, cuda_ok)
    print(f"Problem: n={args.n_stim}, p={args.n_features}, v={args.n_voxels:,}, "
          f"alphas={len(alphas)}, cv={args.cv}")
    print(f"Host: {platform.node()} / {platform.platform()}")
    print(f"CUDA: {'yes, ' + torch.cuda.get_device_name(0) if cuda_ok else 'no'}")
    print(f"Backends: {', '.join(backends)}")
    print()

    records = []
    baseline = None

    for backend in backends:
        label = backend
        try:
            if backend == "sklearn":
                t = _time_sklearn(X, Y, alphas, args.cv, args.warmup, args.repeats)
            elif backend == "torch-cpu":
                t = _time_torch(X, Y, alphas, args.cv, "cpu", args.warmup,
                                args.repeats, "torch")
            elif backend == "torch-cuda":
                if not cuda_ok:
                    print(f"  {label:<12}  SKIP (no CUDA)")
                    continue
                t = _time_torch(X, Y, alphas, args.cv, "cuda", args.warmup,
                                args.repeats, "torch")
            elif backend == "triton":
                if not cuda_ok:
                    print(f"  {label:<12}  SKIP (no CUDA)")
                    continue
                t = _time_torch(X, Y, alphas, args.cv, "cuda", args.warmup,
                                args.repeats, "triton")
            else:
                raise ValueError(f"unknown backend {backend}")
        except Exception as e:
            print(f"  {label:<12}  FAILED: {e}")
            continue

        if baseline is None and backend in {"sklearn", "torch-cpu"}:
            baseline = t
        speedup = (baseline / t) if baseline else None
        throughput = args.n_voxels / t
        records.append({
            "backend": backend, "seconds": t,
            "speedup_vs_baseline": speedup,
            "voxels_per_sec": throughput,
        })
        print(f"  {label:<12}  {t:>8.2f} s  |  {throughput:>10,.0f} voxels/s"
              f"  |  speedup {speedup:>5.1f}x" if speedup else
              f"  {label:<12}  {t:>8.2f} s  |  {throughput:>10,.0f} voxels/s")

    out = {
        "config": {
            "n_stim": args.n_stim, "n_features": args.n_features,
            "n_voxels": args.n_voxels, "alphas": alphas, "cv": args.cv,
            "seed": args.seed,
        },
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "cuda": torch.cuda.get_device_name(0) if cuda_ok else None,
        },
        "baseline_seconds": baseline,
        "results": records,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
