#!/usr/bin/env bash
# 10-minute integration test. Run this BEFORE submitting long jobs.
#
# Catches env problems (CUDA drivers, Triton compile, HF cache perms,
# import errors) on a real GPU but at a scale small enough to finish
# inside a short interactive allocation.
#
# Usage (interactive):
#   srun --gres=gpu:1 --time=15 --cpus-per-task=4 --mem=32G --pty \
#        bash scripts/slurm/smoke_test.sh
#
# Usage (login node, if you have CUDA accessible):
#   bash scripts/slurm/smoke_test.sh
#
# Exit 0 = safe to submit long jobs. Any non-zero = fix before wasting
# a full 2× H200 allocation on a pipeline with a typo.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_setup.sh"

STAMP=$(date +%s)
OUT="${CORTEXLAB_RESULTS}/smoke_${STAMP}"
mkdir -p "$OUT"
echo ""
echo "===== smoke test outputs -> $OUT ====="

cd "$CORTEXLAB_HOME"

# --- step 1: unit tests touching the critical modules -----------------------
echo ""
echo ">>> [1/4] unit tests for ridge / lesion / noise-ceiling / features"
pytest -q -x --no-header \
  tests/test_gpu_ridge.py \
  tests/test_lesion.py \
  tests/test_noise_ceiling.py \
  tests/test_features.py

# --- step 2: tiny Triton benchmark -----------------------------------------
# Verifies that the Triton kernel compiles on this cluster's CUDA toolchain
# and that it produces identical numerics to the torch backend.
echo ""
echo ">>> [2/4] ridge benchmark (tiny, verifies Triton compiles)"
python -m benchmarks.bench_ridge \
  --n-stim 200 --n-features 64 --n-voxels 5000 \
  --alphas 0.01,1,100,10000 --cv 3 \
  --warmup 1 --repeats 1 \
  --output "${OUT}/bench_smoke.json"

# --- step 3: mock lesion run -----------------------------------------------
# Exercises the full orchestration path with synthetic data. Completes in
# seconds and validates that the manifest + per-subject artefacts land on
# the expected paths.
echo ""
echo ">>> [3/4] mock lesion run (verifies orchestrator)"
python -m experiments.causal_modality_ablation \
  --mock --subjects 1 2 \
  --alphas 0.01,1,100 --cv 3 \
  --output "${OUT}/lesion_mock"

# --- step 4: output sanity checks ------------------------------------------
echo ""
echo ">>> [4/4] sanity-checking outputs"
python - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])

# Benchmark JSON has at least one backend result.
bench = json.loads((out / "bench_smoke.json").read_text())
backends = [r["backend"] for r in bench["results"]]
assert backends, "no backends ran in the smoke benchmark"
print(f"  bench backends ran: {backends}")
if "triton" in backends:
    print("  triton compile:    OK")

# Lesion manifest has per-subject entries and ground-truth recovery.
lm = json.loads((out / "lesion_mock/manifest.json").read_text())
assert lm["n_subjects"] == 2
rs = lm["results"][0]["roi_summary"]
for roi, expect in (("text_roi", "dR2_text"),
                    ("audio_roi", "dR2_audio"),
                    ("video_roi", "dR2_video")):
    row = rs[roi]
    assert row[expect] == max(row[f"dR2_{m}"] for m in ("text","audio","video")), \
        f"{roi}: expected {expect} to dominate, got {row}"
print("  lesion ground-truth recovery: OK")

print("\n  SMOKE TEST PASSED")
PYEOF

echo ""
echo "===== smoke test OK  ====="
echo "ready to submit long jobs:"
echo "  sbatch scripts/slurm/bench_ridge.sbatch"
echo "  sbatch scripts/slurm/extract_features.sbatch"
echo "  sbatch scripts/slurm/run_lesion.sbatch"
