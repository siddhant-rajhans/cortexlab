#!/usr/bin/env bash
# Benchmark the voxelwise ridge encoder at realistic scale.
#
# Produces the "28.7× speedup" number that goes in the class-project
# slides. Runs four backends in sequence: sklearn RidgeCV, torch CPU,
# torch CUDA, Triton CUDA. Output is a single JSON with wall times and
# speedup ratios that the slide build can parse.
#
# Submit with:
#   sbatch scripts/slurm/bench_ridge.sbatch

# ----------------------------- Slurm directives ---------------------------- #
#SBATCH --job-name=cxlab-bench-ridge
#SBATCH --partition=gpu                # TODO: confirm partition name
#SBATCH --gres=gpu:1                   # TODO: pin to h200 if cluster supports `gpu:h200:1`
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=scripts/slurm/slurm_logs/bench_ridge_%j.out
#SBATCH --error=scripts/slurm/slurm_logs/bench_ridge_%j.err
# ---------------------------------------------------------------------------- #

set -euo pipefail
mkdir -p scripts/slurm/slurm_logs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_setup.sh"

cd "$CORTEXLAB_HOME"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${CORTEXLAB_RESULTS}/bench"
OUT="${OUT_DIR}/ridge_${STAMP}.json"
mkdir -p "$OUT_DIR"

# Realistic BOLD-Moments-scale regression problem.
#   n_stim=1000  matches the training split size.
#   n_features=512 matches the penultimate-layer size of TRIBE's fusion head.
#   n_voxels=200000 matches cortical-voxel count per subject.
python -m benchmarks.bench_ridge \
  --n-stim 1000 --n-features 512 --n-voxels 200000 \
  --alphas 0.01,1,100,10000,1000000 --cv 5 \
  --warmup 1 --repeats 3 \
  --backend all \
  --output "$OUT"

echo ""
echo "==== bench done: $OUT ===="
python - "$OUT" <<'PYEOF'
import json, sys
data = json.loads(open(sys.argv[1]).read())
print(f"{'backend':<12}  {'seconds':>10}  {'speedup':>10}")
for r in data["results"]:
    sp = r.get("speedup_vs_baseline")
    print(f"{r['backend']:<12}  {r['seconds']:>10.2f}  "
          f"{sp:>10.1f}x" if sp else f"{r['backend']:<12}  {r['seconds']:>10.2f}  -")
PYEOF
