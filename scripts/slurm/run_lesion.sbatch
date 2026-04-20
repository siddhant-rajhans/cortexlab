#!/usr/bin/env bash
# Run the causal modality lesion study across 10 BOLD Moments subjects.
# Requires: feature extraction already completed (run extract_features.sbatch
# first, or place a feature cache at $CORTEXLAB_RESULTS/features/*.npz).
#
# Submit with:
#   sbatch scripts/slurm/run_lesion.sbatch

# ----------------------------- Slurm directives ---------------------------- #
#SBATCH --job-name=cxlab-lesion
#SBATCH --partition=gpu                # TODO: cluster-specific
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:30:00
#SBATCH --output=scripts/slurm/slurm_logs/lesion_%j.out
#SBATCH --error=scripts/slurm/slurm_logs/lesion_%j.err
# ---------------------------------------------------------------------------- #

set -euo pipefail
mkdir -p scripts/slurm/slurm_logs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_setup.sh"

cd "$CORTEXLAB_HOME"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${CORTEXLAB_RESULTS}/lesion/${STAMP}"
mkdir -p "$OUT_DIR"

echo "==== lesion study -> $OUT_DIR ===="

python -m experiments.causal_modality_ablation \
  --config "${SCRIPT_DIR}/config/lesion_bold_moments.yaml" \
  --subjects 1 2 3 4 5 6 7 8 9 10 \
  --alphas 0.01,1,100,10000,1000000 --cv 5 \
  --mask zero \
  --device cuda --backend triton \
  --output "$OUT_DIR"

echo ""
echo "==== lesion done ===="
python - "$OUT_DIR" <<'PYEOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
m = json.loads((out / "manifest.json").read_text())
print(f"subjects: {m['n_subjects']}   mask: {m['mask_strategy']}")
print(f"group noise ceiling (mean): {m.get('group_ceiling_mean')}")
print(f"{'subject':>8}  {'full_R2':>9}  {'dR2_text':>10}  {'dR2_audio':>11}  {'dR2_video':>11}")
for s in m["results"]:
    d = s["delta_r2_means"]
    print(f"{s['subject_id']:>8d}  {s['full_r2_mean']:>9.3f}  "
          f"{d.get('text', 0):>10.3f}  {d.get('audio', 0):>11.3f}  {d.get('video', 0):>11.3f}")
PYEOF

echo ""
echo "results: $OUT_DIR"
echo "next: import per-ROI numbers into Draft 3 / Final slide decks"
