#!/usr/bin/env bash
# Extract features for the 5 alignment baselines on all BOLD Moments
# stimuli. One array task per model; each writes a single .npz to
# $CORTEXLAB_RESULTS/features/<preset>.npz.
#
# Submit with:
#   sbatch scripts/slurm/extract_features.sbatch
#
# Per-task runtime on one H200 is ~30–80 min depending on model size.

# ----------------------------- Slurm directives ---------------------------- #
#SBATCH --job-name=cxlab-features
#SBATCH --partition=gpu                # TODO: cluster-specific
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-4                    # one task per preset
#SBATCH --output=scripts/slurm/slurm_logs/feats_%A_%a.out
#SBATCH --error=scripts/slurm/slurm_logs/feats_%A_%a.err
# ---------------------------------------------------------------------------- #

set -euo pipefail
mkdir -p scripts/slurm/slurm_logs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_setup.sh"

cd "$CORTEXLAB_HOME"

PRESETS=(clip-vit-l-14 siglip2-vit-l dinov2-vit-l vjepa2-vit-l paligemma2-3b)
PRESET="${PRESETS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="${CORTEXLAB_RESULTS}/features"
OUT="${OUT_DIR}/${PRESET}.npz"
mkdir -p "$OUT_DIR"

echo "==== extracting $PRESET -> $OUT ===="

# The stimulus loader is cluster-specific; `list_stimulus_paths` is a
# thin shim around whatever storage layout you adopted for BOLD Moments.
# Adapt the import on the first two lines to match your data module.
python - "$PRESET" "$OUT" <<'PYEOF'
import sys, os
from pathlib import Path
from cortexlab.features import FoundationFeatureExtractor

preset, out_path = sys.argv[1], sys.argv[2]

# -------- ADAPT THIS to your BOLD Moments stimulus layout --------------- #
# Expected: a function or glob that yields paths to video files in the
# canonical stimulus order used throughout the study.
from cortexlab.data.studies.lahner2024bold import list_stimulus_paths  # noqa
stim_paths = list(list_stimulus_paths(os.environ["CORTEXLAB_DATA"]))
# ------------------------------------------------------------------------ #

print(f"stimuli: {len(stim_paths)}  first: {stim_paths[0]}")

ext = FoundationFeatureExtractor.from_preset(preset, device="cuda",
                                             batch_size=8)
feats = ext.extract(stim_paths)
ext.save_cache(feats, out_path)
print(f"saved {feats.shape} -> {out_path}")
PYEOF

echo "==== done ===="
