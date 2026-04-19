#!/usr/bin/env bash
# Shared environment setup for every Slurm job in this directory.
# Idempotent, safe to `source` from compute nodes and login nodes.
#
# Customize the CLUSTER-SPECIFIC section at the top for your site. The
# rest is portable across any Slurm + H100/H200 cluster.

set -euo pipefail

# -------------------- CLUSTER-SPECIFIC (edit me) -------------------------- #
# Module loads. Comment out lines your cluster doesn't provide.
# Stevens Jarvis example; confirm names with `module avail`.
if command -v module &>/dev/null; then
  module load python/3.11 || true
  module load cuda/12.4   || true
  module load gcc/11      || true
fi
# -------------------------------------------------------------------------- #

# -------------------- PATHS ----------------------------------------------- #
# Override any of these by exporting before sourcing this script.
export CORTEXLAB_HOME="${CORTEXLAB_HOME:-$HOME/cortexlab}"
export CORTEXLAB_SCRATCH="${CORTEXLAB_SCRATCH:-/scratch/${USER}/cortexlab}"
export CORTEXLAB_DATA="${CORTEXLAB_DATA:-${CORTEXLAB_SCRATCH}/data}"
export CORTEXLAB_RESULTS="${CORTEXLAB_RESULTS:-${CORTEXLAB_SCRATCH}/results}"
export HF_HOME="${HF_HOME:-${CORTEXLAB_SCRATCH}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

mkdir -p "$CORTEXLAB_SCRATCH" "$CORTEXLAB_DATA" "$CORTEXLAB_RESULTS" \
         "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
# -------------------------------------------------------------------------- #

# -------------------- VIRTUALENV ------------------------------------------ #
VENV="${CORTEXLAB_HOME}/.venv"
if [[ ! -d "$VENV" ]]; then
  echo "[env_setup] creating venv at $VENV"
  python -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Install cortexlab in editable mode if it's missing or the install is stale.
if ! python -c "import cortexlab" 2>/dev/null; then
  echo "[env_setup] installing cortexlab (first run)"
  pip install --upgrade pip
  pip install -e "${CORTEXLAB_HOME}[analysis,dev]"
  # Optional extras used only on cluster runs.
  pip install triton scikit-learn pyyaml 'transformers>=4.44'
fi
# -------------------------------------------------------------------------- #

# -------------------- DIAGNOSTICS ----------------------------------------- #
echo "======================= cortexlab / slurm env ======================"
echo "host:           $(hostname)"
echo "date:           $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "job_id:         ${SLURM_JOB_ID:-interactive}"
echo "array_task:     ${SLURM_ARRAY_TASK_ID:--}"
echo "python:         $(which python) [$(python --version 2>&1)]"
echo "CUDA_VISIBLE:   ${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version \
             --format=csv,noheader | sed 's/^/  gpu[&]/' || true
fi
python - <<'PYEOF'
import torch, cortexlab
print(f"  torch:        {torch.__version__}  cuda={torch.cuda.is_available()}"
      f"  n_gpu={torch.cuda.device_count()}")
print(f"  cortexlab:    {cortexlab.__file__}")
PYEOF
echo "CORTEXLAB_HOME:     $CORTEXLAB_HOME"
echo "CORTEXLAB_SCRATCH:  $CORTEXLAB_SCRATCH"
echo "CORTEXLAB_RESULTS:  $CORTEXLAB_RESULTS"
echo "HF_HOME:            $HF_HOME"
echo "===================================================================="
