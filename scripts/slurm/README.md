# Slurm submission scripts for CortexLab

Reproducible jobs for running the voxelwise ridge benchmark, feature
extraction, and the causal modality lesion study on a Slurm cluster
with H100 / H200 GPUs.

## Layout

```
scripts/slurm/
├── env_setup.sh                  source from every job / interactive shell
├── smoke_test.sh                 10-minute pipeline integrity check
├── bench_ridge.sbatch            ridge speedup benchmark (sklearn / torch / triton)
├── extract_features.sbatch       feature extraction for 5 alignment baselines (array)
├── run_lesion.sbatch             causal modality lesion across subjects
├── config/
│   └── lesion_bold_moments.yaml  paths and hyperparameters for the lesion run
└── slurm_logs/                   per-job stdout / stderr (created on first run)
```

## First-time setup

1. Clone the repo into `$HOME/cortexlab` (or export `CORTEXLAB_HOME` to a different path before sourcing).
2. Edit the `CLUSTER-SPECIFIC` block at the top of `env_setup.sh`. Confirm module names with `module avail python cuda`.
3. Mount or stage the BOLD Moments dataset at `$CORTEXLAB_DATA/bold_moments`. Default path is `/scratch/$USER/cortexlab/data/bold_moments`.
4. Authenticate once with HuggingFace so the feature extractors can pull weights:

   ```bash
   huggingface-cli login
   ```

5. If you haven't accepted the LLaMA license yet, do it now. TRIBE v2 requires LLaMA 3.2-3B and will fail to load otherwise.

## Running, in order

```bash
# 1. smoke test (10 min, one interactive GPU). Do this FIRST every run.
srun --gres=gpu:1 --time=15 --cpus-per-task=4 --mem=32G --pty \
     bash scripts/slurm/smoke_test.sh

# 2. ridge benchmark (~1 hr). Produces the headline speedup number.
sbatch scripts/slurm/bench_ridge.sbatch

# 3. feature extraction (~2-3 hr, 5-task array). Caches per-model .npz.
sbatch scripts/slurm/extract_features.sbatch

# 4. lesion study (~1 hr on 1 H200). Per-subject delta R^2 artefacts + manifest.
sbatch scripts/slurm/run_lesion.sbatch
```

Results land in `$CORTEXLAB_RESULTS/{bench,features,lesion}/...`.

## What you'll still need to adapt

- Slurm partition / QoS directives. Every cluster names them differently. Replace `--partition=gpu` with whatever yours calls the H100/H200 pool.
- GRES specifier. On many clusters you can pin to a specific GPU with `--gres=gpu:h200:1`. Check with `sinfo -o "%.10P %G"`.
- `extract_features.sbatch` calls `cortexlab.data.studies.lahner2024bold.list_stimulus_paths(...)`. Make sure the dataset layout matches the helper's expectations.

## After the jobs finish

Drop the numbers into the slide decks that live in your project workspace:

- `bench` goes into the Triton kernel slide.
- `lesion/.../manifest.json` goes into the per-ROI delta R^2 table.
- `lesion/noise_ceiling.npy` goes into the noise-ceiling-normalized R^2 column.
