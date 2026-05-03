# Contributing to CortexLab

Thanks for your interest in contributing to CortexLab! This guide will help you get started.

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/cortexlab.git
cd cortexlab
```

### 2. Set up the development environment

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,plotting]"
```

The `[plotting]` extras include `nilearn` and `pyvista`, which most
viz tests need; `[dev]` ships pytest, ruff, and coverage. If you're
only touching analysis code, `[dev,analysis]` is enough.

### 3. Verify everything works

```bash
pytest tests/ -v          # 280 tests, 3 CUDA-gated (skipped on CPU)
ruff check src/ scripts/ tests/
```

## Development Workflow

1. **Create a branch** from `master` for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** - keep commits focused and atomic.

3. **Run tests and lint** before committing:
   ```bash
   pytest tests/ -v
   ruff check src/ scripts/ tests/
   ```

4. **Open a pull request** with a clear description of what you changed and why.

## Project Structure

```
src/cortexlab/
  core/          Model architecture, attention extraction, subject adaptation
  data/          Dataset loading, parcellations, HCP ROI utilities, fMRI studies
  features/      Foundation-model feature extractors (vision + text)
  gpu/           Voxelwise ridge encoder (torch + Triton backends)
  training/      PyTorch Lightning training pipeline
  inference/     Predictor, streaming, modality attribution
  analysis/      Brain-alignment benchmark, causal lesion, noise ceiling,
                 BH-FDR, cognitive load, temporal dynamics, ROI connectivity
  viz/           Cortical surface renderer (matplotlib / plotly+WebGL /
                 pyvista+VTK), brain-region visualization
experiments/     Lesion orchestrator, feature cache builder, alignment comparison
scripts/         Cortical surface plotting + animation, post-processing tools
tests/           Unit tests (pytest); 280 tests, 3 CUDA-gated
notebooks/       Tutorial notebooks
```

## What to Work On

- Check [issues labeled `good first issue`](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for beginner-friendly tasks
- Check [issues labeled `help wanted`](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) for tasks where we need help
- Look at the [experiment issues](../../issues?q=is%3Aissue+is%3Aopen+label%3Aexperiment) if you want to run evaluations

## Code Style

- **Linting**: We use [ruff](https://docs.astral.sh/ruff/) with a 100-character line limit
- **Tests**: Write pytest tests for new functionality. Use synthetic data (no real fMRI needed)
- **Docstrings**: Use NumPy-style docstrings for public functions
- **Imports**: Let ruff sort imports automatically (`ruff check --fix`)

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality before commits. These hooks automatically run linting and formatting checks before each commit.

### Setup

Install `pre-commit` and wire up the git hook:

```bash
pip install pre-commit
pre-commit install
```

After `pre-commit install`, the hooks (ruff, end-of-file-fixer, trailing-whitespace, check-yaml, check-added-large-files) run automatically on every `git commit`. To run them on the whole repo manually (without committing):

```bash
pre-commit run --all-files
```

## Writing Tests

Tests use synthetic data and mock objects so you don't need real fMRI datasets or GPU access:

```python
import torch
from neuralset.dataloader import SegmentData
import neuralset.segments as seg

# Create dummy segments matching batch size
segments = [seg.Segment(start=float(i), duration=1.0, timeline="test") for i in range(batch_size)]

# Create synthetic batch
data = {"text": torch.randn(batch_size, 2, 32, seq_len), "subject_id": torch.zeros(batch_size, dtype=torch.long)}
batch = SegmentData(data=data, segments=segments)
```

## Adding New Features

If you're adding a new analysis method or inference capability:

1. Add the implementation in the appropriate subpackage
2. Export it from the subpackage's `__init__.py`
3. Write tests in `tests/test_yourfeature.py`
4. Add a usage example in the README, `notebooks/`, or `scripts/`
5. Add a one-line entry to `CHANGELOG.md` under `[Unreleased]`

## Reporting Bugs

When filing a bug report, please include:
- Python version and OS
- PyTorch version
- Steps to reproduce
- Full error traceback
- What you expected to happen

## Questions?

Open an issue with the `question` label and we'll help out.
