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
pip install -e ".[dev,analysis]"
```

### 3. Verify everything works

```bash
pytest tests/ -v
ruff check src/ tests/
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
   ruff check src/ tests/
   ```

4. **Open a pull request** with a clear description of what you changed and why.

## Project Structure

```
src/cortexlab/
  core/          Model architecture, attention extraction, subject adaptation
  data/          Dataset loading, transforms, HCP ROI utilities
  training/      PyTorch Lightning training pipeline
  inference/     Predictor, streaming, modality attribution
  analysis/      Brain-alignment benchmark, cognitive load scorer
  viz/           Brain surface visualization
tests/           Unit tests (pytest)
examples/        Usage examples
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
4. Add a usage example in the README or `examples/`

## Reporting Bugs

When filing a bug report, please include:
- Python version and OS
- PyTorch version
- Steps to reproduce
- Full error traceback
- What you expected to happen

## Questions?

Open an issue with the `question` label and we'll help out.
