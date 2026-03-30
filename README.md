# CortexLab

Enhanced multimodal fMRI brain encoding toolkit built on [Meta's TRIBE v2](https://github.com/facebookresearch/tribev2).

CortexLab extends TRIBE v2 with streaming inference, interpretability tools, cross-subject adaptation, brain-alignment benchmarking, and cognitive load scoring.

## Features

| Feature | Description |
|---|---|
| **Streaming Inference** | Sliding-window real-time predictions from live feature streams |
| **ROI Attention Maps** | Visualize which brain regions attend to which temporal moments |
| **Modality Attribution** | Per-vertex importance scores for text, audio, and video |
| **Cross-Subject Adaptation** | Ridge regression or nearest-neighbour adaptation for new subjects |
| **Brain-Alignment Benchmark** | Score how "brain-like" any AI model's representations are (RSA, CKA, Procrustes) |
| **Cognitive Load Scorer** | Predict cognitive demand of media from predicted brain activation patterns |

## Installation

```bash
pip install -e "."

# With optional dependencies
pip install -e ".[plotting]"       # Brain visualization
pip install -e ".[training]"       # PyTorch Lightning training
pip install -e ".[analysis]"       # RSA/CKA benchmarking (scipy)
pip install -e ".[dev]"            # Testing and linting
```

## Quick Start

### Inference

```python
from cortexlab.inference.predictor import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", device="auto")
events = model.get_events_dataframe(video_path="clip.mp4")
preds, segments = model.predict(events)
```

### Brain-Alignment Benchmark

```python
from cortexlab.analysis import BrainAlignmentBenchmark

bench = BrainAlignmentBenchmark(brain_predictions, roi_indices=roi_indices)
result = bench.score_model(clip_features, method="rsa")
print(f"Alignment: {result.aggregate_score:.3f}")
print(f"V1 alignment: {result.roi_scores['V1']:.3f}")
```

### Cognitive Load Scoring

```python
from cortexlab.analysis import CognitiveLoadScorer

scorer = CognitiveLoadScorer(roi_indices)
result = scorer.score_predictions(predictions)
print(f"Overall load: {result.overall_load:.2f}")
print(f"Visual complexity: {result.visual_complexity:.2f}")
print(f"Language processing: {result.language_processing:.2f}")
```

### Streaming Inference

```python
from cortexlab.inference import StreamingPredictor

sp = StreamingPredictor(model._model, window_trs=40, step_trs=1, device="cuda")
for features in live_feature_stream():
    pred = sp.push_frame(features)
    if pred is not None:
        visualize(pred)  # (n_vertices,)
```

### Modality Attribution

```python
from cortexlab.inference import ModalityAttributor

attributor = ModalityAttributor(model._model, roi_indices=roi_indices)
scores = attributor.attribute(batch)
# scores["text"], scores["audio"], scores["video"] -> (n_vertices,)
```

### Cross-Subject Adaptation

```python
from cortexlab.core.subject import SubjectAdapter

adapter = SubjectAdapter.from_ridge(model._model, calibration_loader, regularization=1e-3)
new_subject_id = adapter.inject_into_model(model._model)
```

## Architecture

```
src/cortexlab/
  core/          Model architecture, attention extraction, subject adaptation
  data/          Dataset loading, transforms, HCP ROI utilities
  training/      PyTorch Lightning training pipeline
  inference/     Predictor, streaming, modality attribution
  analysis/      Brain-alignment benchmark, cognitive load scorer
  viz/           Brain surface visualization (nilearn, pyvista)
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

## License

CC BY-NC 4.0 (inherited from TRIBE v2). See [LICENSE](LICENSE).

## Acknowledgements

Built on [TRIBE v2](https://github.com/facebookresearch/tribev2) by Meta FAIR.
