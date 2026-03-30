"""Integration tests spanning multiple modules."""

import numpy as np
import pytest
import torch
from neuralset.dataloader import SegmentData
from neuraltrain.models.transformer import TransformerEncoder

from tests.conftest import make_segments


def _build_model_and_predict(batch_size=2, seq_len=20, n_vertices=50):
    """Helper: build model, run forward, return predictions as numpy."""
    from cortexlab.core.model import FmriEncoder

    config = FmriEncoder(
        hidden=256, encoder=TransformerEncoder(depth=2, heads=4)
    )
    modalities = {"text": (2, 32), "audio": (2, 32)}
    model = config.build(feature_dims=modalities, n_outputs=n_vertices, n_output_timesteps=seq_len)
    model.eval()

    data = {
        "text": torch.randn(batch_size, 2, 32, seq_len),
        "audio": torch.randn(batch_size, 2, 32, seq_len),
        "subject_id": torch.zeros(batch_size, dtype=torch.long),
    }
    batch = SegmentData(data=data, segments=make_segments(batch_size))

    with torch.inference_mode():
        out = model(batch)  # (B, V, T)

    # Reshape to (T, V) by taking first batch and transposing
    preds = out[0].cpu().numpy().T  # (T, V)
    return model, batch, preds


class TestModelToAlignment:
    def test_full_pipeline_rsa(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        _, _, preds = _build_model_and_predict()
        model_features = np.random.randn(preds.shape[0], 64)

        bench = BrainAlignmentBenchmark(preds)
        result = bench.score_model(model_features, method="rsa")
        assert isinstance(result.aggregate_score, float)

    def test_full_pipeline_all_methods(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        _, _, preds = _build_model_and_predict()
        model_features = np.random.randn(preds.shape[0], 64)
        bench = BrainAlignmentBenchmark(preds)

        for method in ["rsa", "cka", "procrustes"]:
            result = bench.score_model(model_features, method=method)
            assert isinstance(result.aggregate_score, float)

    def test_pipeline_with_permutation_test(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        _, _, preds = _build_model_and_predict()
        bench = BrainAlignmentBenchmark(preds)

        score, p = bench.permutation_test(preds, method="cka", n_permutations=20, seed=42)
        assert isinstance(score, float)
        assert 0.0 <= p <= 1.0


class TestModelToCognitiveLoad:
    def test_full_pipeline(self, roi_indices):
        from cortexlab.analysis.cognitive_load import CognitiveLoadScorer

        _, _, preds = _build_model_and_predict()
        scorer = CognitiveLoadScorer(roi_indices)
        result = scorer.score_predictions(preds)

        assert 0.0 <= result.overall_load <= 1.0
        assert len(result.timeline) == preds.shape[0]


class TestModelToTemporalDynamics:
    def test_full_pipeline(self, roi_indices):
        from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer

        _, _, preds = _build_model_and_predict()
        analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=1.0)
        result = analyzer.analyze(preds)

        assert len(result.peak_latencies) == len(roi_indices)
        assert len(result.sustained_components) == len(roi_indices)


class TestModelToConnectivity:
    def test_full_pipeline(self, roi_indices):
        from cortexlab.analysis.connectivity import ROIConnectivityAnalyzer

        _, _, preds = _build_model_and_predict()
        analyzer = ROIConnectivityAnalyzer(roi_indices)
        result = analyzer.analyze(preds, n_clusters=3)

        assert result.correlation_matrix.shape[0] == len(roi_indices)
        assert len(result.clusters) > 0


class TestModelToAttribution:
    def test_full_pipeline(self):
        from cortexlab.inference.attribution import ModalityAttributor

        model, batch, _ = _build_model_and_predict()
        attributor = ModalityAttributor(model)
        scores = attributor.attribute(batch)

        assert "text" in scores
        assert "audio" in scores
