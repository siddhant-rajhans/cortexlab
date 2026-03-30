"""Edge case tests across all modules."""

import numpy as np
import pytest
import torch

from tests.conftest import FakeModel, make_segments


class TestBrainAlignmentEdgeCases:
    def test_nan_input(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat = np.full((10, 32), np.nan)
        brain_pred = np.random.randn(10, 50)
        bench = BrainAlignmentBenchmark(brain_pred)
        result = bench.score_model(model_feat, method="rsa")
        # Should not crash; score may be NaN or 0
        assert isinstance(result.aggregate_score, float)

    def test_inf_input(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat = np.random.randn(10, 32)
        model_feat[0, 0] = np.inf
        brain_pred = np.random.randn(10, 50)
        bench = BrainAlignmentBenchmark(brain_pred)
        result = bench.score_model(model_feat, method="cka")
        assert isinstance(result.aggregate_score, float)

    def test_single_stimulus(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat = np.random.randn(1, 32)
        brain_pred = np.random.randn(1, 50)
        bench = BrainAlignmentBenchmark(brain_pred)
        result = bench.score_model(model_feat, method="rsa")
        assert isinstance(result.aggregate_score, float)

    def test_empty_roi_indices(self):
        from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

        model_feat = np.random.randn(10, 32)
        brain_pred = np.random.randn(10, 50)
        bench = BrainAlignmentBenchmark(brain_pred, roi_indices={})
        result = bench.score_model(model_feat, method="rsa")
        assert len(result.roi_scores) == 0


class TestCognitiveLoadEdgeCases:
    def test_zero_activation(self):
        from cortexlab.analysis.cognitive_load import CognitiveLoadScorer

        roi_indices = {"V1": np.array([0, 1, 2]), "A1": np.array([3, 4])}
        scorer = CognitiveLoadScorer(roi_indices)
        predictions = np.zeros((10, 10))
        result = scorer.score_predictions(predictions)
        assert result.overall_load >= 0.0
        assert not np.isnan(result.overall_load)

    def test_single_vertex(self):
        from cortexlab.analysis.cognitive_load import CognitiveLoadScorer

        roi_indices = {"V1": np.array([0])}
        scorer = CognitiveLoadScorer(roi_indices)
        predictions = np.random.randn(5, 5)
        result = scorer.score_predictions(predictions)
        assert isinstance(result.overall_load, float)


class TestModelEdgeCases:
    def test_single_sample_batch(self):
        from neuralset.dataloader import SegmentData
        from neuraltrain.models.transformer import TransformerEncoder

        from cortexlab.core.model import FmriEncoder

        config = FmriEncoder(
            hidden=256, encoder=TransformerEncoder(depth=2, heads=4)
        )
        modalities = {"text": (2, 32)}
        model = config.build(feature_dims=modalities, n_outputs=50, n_output_timesteps=5)
        data = {
            "text": torch.randn(1, 2, 32, 10),
            "subject_id": torch.zeros(1, dtype=torch.long),
        }
        batch = SegmentData(data=data, segments=make_segments(1))
        out = model(batch)
        assert out.shape[0] == 1

    def test_all_modalities_missing(self):
        from neuralset.dataloader import SegmentData
        from neuraltrain.models.transformer import TransformerEncoder

        from cortexlab.core.model import FmriEncoder

        config = FmriEncoder(
            hidden=256, encoder=TransformerEncoder(depth=2, heads=4)
        )
        modalities = {"text": (2, 32), "audio": (2, 32)}
        model = config.build(feature_dims=modalities, n_outputs=50, n_output_timesteps=5)
        # Provide neither text nor audio
        data = {
            "subject_id": torch.zeros(2, dtype=torch.long),
            "dummy": torch.randn(2, 2, 32, 10),  # unrecognized modality
        }
        batch = SegmentData(data=data, segments=make_segments(2))
        out = model(batch)
        assert out.shape == (2, 50, 5)


class TestStreamingEdgeCases:
    def test_reset_on_empty_buffer(self):
        from unittest.mock import MagicMock

        from cortexlab.inference.streaming import StreamingPredictor

        model = MagicMock()
        model.feature_dims = {"text": (2, 32)}
        sp = StreamingPredictor(model, window_trs=5, step_trs=1)
        sp.reset()  # Should not error
        assert len(sp._buffer) == 0

    def test_flush_on_empty_buffer(self):
        from unittest.mock import MagicMock

        from cortexlab.inference.streaming import StreamingPredictor

        model = MagicMock()
        model.feature_dims = {"text": (2, 32)}
        sp = StreamingPredictor(model, window_trs=5, step_trs=1)
        results = sp.flush()
        assert results == []


class TestAttributionEdgeCases:
    def test_single_modality_normalized(self):
        from neuralset.dataloader import SegmentData

        from cortexlab.inference.attribution import ModalityAttributor

        class SingleModalModel:
            feature_dims = {"text": (2, 32)}
            def eval(self): pass
            def __call__(self, batch, **kwargs):
                return torch.ones(2, 50, 10)

        model = SingleModalModel()
        attributor = ModalityAttributor(model)
        data = {
            "text": torch.randn(2, 2, 32, 20),
            "subject_id": torch.zeros(2, dtype=torch.long),
        }
        batch = SegmentData(data=data, segments=make_segments(2))
        scores = attributor.attribute(batch)
        # With only one modality, normalized score should be 1.0 everywhere
        # (or all zeros if ablation produces no difference, which normalises to 1.0)
        assert scores["text_normalised"].shape == (50,)
        assert np.all(np.isfinite(scores["text_normalised"]))
