"""Tests for modality attribution."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.conftest import make_segments


def _make_mock_model(n_vertices=100):
    """Create a mock model for attribution testing."""
    model = MagicMock()
    model.feature_dims = {"text": (2, 32), "audio": (2, 32), "video": (2, 32)}
    model.eval = MagicMock()

    call_count = [0]

    def fake_forward(batch, **kwargs):
        call_count[0] += 1
        B = 2
        # Make text-ablated predictions differ more to simulate importance
        if torch.all(batch.data.get("text", torch.ones(1)) == 0):
            return torch.ones(B, n_vertices, 10) * 0.5
        elif torch.all(batch.data.get("audio", torch.ones(1)) == 0):
            return torch.ones(B, n_vertices, 10) * 0.8
        elif torch.all(batch.data.get("video", torch.ones(1)) == 0):
            return torch.ones(B, n_vertices, 10) * 0.9
        return torch.ones(B, n_vertices, 10) * 1.0

    model.side_effect = fake_forward
    model.return_value = torch.ones(2, n_vertices, 10)
    # Override __call__ to use our function
    model.__class__ = type("MockModel", (), {
        "__call__": staticmethod(fake_forward),
        "feature_dims": {"text": (2, 32), "audio": (2, 32), "video": (2, 32)},
        "eval": lambda self: None,
    })
    # Simpler approach: just use a plain class
    class FakeModel:
        feature_dims = {"text": (2, 32), "audio": (2, 32), "video": (2, 32)}
        def eval(self): pass
        def __call__(self, batch, **kwargs):
            return fake_forward(batch, **kwargs)
    return FakeModel()


class TestModalityAttributor:
    def test_ablation_basic(self):
        from neuralset.dataloader import SegmentData

        from cortexlab.inference.attribution import ModalityAttributor

        model = _make_mock_model()
        attributor = ModalityAttributor(model)

        data = {
            "text": torch.randn(2, 2, 32, 20),
            "audio": torch.randn(2, 2, 32, 20),
            "video": torch.randn(2, 2, 32, 20),
            "subject_id": torch.zeros(2, dtype=torch.long),
        }
        batch = SegmentData(data=data, segments=make_segments(2))
        scores = attributor.attribute(batch)

        assert "text" in scores
        assert "audio" in scores
        assert "video" in scores
        assert scores["text"].shape == (100,)

    def test_text_most_important(self):
        from neuralset.dataloader import SegmentData

        from cortexlab.inference.attribution import ModalityAttributor

        model = _make_mock_model()
        attributor = ModalityAttributor(model)

        data = {
            "text": torch.randn(2, 2, 32, 20),
            "audio": torch.randn(2, 2, 32, 20),
            "video": torch.randn(2, 2, 32, 20),
            "subject_id": torch.zeros(2, dtype=torch.long),
        }
        batch = SegmentData(data=data, segments=make_segments(2))
        scores = attributor.attribute(batch)

        # Text ablation causes the biggest change (1.0 -> 0.5 = 0.5 diff)
        assert scores["text"].mean() > scores["audio"].mean()
        assert scores["audio"].mean() > scores["video"].mean()

    def test_normalised_scores_sum_to_one(self):
        from neuralset.dataloader import SegmentData

        from cortexlab.inference.attribution import ModalityAttributor

        model = _make_mock_model()
        attributor = ModalityAttributor(model)

        data = {
            "text": torch.randn(2, 2, 32, 20),
            "audio": torch.randn(2, 2, 32, 20),
            "video": torch.randn(2, 2, 32, 20),
            "subject_id": torch.zeros(2, dtype=torch.long),
        }
        batch = SegmentData(data=data, segments=make_segments(2))
        scores = attributor.attribute(batch)

        total = scores["text_normalised"] + scores["audio_normalised"] + scores["video_normalised"]
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_with_roi_indices(self):
        from neuralset.dataloader import SegmentData

        from cortexlab.inference.attribution import ModalityAttributor

        roi_indices = {
            "V1": np.array([0, 1, 2, 3, 4]),
            "MT": np.array([10, 11, 12]),
        }
        model = _make_mock_model()
        attributor = ModalityAttributor(model, roi_indices=roi_indices)

        data = {
            "text": torch.randn(2, 2, 32, 20),
            "audio": torch.randn(2, 2, 32, 20),
            "video": torch.randn(2, 2, 32, 20),
            "subject_id": torch.zeros(2, dtype=torch.long),
        }
        batch = SegmentData(data=data, segments=make_segments(2))
        scores = attributor.attribute(batch)

        assert "text_roi" in scores
        assert "V1" in scores["text_roi"]
        assert "MT" in scores["text_roi"]
