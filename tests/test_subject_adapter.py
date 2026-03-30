"""Tests for cross-subject adaptation."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _make_mock_model(n_subjects=3, hidden=32, n_vertices=50):
    """Create a mock model with predictor weights."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.linear_baseline = False
    model.config.low_rank_head = None
    model.config.hidden = hidden

    model.aggregate_features = MagicMock(
        return_value=torch.randn(2, 10, hidden)
    )
    model.transformer_forward = MagicMock(
        return_value=torch.randn(2, 10, hidden)
    )
    model.pooler = MagicMock(
        return_value=torch.randn(2, hidden, 10)
    )
    model.eval = MagicMock()

    predictor = MagicMock()
    predictor.weights = torch.nn.Parameter(
        torch.randn(n_subjects, hidden, n_vertices)
    )
    predictor.bias = None
    model.predictor = predictor

    return model


def _make_calibration_loader(n_batches=2, batch_size=2, n_vertices=50):
    """Create a mock calibration data loader."""
    from neuralset.dataloader import SegmentData

    batches = []
    for _ in range(n_batches):
        data = {
            "text": torch.randn(batch_size, 2, 32, 10),
            "fmri": torch.randn(batch_size, n_vertices, 10),
            "subject_id": torch.zeros(batch_size, dtype=torch.long),
        }
        import neuralset.segments as seg
        segments = [seg.Segment(start=float(i), duration=1.0, timeline="test") for i in range(batch_size)]
        batches.append(SegmentData(data=data, segments=segments))
    return batches


class TestSubjectAdapter:
    def test_nearest_neighbor(self):
        from cortexlab.core.subject import SubjectAdapter

        model = _make_mock_model()
        loader = _make_calibration_loader()
        adapter = SubjectAdapter.from_nearest_neighbor(model, loader)

        assert adapter._weights.shape[0] == 1  # one new subject
        assert adapter._weights.shape[1] == 32  # hidden dim
        assert adapter._weights.shape[2] == 50  # n_vertices

    def test_inject_into_model(self):
        from cortexlab.core.subject import SubjectAdapter

        model = _make_mock_model(n_subjects=3)
        adapter = SubjectAdapter(weights=torch.randn(1, 32, 50))
        new_id = adapter.inject_into_model(model)

        assert new_id == 3
        assert model.predictor.weights.shape[0] == 4
