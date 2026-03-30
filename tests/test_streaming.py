"""Tests for the streaming predictor."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def _make_mock_model(n_vertices=100, hidden=64):
    """Create a mock model that returns predictable outputs."""
    model = MagicMock()
    model.feature_dims = {"text": (2, 32), "audio": (2, 32)}
    model.config = MagicMock()
    model.config.hidden = hidden

    def fake_forward(batch, pool_outputs=True):
        return torch.randn(1, n_vertices, 10)

    model.__call__ = fake_forward
    model.return_value = torch.randn(1, n_vertices, 10)
    return model


class TestStreamingPredictor:
    def test_buffer_fill(self):
        from cortexlab.inference.streaming import StreamingPredictor

        model = _make_mock_model()
        sp = StreamingPredictor(model, window_trs=5, step_trs=1)

        # First 4 frames should return None (buffer not full)
        for _ in range(4):
            features = {
                "text": torch.randn(2, 32),
                "audio": torch.randn(2, 32),
            }
            result = sp.push_frame(features)
            assert result is None

    def test_prediction_emission(self):
        from cortexlab.inference.streaming import StreamingPredictor

        model = _make_mock_model()
        sp = StreamingPredictor(model, window_trs=3, step_trs=1)

        for i in range(3):
            features = {"text": torch.randn(2, 32), "audio": torch.randn(2, 32)}
            result = sp.push_frame(features)

        # 3rd frame should trigger prediction
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)

    def test_step_control(self):
        from cortexlab.inference.streaming import StreamingPredictor

        model = _make_mock_model()
        sp = StreamingPredictor(model, window_trs=3, step_trs=2)

        results = []
        for _ in range(5):
            features = {"text": torch.randn(2, 32), "audio": torch.randn(2, 32)}
            result = sp.push_frame(features)
            results.append(result)

        # With window=3, step=2: first prediction at frame 4 (buffer full at 3, then wait 2 more)
        # Frame 0: buffer[0], not full -> None
        # Frame 1: buffer[0,1], not full -> None
        # Frame 2: buffer[0,1,2], full, frames_since=3 >= step=2 -> predict
        # Frame 3: frames_since=1 < step=2 -> None
        # Frame 4: frames_since=2 >= step=2 -> predict
        assert results[0] is None
        assert results[1] is None
        assert results[2] is not None
        assert results[3] is None
        assert results[4] is not None

    def test_missing_modality(self):
        from cortexlab.inference.streaming import StreamingPredictor

        model = _make_mock_model()
        sp = StreamingPredictor(model, window_trs=2, step_trs=1)

        # Only provide text, not audio
        for _ in range(2):
            features = {"text": torch.randn(2, 32)}
            result = sp.push_frame(features)

        assert result is not None

    def test_flush(self):
        from cortexlab.inference.streaming import StreamingPredictor

        model = _make_mock_model()
        sp = StreamingPredictor(model, window_trs=3, step_trs=1)

        for _ in range(3):
            features = {"text": torch.randn(2, 32), "audio": torch.randn(2, 32)}
            sp.push_frame(features)

        results = sp.flush()
        assert len(results) >= 1

    def test_reset(self):
        from cortexlab.inference.streaming import StreamingPredictor

        model = _make_mock_model()
        sp = StreamingPredictor(model, window_trs=3, step_trs=1)

        for _ in range(3):
            features = {"text": torch.randn(2, 32)}
            sp.push_frame(features)

        sp.reset()
        assert len(sp._buffer) == 0
        assert sp._frames_since_emit == 0
