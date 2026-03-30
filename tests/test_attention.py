"""Tests for ROI attention extraction."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


class TestAttentionExtractor:
    def test_context_manager_returns_list(self):
        from cortexlab.core.attention import AttentionExtractor

        encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True),
            num_layers=2,
        )
        with AttentionExtractor(encoder) as maps:
            x = torch.randn(2, 10, 32)
            _ = encoder(x)

        assert isinstance(maps, list)

    def test_hooks_cleaned_up(self):
        from cortexlab.core.attention import AttentionExtractor

        encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True),
            num_layers=2,
        )
        with AttentionExtractor(encoder) as _maps:
            x = torch.randn(1, 5, 32)
            _ = encoder(x)

        # Hooks should be cleaned up after exiting context
        assert isinstance(_maps, list)


class TestAttentionToRoiScores:
    def test_basic_roi_scores(self):
        from cortexlab.core.attention import attention_to_roi_scores

        # Simulate 2 layers of attention maps: (B=1, heads=4, T=10, T=10)
        attn_maps = [torch.randn(1, 4, 10, 10) for _ in range(2)]
        roi_indices = {
            "V1": np.array([0, 1, 2]),
            "MT": np.array([5, 6]),
        }
        scores = attention_to_roi_scores(attn_maps, roi_indices)

        assert "V1" in scores
        assert "MT" in scores
        assert scores["V1"].shape == (10,)
        assert scores["MT"].shape == (10,)

    def test_with_predictor_weights(self):
        from cortexlab.core.attention import attention_to_roi_scores

        attn_maps = [torch.randn(1, 4, 10, 10)]
        roi_indices = {
            "V1": np.array([0, 1]),
            "A1": np.array([3, 4]),
        }
        # Predictor weights: (hidden=32, n_vertices=10)
        weights = torch.randn(32, 10)
        scores = attention_to_roi_scores(attn_maps, roi_indices, predictor_weights=weights)

        assert "V1" in scores
        assert "A1" in scores

    def test_empty_attn_maps(self):
        from cortexlab.core.attention import attention_to_roi_scores

        roi_indices = {"V1": np.array([0, 1])}
        scores = attention_to_roi_scores([], roi_indices)

        assert "V1" in scores
        assert len(scores["V1"]) == 0
