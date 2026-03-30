"""Tests for the core FmriEncoder model."""

from unittest.mock import MagicMock

import pytest
import torch


def _make_model(hidden=256, n_outputs=100, n_timesteps=10, modalities=None):
    """Build a small FmriEncoderModel for testing."""
    from neuraltrain.models.transformer import TransformerEncoder

    from cortexlab.core.model import FmriEncoder

    if modalities is None:
        modalities = {"text": (2, 32), "audio": (2, 32), "video": (2, 32)}

    config = FmriEncoder(
        hidden=hidden,
        max_seq_len=128,
        dropout=0.0,
        modality_dropout=0.0,
        temporal_dropout=0.0,
        linear_baseline=False,
        encoder=TransformerEncoder(depth=2, heads=4),
    )
    model = config.build(
        feature_dims=modalities,
        n_outputs=n_outputs,
        n_output_timesteps=n_timesteps,
    )
    return model


def _make_segments(n):
    """Create dummy segments for SegmentData."""
    import neuralset.segments as seg
    return [seg.Segment(start=float(i), duration=1.0, timeline="test") for i in range(n)]


def _make_batch(modalities, batch_size=2, seq_len=20):
    """Create a synthetic SegmentData-like batch."""
    from neuralset.dataloader import SegmentData

    data = {}
    for name, (n_layers, feat_dim) in modalities.items():
        data[name] = torch.randn(batch_size, n_layers, feat_dim, seq_len)
    data["subject_id"] = torch.zeros(batch_size, dtype=torch.long)
    return SegmentData(data=data, segments=_make_segments(batch_size))


class TestFmriEncoderModel:
    def test_forward_shape(self):
        modalities = {"text": (2, 32), "audio": (2, 32)}
        model = _make_model(modalities=modalities)
        batch = _make_batch(modalities)
        out = model(batch)
        assert out.shape == (2, 100, 10), f"Expected (2, 100, 10), got {out.shape}"

    def test_forward_no_pool(self):
        modalities = {"text": (2, 32)}
        model = _make_model(modalities=modalities)
        batch = _make_batch(modalities)
        out = model(batch, pool_outputs=False)
        assert out.shape[0] == 2
        assert out.shape[1] == 100

    def test_return_attn(self):
        modalities = {"text": (2, 32)}
        model = _make_model(modalities=modalities)
        batch = _make_batch(modalities)
        result = model(batch, return_attn=True)
        assert isinstance(result, tuple)
        out, attn_maps = result
        assert out.shape == (2, 100, 10)
        # attn_maps may be empty if the transformer doesn't expose weights
        assert isinstance(attn_maps, list)

    def test_missing_modality_zeros(self):
        modalities = {"text": (2, 32), "audio": (2, 32)}
        model = _make_model(modalities=modalities)
        # Only provide text, not audio
        from neuralset.dataloader import SegmentData
        data = {"text": torch.randn(2, 2, 32, 20), "subject_id": torch.zeros(2, dtype=torch.long)}
        batch = SegmentData(data=data, segments=_make_segments(2))
        out = model(batch)
        assert out.shape == (2, 100, 10)

    def test_modality_dropout_training(self):
        modalities = {"text": (2, 32), "audio": (2, 32)}
        from neuraltrain.models.transformer import TransformerEncoder

        from cortexlab.core.model import FmriEncoder
        config = FmriEncoder(
            hidden=256, max_seq_len=128, modality_dropout=0.5,
            encoder=TransformerEncoder(depth=2, heads=4),
        )
        model = config.build(feature_dims=modalities, n_outputs=100, n_output_timesteps=10)
        model.train()
        batch = _make_batch(modalities)
        out = model(batch)
        assert out.shape == (2, 100, 10)

    def test_linear_baseline(self):
        modalities = {"text": (2, 32)}
        from cortexlab.core.model import FmriEncoder
        config = FmriEncoder(hidden=256, linear_baseline=True)
        model = config.build(feature_dims=modalities, n_outputs=100, n_output_timesteps=10)
        batch = _make_batch(modalities)
        out = model(batch)
        assert out.shape == (2, 100, 10)
