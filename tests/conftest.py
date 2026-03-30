"""Shared test utilities and fixtures."""

import neuralset.segments as seg
import numpy as np
import pytest
import torch
from neuralset.dataloader import SegmentData


def make_segments(n):
    """Create n dummy segments for SegmentData construction."""
    return [seg.Segment(start=float(i), duration=1.0, timeline="test") for i in range(n)]


@pytest.fixture
def roi_indices():
    """Standard HCP ROI indices for testing across cognitive dimensions."""
    return {
        # Executive
        "46": np.array([0, 1]),
        "FEF": np.array([2, 3]),
        "p32pr": np.array([4]),
        # Visual
        "V1": np.array([10, 11, 12]),
        "V2": np.array([13, 14]),
        "MT": np.array([15, 16]),
        # Auditory
        "A1": np.array([20, 21, 22]),
        "LBelt": np.array([23]),
        # Language
        "44": np.array([30, 31]),
        "45": np.array([32, 33]),
        "TPOJ1": np.array([34]),
    }


@pytest.fixture
def sample_brain_predictions():
    """Brain predictions shaped (n_timepoints=30, n_vertices=50)."""
    np.random.seed(42)
    return np.random.randn(30, 50)


@pytest.fixture
def sample_model_features():
    """Model features shaped (n_stimuli=20, feature_dim=64)."""
    np.random.seed(42)
    return np.random.randn(20, 64)


class FakeModel:
    """Minimal mock fMRI model for testing without real weights."""

    feature_dims = {"text": (2, 32), "audio": (2, 32)}

    def eval(self):
        pass

    def __call__(self, batch, **kwargs):
        B = next(iter(batch.data.values())).shape[0]
        return torch.ones(B, 100, 10)


@pytest.fixture
def mock_fmri_model():
    """A fake model that returns constant predictions."""
    return FakeModel()


@pytest.fixture
def sample_batch():
    """SegmentData batch with text + audio modalities, batch_size=2."""
    data = {
        "text": torch.randn(2, 2, 32, 20),
        "audio": torch.randn(2, 2, 32, 20),
        "subject_id": torch.zeros(2, dtype=torch.long),
    }
    return SegmentData(data=data, segments=make_segments(2))
