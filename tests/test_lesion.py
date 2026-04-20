"""Tests for the causal modality lesion protocol.

We construct synthetic multimodal encoders where each voxel depends on
a *known* modality, then verify that the lesion detects those dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cortexlab.analysis.lesion import LesionResult, roi_summary, run_modality_lesion


def _synth_multimodal(n_train=400, n_test=100, p_per_mod=16, noise=0.1, seed=0):
    """Generate features and voxel responses where each voxel depends
    on exactly one modality. Voxel 0..3 -> text; 4..7 -> audio; 8..11 -> video.
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    mods = {
        "text":  rng.standard_normal((n_total, p_per_mod)).astype(np.float32),
        "audio": rng.standard_normal((n_total, p_per_mod)).astype(np.float32),
        "video": rng.standard_normal((n_total, p_per_mod)).astype(np.float32),
    }
    n_vox_per_mod = 4
    n_vox = 3 * n_vox_per_mod
    Y = np.zeros((n_total, n_vox), dtype=np.float32)
    assignments = {"text": slice(0, 4), "audio": slice(4, 8), "video": slice(8, 12)}
    for m, sl in assignments.items():
        w = rng.standard_normal((p_per_mod, n_vox_per_mod)).astype(np.float32)
        Y[:, sl] = mods[m] @ w
    Y += noise * rng.standard_normal(Y.shape).astype(np.float32)

    train = {m: mods[m][:n_train] for m in mods}
    test = {m: mods[m][n_train:] for m in mods}
    return train, test, Y[:n_train], Y[n_train:], assignments


# --------------------------------------------------------------------------- #

def test_lesion_recovers_ground_truth_dependencies():
    train, test, y_tr, y_te, assignments = _synth_multimodal()
    result = run_modality_lesion(
        train, test, y_tr, y_te,
        alphas=[1e-2, 1.0, 1e2], cv=3, mask_strategy="zero",
    )
    assert isinstance(result, LesionResult)
    assert result.full_r2.shape == (12,)
    assert set(result.delta_r2) == {"text", "audio", "video"}
    for m in ("text", "audio", "video"):
        assert result.delta_r2[m].shape == (12,)

    # For each modality's voxel block, ablating that modality should
    # drop R^2 the most; ablating the other two should be near-zero.
    for m, sl in assignments.items():
        own = result.delta_r2[m][sl.start:sl.stop].mean().item()
        others = [
            result.delta_r2[m2][sl.start:sl.stop].mean().item()
            for m2 in assignments if m2 != m
        ]
        assert own > 0.2, f"lesioning {m} should drop {m}-voxels' R^2 a lot (got {own:.3f})"
        for other in others:
            assert other < own / 3, (
                f"lesioning non-owner modality should hurt {m}-voxels less "
                f"(own={own:.3f}, other={other:.3f})"
            )


def test_lesion_learned_mask_roughly_agrees_with_zero_mask():
    """The two mask strategies should qualitatively rank modalities the same."""
    train, test, y_tr, y_te, assignments = _synth_multimodal(seed=1)
    zero_res = run_modality_lesion(train, test, y_tr, y_te,
                                   alphas=[1.0], cv=2, mask_strategy="zero")
    learned_res = run_modality_lesion(train, test, y_tr, y_te,
                                      alphas=[1.0], cv=2, mask_strategy="learned")
    for m, sl in assignments.items():
        z = zero_res.delta_r2[m][sl].mean().item()
        l_ = learned_res.delta_r2[m][sl].mean().item()
        # Both should identify the dependency with the same sign and
        # roughly similar magnitude (within 2x).
        assert z > 0 and l_ > 0
        assert 0.5 < (l_ / z) < 2.0, f"modality {m}: zero={z:.3f}, learned={l_:.3f}"


def test_lesion_roi_summary_aggregates_correctly():
    train, test, y_tr, y_te, _ = _synth_multimodal()
    result = run_modality_lesion(train, test, y_tr, y_te,
                                 alphas=[1.0], cv=2, mask_strategy="zero")
    rois = {
        "text_roi":  np.array([0, 1, 2, 3]),
        "audio_roi": np.array([4, 5, 6, 7]),
        "video_roi": np.array([8, 9, 10, 11]),
    }
    summary = roi_summary(result, rois)
    assert set(summary) == set(rois)
    assert summary["text_roi"]["dR2_text"] > summary["text_roi"]["dR2_audio"]
    assert summary["text_roi"]["dR2_text"] > summary["text_roi"]["dR2_video"]
    assert summary["audio_roi"]["dR2_audio"] > summary["audio_roi"]["dR2_text"]
    assert summary["video_roi"]["dR2_video"] > summary["video_roi"]["dR2_audio"]


def test_lesion_rejects_single_modality():
    train = {"text": np.random.randn(20, 8).astype(np.float32)}
    test = {"text": np.random.randn(5, 8).astype(np.float32)}
    with pytest.raises(ValueError, match="at least 2"):
        run_modality_lesion(train, test, np.zeros((20, 3), dtype=np.float32),
                            np.zeros((5, 3), dtype=np.float32))


def test_lesion_rejects_mismatched_keys():
    train = {"text": np.random.randn(10, 4).astype(np.float32),
             "audio": np.random.randn(10, 4).astype(np.float32)}
    test = {"text": np.random.randn(3, 4).astype(np.float32),
            "video": np.random.randn(3, 4).astype(np.float32)}
    with pytest.raises(ValueError, match="share keys"):
        run_modality_lesion(train, test, np.zeros((10, 2), dtype=np.float32),
                            np.zeros((3, 2), dtype=np.float32))
