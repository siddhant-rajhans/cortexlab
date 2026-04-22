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


def test_roi_summary_with_ceiling_adds_normalized_column():
    """Ceiling-aware roi_summary should report both raw and normalized
    per-ROI R^2, and should skip voxels whose ceiling falls below
    ``min_ceiling``.
    """
    train, test, y_tr, y_te, _ = _synth_multimodal()
    result = run_modality_lesion(train, test, y_tr, y_te,
                                 alphas=[1.0], cv=2, mask_strategy="zero")
    rois = {"text_roi": np.array([0, 1, 2, 3])}
    # Build a ceiling of the right length where two voxels are well below
    # the threshold; the other two sit at 0.5 so normalized = full_r2 / 0.5.
    n_vox = result.full_r2.shape[0]
    ceiling = np.full(n_vox, 0.5, dtype=np.float32)
    ceiling[0] = 0.0         # dropped
    ceiling[1] = 0.001       # below min_ceiling
    summary = roi_summary(result, rois, ceiling=ceiling)

    assert "full_r2_normalized" in summary["text_roi"]
    assert "ceiling_mean" in summary["text_roi"]
    # Normalized average should only average across voxels 2 and 3.
    full = result.full_r2.cpu().numpy()
    expected = float((full[2:4] / 0.5).mean())
    assert abs(summary["text_roi"]["full_r2_normalized"] - expected) < 1e-6


def test_roi_summary_rejects_mismatched_ceiling_shape():
    train, test, y_tr, y_te, _ = _synth_multimodal()
    result = run_modality_lesion(train, test, y_tr, y_te,
                                 alphas=[1.0], cv=2, mask_strategy="zero")
    bad_ceiling = np.zeros(result.full_r2.shape[0] + 5, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        roi_summary(result, {"x": np.array([0])}, ceiling=bad_ceiling)


def test_roi_summary_without_ceiling_unchanged():
    """Backward compat: calls without ceiling return the pre-existing schema."""
    train, test, y_tr, y_te, _ = _synth_multimodal()
    result = run_modality_lesion(train, test, y_tr, y_te,
                                 alphas=[1.0], cv=2, mask_strategy="zero")
    summary = roi_summary(result, {"text_roi": np.array([0, 1])})
    row = summary["text_roi"]
    assert "full_r2_normalized" not in row
    assert "ceiling_mean" not in row
    assert "full_r2" in row


# --------------------------------------------------------------------------- #
# permutation test                                                            #
# --------------------------------------------------------------------------- #

def test_permutation_test_produces_small_p_for_true_signal():
    """When a modality genuinely drives voxels, permuting its test-time
    rows should make those voxels' delta_R^2 shrink, so the observed
    delta should sit near the top of the null distribution and the
    corresponding p-values should be small."""
    train, test, y_tr, y_te, assignments = _synth_multimodal(noise=0.05)
    result = run_modality_lesion(
        train, test, y_tr, y_te,
        alphas=[1e-2, 1.0, 1e2], cv=3, mask_strategy="zero",
        n_permutations=200, permutation_seed=0,
    )
    assert result.p_values is not None
    assert result.n_permutations == 200
    # For each modality's true voxels, at least most should be significant.
    for m, sl in assignments.items():
        p_m = result.p_values[m][sl.start:sl.stop].cpu().numpy()
        frac_sig = (p_m < 0.05).mean()
        assert frac_sig > 0.5, (
            f"modality {m}: expected most true voxels significant, "
            f"got frac_sig={frac_sig:.2f}, p={p_m}"
        )


def test_permutation_test_large_p_for_noise_voxels():
    """Voxels that don't depend on modality m should have p-values that
    are NOT uniformly tiny. We allow some leakage through ridge
    regularization but demand it doesn't look like a clean signal."""
    train, test, y_tr, y_te, assignments = _synth_multimodal(noise=0.05)
    result = run_modality_lesion(
        train, test, y_tr, y_te,
        alphas=[1.0], cv=2, mask_strategy="zero",
        n_permutations=200, permutation_seed=1,
    )
    text_sl = assignments["text"]
    for m in ("audio", "video"):
        p_m = result.p_values[m][text_sl.start:text_sl.stop].cpu().numpy()
        frac_sig = (p_m < 0.05).mean()
        assert frac_sig <= 0.5, (
            f"null-modality {m} on text voxels: too many false positives "
            f"(frac_sig={frac_sig:.2f})"
        )


def test_permutation_test_zero_permutations_keeps_p_none():
    train, test, y_tr, y_te, _ = _synth_multimodal()
    result = run_modality_lesion(
        train, test, y_tr, y_te,
        alphas=[1.0], cv=2, mask_strategy="zero",
        n_permutations=0,
    )
    assert result.p_values is None
    assert result.n_permutations == 0


def test_permutation_test_reproducible_with_seed():
    """Same seed => identical p-values; different seed => different."""
    train, test, y_tr, y_te, _ = _synth_multimodal()
    kwargs = dict(
        alphas=[1.0], cv=2, mask_strategy="zero",
        n_permutations=50,
    )
    a = run_modality_lesion(train, test, y_tr, y_te,
                            permutation_seed=7, **kwargs)
    b = run_modality_lesion(train, test, y_tr, y_te,
                            permutation_seed=7, **kwargs)
    c = run_modality_lesion(train, test, y_tr, y_te,
                            permutation_seed=13, **kwargs)
    for m in a.modality_order:
        assert torch.equal(a.p_values[m], b.p_values[m]), f"seed reproducibility broken for {m}"
        assert not torch.equal(a.p_values[m], c.p_values[m])


def test_permutation_p_values_respect_one_sided_bounds():
    """All p-values live in [1/(B+1), 1] because of the Phipson-Smyth +1 smoothing."""
    train, test, y_tr, y_te, _ = _synth_multimodal()
    result = run_modality_lesion(
        train, test, y_tr, y_te,
        alphas=[1.0], cv=2, mask_strategy="zero",
        n_permutations=10, permutation_seed=0,
    )
    expected_floor = 1.0 / (10 + 1)
    for m in result.modality_order:
        p = result.p_values[m]
        assert (p >= expected_floor - 1e-6).all()
        assert (p <= 1.0 + 1e-6).all()


def test_roi_summary_exposes_permutation_columns():
    """roi_summary should surface p-value aggregates per ROI when the
    LesionResult has populated p_values."""
    train, test, y_tr, y_te, _ = _synth_multimodal()
    result = run_modality_lesion(
        train, test, y_tr, y_te,
        alphas=[1.0], cv=2, mask_strategy="zero",
        n_permutations=50, permutation_seed=0,
    )
    rois = {
        "text_roi":  np.array([0, 1, 2, 3]),
        "audio_roi": np.array([4, 5, 6, 7]),
        "video_roi": np.array([8, 9, 10, 11]),
    }
    summary = roi_summary(result, rois)
    for roi_name in rois:
        row = summary[roi_name]
        for m in result.modality_order:
            assert f"p_{m}_median" in row
            assert f"frac_sig_{m}" in row
            assert 0.0 <= row[f"p_{m}_median"] <= 1.0
            assert 0.0 <= row[f"frac_sig_{m}"] <= 1.0
    # The "own" modality should be at least as significant as the non-owner ones.
    assert summary["text_roi"]["p_text_median"] <= summary["text_roi"]["p_audio_median"]
    assert summary["audio_roi"]["p_audio_median"] <= summary["audio_roi"]["p_video_median"]


def test_roi_summary_without_permutations_omits_p_columns():
    train, test, y_tr, y_te, _ = _synth_multimodal()
    result = run_modality_lesion(train, test, y_tr, y_te,
                                 alphas=[1.0], cv=2, mask_strategy="zero")
    summary = roi_summary(result, {"text_roi": np.array([0, 1, 2, 3])})
    row = summary["text_roi"]
    for m in result.modality_order:
        assert f"p_{m}_median" not in row
        assert f"frac_sig_{m}" not in row


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_lesion_handles_cuda_device_end_to_end():
    """Regression test: y_test arrives on CPU, encoder moves inputs to
    CUDA, predict returns CUDA; _r2_score must not fail on device mismatch.
    The smoke-test run on an L40S surfaced this before the fix.
    """
    train, test, y_tr, y_te, assignments = _synth_multimodal(seed=5)
    result = run_modality_lesion(
        train, test, y_tr, y_te,
        alphas=[1.0], cv=2, mask_strategy="zero",
        device="cuda", backend="torch",
    )
    assert result.full_r2.device.type == "cuda"
    for m in result.modality_order:
        assert result.delta_r2[m].device.type == "cuda"
    # Sanity: the ground-truth recovery still holds on GPU.
    for m, sl in assignments.items():
        own = result.delta_r2[m][sl.start : sl.stop].mean().item()
        assert own > 0.2, f"{m}: expected large dR^2, got {own:.3f}"


def test_lesion_rejects_mismatched_keys():
    train = {"text": np.random.randn(10, 4).astype(np.float32),
             "audio": np.random.randn(10, 4).astype(np.float32)}
    test = {"text": np.random.randn(3, 4).astype(np.float32),
            "video": np.random.randn(3, 4).astype(np.float32)}
    with pytest.raises(ValueError, match="share keys"):
        run_modality_lesion(train, test, np.zeros((10, 2), dtype=np.float32),
                            np.zeros((3, 2), dtype=np.float32))
