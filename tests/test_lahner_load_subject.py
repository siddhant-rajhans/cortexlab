"""Tests for ``cortexlab.data.studies.lahner2024bold.load_subject``.

The real betas are 10+ GB, so these tests fabricate a tiny on-disk
stand-in that matches the BOLD Moments GLMsingle prepared-betas layout.
Only the shapes and pickle format are preserved; the values are dummy.
"""

from __future__ import annotations

import pickle as pkl
from pathlib import Path

import numpy as np
import pytest

from cortexlab.data.studies.lahner2024bold import (
    BETAS_SUBPATH,
    N_TEST_STIMULI,
    N_TRAIN_STIMULI,
    N_VERTICES_PER_HEMI,
    load_subject,
)

# --------------------------------------------------------------------------- #
# fabricate a tiny on-disk BOLD Moments tree                                  #
# --------------------------------------------------------------------------- #

def _make_fake_subject(
    root: Path,
    subject_id: int = 1,
    n_train: int = N_TRAIN_STIMULI,
    n_test: int = N_TEST_STIMULI,
    n_vertices: int = N_VERTICES_PER_HEMI,
    n_reps_train: int = 3,
    n_reps_test: int = 10,
    seed: int = 0,
) -> None:
    """Write four pickle files matching the shipped beta layout."""
    rng = np.random.default_rng(seed)
    sub_dir = (
        root
        / BETAS_SUBPATH
        / f"sub-{subject_id:02d}"
        / "prepared_betas"
    )
    sub_dir.mkdir(parents=True, exist_ok=True)

    for split, n_trials, n_reps in (("train", n_train, n_reps_train),
                                    ("test",  n_test,  n_reps_test)):
        stim_list = [f"{i:04d}" for i in range(1, n_trials + 1)]
        for hemi in ("left", "right"):
            betas = rng.standard_normal(
                (n_trials, n_reps, n_vertices)
            ).astype(np.float32)
            fp = (
                sub_dir
                / f"sub-{subject_id:02d}_organized_betas_task-{split}_hemi-{hemi}_normalized.pkl"
            )
            with fp.open("wb") as f:
                pkl.dump([betas, stim_list], f)


def _make_fake_feature_cache(
    cache_dir: Path,
    modalities=("vision", "text"),
    n_total: int = N_TRAIN_STIMULI + N_TEST_STIMULI,
    dim: int = 32,
    seed: int = 1,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for m in modalities:
        arr = rng.standard_normal((n_total, dim)).astype(np.float32)
        np.savez(cache_dir / f"{m}.npz", features=arr)


# Full-size fake data is ~2.5 GB per subject (1000 trials x 3 reps x 163842).
# For tests we use a tiny variant; load_subject does not hard-code sizes
# except via the N_TRAIN_STIMULI / N_TEST_STIMULI module constants. Patch
# those for the duration of the tiny tests so the helper accepts our shrunken
# synthetic data.

@pytest.fixture
def tiny_sizes(monkeypatch):
    monkeypatch.setattr(
        "cortexlab.data.studies.lahner2024bold.N_TRAIN_STIMULI", 8,
    )
    monkeypatch.setattr(
        "cortexlab.data.studies.lahner2024bold.N_TEST_STIMULI", 4,
    )
    monkeypatch.setattr(
        "cortexlab.data.studies.lahner2024bold.N_VERTICES_PER_HEMI", 16,
    )
    yield {"train": 8, "test": 4, "vert": 16}


# --------------------------------------------------------------------------- #
# happy path                                                                  #
# --------------------------------------------------------------------------- #

def test_load_subject_shapes(tmp_path: Path, tiny_sizes):
    _make_fake_subject(
        tmp_path, subject_id=1,
        n_train=tiny_sizes["train"], n_test=tiny_sizes["test"],
        n_vertices=tiny_sizes["vert"],
    )
    rec = load_subject(subject_id=1, root=tmp_path)
    assert rec["subject_id"] == 1
    assert rec["y_train"].shape == (tiny_sizes["train"], 2 * tiny_sizes["vert"])
    assert rec["y_test"].shape == (tiny_sizes["test"], 2 * tiny_sizes["vert"])
    assert rec["y_train"].dtype == np.float32
    assert rec["features_train"] == {}
    assert rec["features_test"] == {}
    assert set(rec["roi_indices"]) == {"all_cortex"}
    assert rec["roi_indices"]["all_cortex"].shape == (2 * tiny_sizes["vert"],)
    assert rec["stimulus_ids_train"] == [f"{i:04d}" for i in range(1, tiny_sizes["train"] + 1)]
    assert rec["stimulus_ids_test"] == [f"{i:04d}" for i in range(1, tiny_sizes["test"] + 1)]


def test_load_subject_averages_across_reps(tmp_path: Path, tiny_sizes):
    _make_fake_subject(
        tmp_path, subject_id=1,
        n_train=tiny_sizes["train"], n_test=tiny_sizes["test"],
        n_vertices=tiny_sizes["vert"],
        n_reps_train=3, n_reps_test=10,
    )
    rec = load_subject(subject_id=1, root=tmp_path)
    # Re-read one pickle and check that load_subject produced the repetition mean.
    fp = (
        tmp_path / BETAS_SUBPATH / "sub-01" / "prepared_betas"
        / "sub-01_organized_betas_task-train_hemi-left_normalized.pkl"
    )
    with fp.open("rb") as f:
        betas, _ = pkl.load(f)
    expected_left_mean = betas.mean(axis=1)
    got_left = rec["y_train"][:, : tiny_sizes["vert"]]
    np.testing.assert_allclose(got_left, expected_left_mean, rtol=1e-5, atol=1e-6)


def test_load_subject_with_feature_cache(tmp_path: Path, tiny_sizes):
    _make_fake_subject(
        tmp_path, subject_id=1,
        n_train=tiny_sizes["train"], n_test=tiny_sizes["test"],
        n_vertices=tiny_sizes["vert"],
    )
    cache = tmp_path / "features"
    _make_fake_feature_cache(
        cache,
        n_total=tiny_sizes["train"] + tiny_sizes["test"],
        dim=6,
    )
    rec = load_subject(
        subject_id=1, root=tmp_path,
        feature_cache=cache, modalities=("vision", "text"),
    )
    assert set(rec["features_train"]) == {"vision", "text"}
    assert rec["features_train"]["vision"].shape == (tiny_sizes["train"], 6)
    assert rec["features_test"]["vision"].shape == (tiny_sizes["test"], 6)
    # Round-trip: rebuilt cache agrees with the on-disk array split.
    stored = np.load(cache / "vision.npz")["features"]
    np.testing.assert_array_equal(rec["features_train"]["vision"], stored[: tiny_sizes["train"]])
    np.testing.assert_array_equal(
        rec["features_test"]["vision"], stored[tiny_sizes["train"] :],
    )


def test_load_subject_pilot_truncation(tmp_path: Path, tiny_sizes):
    _make_fake_subject(
        tmp_path, subject_id=1,
        n_train=tiny_sizes["train"], n_test=tiny_sizes["test"],
        n_vertices=tiny_sizes["vert"],
    )
    cache = tmp_path / "features"
    _make_fake_feature_cache(
        cache,
        n_total=tiny_sizes["train"] + tiny_sizes["test"],
        dim=4,
    )
    rec = load_subject(
        subject_id=1, root=tmp_path,
        feature_cache=cache, modalities=("vision",),
        n_trimmed_stimuli=3,
    )
    assert rec["y_train"].shape[0] == 3
    assert rec["features_train"]["vision"].shape == (3, 4)
    # Test split is untouched.
    assert rec["y_test"].shape[0] == tiny_sizes["test"]


def test_load_subject_custom_parcellation(tmp_path: Path, tiny_sizes):
    _make_fake_subject(
        tmp_path, subject_id=1,
        n_train=tiny_sizes["train"], n_test=tiny_sizes["test"],
        n_vertices=tiny_sizes["vert"],
    )
    parcellation = {
        "left_half":  np.arange(tiny_sizes["vert"]),
        "right_half": np.arange(tiny_sizes["vert"], 2 * tiny_sizes["vert"]),
    }
    rec = load_subject(subject_id=1, root=tmp_path, parcellation=parcellation)
    assert set(rec["roi_indices"]) == {"left_half", "right_half"}
    np.testing.assert_array_equal(
        rec["roi_indices"]["right_half"],
        np.arange(tiny_sizes["vert"], 2 * tiny_sizes["vert"]),
    )


# --------------------------------------------------------------------------- #
# failure modes                                                               #
# --------------------------------------------------------------------------- #

def test_load_subject_missing_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="prepared betas"):
        load_subject(subject_id=1, root=tmp_path)


def test_load_subject_missing_feature_cache(tmp_path: Path, tiny_sizes):
    _make_fake_subject(
        tmp_path, subject_id=1,
        n_train=tiny_sizes["train"], n_test=tiny_sizes["test"],
        n_vertices=tiny_sizes["vert"],
    )
    with pytest.raises(FileNotFoundError, match="feature cache directory"):
        load_subject(
            subject_id=1, root=tmp_path,
            feature_cache=tmp_path / "does_not_exist",
        )


def test_load_subject_missing_modality_file(tmp_path: Path, tiny_sizes):
    _make_fake_subject(
        tmp_path, subject_id=1,
        n_train=tiny_sizes["train"], n_test=tiny_sizes["test"],
        n_vertices=tiny_sizes["vert"],
    )
    cache = tmp_path / "features"
    _make_fake_feature_cache(
        cache, modalities=("vision",),
        n_total=tiny_sizes["train"] + tiny_sizes["test"], dim=4,
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        load_subject(
            subject_id=1, root=tmp_path,
            feature_cache=cache, modalities=("vision", "audio"),
        )


def test_load_subject_feature_row_mismatch(tmp_path: Path, tiny_sizes):
    _make_fake_subject(
        tmp_path, subject_id=1,
        n_train=tiny_sizes["train"], n_test=tiny_sizes["test"],
        n_vertices=tiny_sizes["vert"],
    )
    cache = tmp_path / "features"
    cache.mkdir()
    # Wrong row count: 5 rows instead of n_train+n_test.
    np.savez(cache / "vision.npz", features=np.zeros((5, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="expected"):
        load_subject(
            subject_id=1, root=tmp_path,
            feature_cache=cache, modalities=("vision",),
        )


def test_load_subject_hemisphere_mismatch(tmp_path: Path, tiny_sizes):
    _make_fake_subject(
        tmp_path, subject_id=1,
        n_train=tiny_sizes["train"], n_test=tiny_sizes["test"],
        n_vertices=tiny_sizes["vert"],
    )
    # Corrupt one pickle: change its stimulus-id list order.
    fp = (
        tmp_path / BETAS_SUBPATH / "sub-01" / "prepared_betas"
        / "sub-01_organized_betas_task-train_hemi-right_normalized.pkl"
    )
    with fp.open("rb") as f:
        betas, stims = pkl.load(f)
    with fp.open("wb") as f:
        pkl.dump([betas, list(reversed(stims))], f)
    with pytest.raises(ValueError, match="stimulus order disagrees"):
        load_subject(subject_id=1, root=tmp_path)
