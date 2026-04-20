"""Tests for ``cortexlab.data.studies.lahner2024bold.list_stimulus_paths``.

The helper does not touch the real dataset. Tests use a tiny tmp tree
that mirrors BOLD Moments' on-disk layout.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cortexlab.data.studies.lahner2024bold import (
    STIMULI_SUBPATH,
    list_stimulus_paths,
)

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _make_fake_dataset(root: Path, n_train: int = 3, n_test: int = 2,
                      suffix: str = ".mp4") -> Path:
    """Create a tree that matches the expected BOLD Moments layout."""
    stim = root / STIMULI_SUBPATH
    (stim / "train").mkdir(parents=True, exist_ok=True)
    (stim / "test").mkdir(parents=True, exist_ok=True)
    for i in range(n_train):
        (stim / "train" / f"{i:04d}_train{suffix}").write_bytes(b"")
    for i in range(n_test):
        (stim / "test" / f"{i:04d}_test{suffix}").write_bytes(b"")
    return root


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #

def test_returns_train_then_test(tmp_path: Path):
    _make_fake_dataset(tmp_path, n_train=3, n_test=2)
    paths = list_stimulus_paths(tmp_path)
    assert len(paths) == 5
    # First three are train, last two are test, each group is sorted.
    assert [p.parent.name for p in paths[:3]] == ["train"] * 3
    assert [p.parent.name for p in paths[3:]] == ["test"] * 2
    assert paths[:3] == sorted(paths[:3])
    assert paths[3:] == sorted(paths[3:])


def test_split_train_only(tmp_path: Path):
    _make_fake_dataset(tmp_path, n_train=4, n_test=1)
    paths = list_stimulus_paths(tmp_path, split="train")
    assert len(paths) == 4
    assert all(p.parent.name == "train" for p in paths)


def test_split_test_only(tmp_path: Path):
    _make_fake_dataset(tmp_path, n_train=1, n_test=3)
    paths = list_stimulus_paths(tmp_path, split="test")
    assert len(paths) == 3
    assert all(p.parent.name == "test" for p in paths)


def test_invalid_split_rejected(tmp_path: Path):
    _make_fake_dataset(tmp_path)
    with pytest.raises(ValueError, match="split must be"):
        list_stimulus_paths(tmp_path, split="val")


# --------------------------------------------------------------------------- #
# suffix + variants
# --------------------------------------------------------------------------- #

def test_custom_suffix(tmp_path: Path):
    _make_fake_dataset(tmp_path, suffix=".webm")
    paths = list_stimulus_paths(tmp_path, suffix=".webm")
    assert len(paths) == 5
    assert all(p.suffix == ".webm" for p in paths)


def test_returns_path_objects(tmp_path: Path):
    _make_fake_dataset(tmp_path)
    for p in list_stimulus_paths(tmp_path):
        assert isinstance(p, Path)
        assert p.exists()


# --------------------------------------------------------------------------- #
# env var fallback
# --------------------------------------------------------------------------- #

def test_env_var_fallback(tmp_path: Path, monkeypatch):
    _make_fake_dataset(tmp_path)
    monkeypatch.setenv("CORTEXLAB_DATA", str(tmp_path))
    paths = list_stimulus_paths()
    assert len(paths) == 5


def test_no_root_and_no_env_raises(monkeypatch):
    monkeypatch.delenv("CORTEXLAB_DATA", raising=False)
    with pytest.raises(RuntimeError, match="CORTEXLAB_DATA"):
        list_stimulus_paths()


# --------------------------------------------------------------------------- #
# parent-of-root fallback
# --------------------------------------------------------------------------- #

def test_parent_folder_resolves_to_bold_moments_subdir(tmp_path: Path):
    """If the user passes the parent that contains a bold_moments/ dir,
    we should find the dataset inside it."""
    inner = tmp_path / "bold_moments"
    _make_fake_dataset(inner)
    paths = list_stimulus_paths(tmp_path)
    assert len(paths) == 5
    assert all("bold_moments" in str(p) for p in paths)


# --------------------------------------------------------------------------- #
# error paths                                                                 #
# --------------------------------------------------------------------------- #

def test_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        list_stimulus_paths(tmp_path / "does-not-exist")


def test_missing_stim_tree_raises(tmp_path: Path):
    (tmp_path / "something_else").mkdir()
    with pytest.raises(FileNotFoundError, match="expected BOLD Moments stimuli"):
        list_stimulus_paths(tmp_path)


def test_missing_split_dir_raises(tmp_path: Path):
    """Train dir absent but stim tree present: raise the right error."""
    stim = tmp_path / STIMULI_SUBPATH
    (stim / "test").mkdir(parents=True)
    (stim / "test" / "0001.mp4").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="missing split directory"):
        list_stimulus_paths(tmp_path)


def test_empty_split_dir_raises(tmp_path: Path):
    _make_fake_dataset(tmp_path, n_train=0, n_test=3)
    with pytest.raises(FileNotFoundError, match="no \\*.mp4 files"):
        list_stimulus_paths(tmp_path)
