"""Tests for :func:`cortexlab.data.studies.lahner2024bold.load_noise_ceiling`.

Synthesizes a tiny on-disk mirror of the BOLD Moments
``prepared_betas/`` layout for one subject, writes per-hemisphere
ceiling pickles in the author's naming convention, and exercises the
loader's parsing and error paths without needing the real dataset.
"""

from __future__ import annotations

import pickle as pkl
from pathlib import Path

import numpy as np
import pytest

from cortexlab.data.studies.lahner2024bold import (
    BETAS_SUBPATH,
    NOISE_CEILING_FILENAME_TEMPLATE,
    load_noise_ceiling,
)


@pytest.fixture
def tiny_vertices(monkeypatch):
    """Shrink N_VERTICES_PER_HEMI so tiny synthetic pickles are valid."""
    monkeypatch.setattr(
        "cortexlab.data.studies.lahner2024bold.N_VERTICES_PER_HEMI", 16,
    )
    return 16


def _write_ceiling(
    root: Path,
    subject_id: int,
    split: str,
    n: int,
    lh: np.ndarray,
    rh: np.ndarray,
    *,
    payload_wrap: str = "array",
) -> None:
    """Write both hemisphere ceiling pickles for one subject.

    ``payload_wrap`` selects how the per-hemisphere array is stored:
    ``"array"`` plain ndarray, ``"tuple"`` single-element tuple,
    ``"list"`` single-element list.
    """
    sub_dir = root / BETAS_SUBPATH / f"sub-{subject_id:02d}" / "prepared_betas"
    sub_dir.mkdir(parents=True, exist_ok=True)
    for hemi, arr in (("left", lh), ("right", rh)):
        fp = sub_dir / NOISE_CEILING_FILENAME_TEMPLATE.format(
            subject_id=subject_id, split=split, hemi=hemi, n=n,
        )
        if payload_wrap == "array":
            payload = arr
        elif payload_wrap == "tuple":
            payload = (arr,)
        elif payload_wrap == "list":
            payload = [arr]
        else:
            raise ValueError(payload_wrap)
        with fp.open("wb") as f:
            pkl.dump(payload, f)


# --------------------------------------------------------------------------- #
# happy path                                                                  #
# --------------------------------------------------------------------------- #

def test_load_noise_ceiling_shape_and_dtype(tmp_path, tiny_vertices):
    lh = np.linspace(0.0, 0.5, tiny_vertices, dtype=np.float32)
    rh = np.linspace(0.2, 0.8, tiny_vertices, dtype=np.float32)
    _write_ceiling(tmp_path, subject_id=1, split="test", n=10, lh=lh, rh=rh)

    ceiling = load_noise_ceiling(subject_id=1, root=tmp_path, split="test", n=10)
    assert ceiling.shape == (2 * tiny_vertices,)
    assert ceiling.dtype == np.float32
    np.testing.assert_allclose(ceiling[:tiny_vertices], lh)
    np.testing.assert_allclose(ceiling[tiny_vertices:], rh)


def test_load_noise_ceiling_accepts_tuple_payload(tmp_path, tiny_vertices):
    lh = np.random.default_rng(0).random(tiny_vertices).astype(np.float32)
    rh = np.random.default_rng(1).random(tiny_vertices).astype(np.float32)
    _write_ceiling(tmp_path, 1, "test", 10, lh, rh, payload_wrap="tuple")
    ceiling = load_noise_ceiling(subject_id=1, root=tmp_path)
    assert ceiling.shape == (2 * tiny_vertices,)


def test_load_noise_ceiling_accepts_list_payload(tmp_path, tiny_vertices):
    lh = np.zeros(tiny_vertices, dtype=np.float32)
    rh = np.ones(tiny_vertices, dtype=np.float32) * 0.3
    _write_ceiling(tmp_path, 1, "test", 10, lh, rh, payload_wrap="list")
    ceiling = load_noise_ceiling(subject_id=1, root=tmp_path)
    assert abs(ceiling.mean() - 0.15) < 1e-6


# --------------------------------------------------------------------------- #
# split + n variations                                                        #
# --------------------------------------------------------------------------- #

def test_load_noise_ceiling_train_split(tmp_path, tiny_vertices):
    lh = np.full(tiny_vertices, 0.25, dtype=np.float32)
    rh = np.full(tiny_vertices, 0.35, dtype=np.float32)
    _write_ceiling(tmp_path, 1, "train", 10, lh, rh)
    ceiling = load_noise_ceiling(subject_id=1, root=tmp_path, split="train")
    assert ceiling.shape == (2 * tiny_vertices,)
    assert abs(ceiling[:tiny_vertices].mean() - 0.25) < 1e-6


def test_load_noise_ceiling_nondefault_n(tmp_path, tiny_vertices):
    lh = np.full(tiny_vertices, 0.1, dtype=np.float32)
    rh = np.full(tiny_vertices, 0.1, dtype=np.float32)
    _write_ceiling(tmp_path, 1, "test", n=5, lh=lh, rh=rh)
    ceiling = load_noise_ceiling(subject_id=1, root=tmp_path, n=5)
    assert ceiling.shape == (2 * tiny_vertices,)


# --------------------------------------------------------------------------- #
# error paths                                                                 #
# --------------------------------------------------------------------------- #

def test_load_noise_ceiling_rejects_bad_split(tmp_path, tiny_vertices):
    with pytest.raises(ValueError, match="split must be"):
        load_noise_ceiling(subject_id=1, root=tmp_path, split="val")


def test_load_noise_ceiling_missing_betas_dir(tmp_path, tiny_vertices):
    with pytest.raises(FileNotFoundError, match="prepared betas"):
        load_noise_ceiling(subject_id=1, root=tmp_path)


def test_load_noise_ceiling_missing_hemisphere_file(tmp_path, tiny_vertices):
    lh = np.zeros(tiny_vertices, dtype=np.float32)
    rh = np.zeros(tiny_vertices, dtype=np.float32)
    _write_ceiling(tmp_path, 1, "test", 10, lh, rh)
    # Remove the right-hemisphere file to simulate an incomplete staging.
    fp_rh = (
        tmp_path / BETAS_SUBPATH / "sub-01" / "prepared_betas"
        / NOISE_CEILING_FILENAME_TEMPLATE.format(
            subject_id=1, split="test", hemi="right", n=10,
        )
    )
    fp_rh.unlink()
    with pytest.raises(FileNotFoundError, match="noise-ceiling"):
        load_noise_ceiling(subject_id=1, root=tmp_path)


def test_load_noise_ceiling_wrong_shape_raises(tmp_path, tiny_vertices):
    lh = np.zeros(tiny_vertices + 3, dtype=np.float32)   # wrong length
    rh = np.zeros(tiny_vertices, dtype=np.float32)
    _write_ceiling(tmp_path, 1, "test", 10, lh, rh)
    with pytest.raises(ValueError, match="expected ceiling shape"):
        load_noise_ceiling(subject_id=1, root=tmp_path)


def test_load_noise_ceiling_unknown_payload_type(tmp_path, tiny_vertices):
    sub_dir = tmp_path / BETAS_SUBPATH / "sub-01" / "prepared_betas"
    sub_dir.mkdir(parents=True, exist_ok=True)
    for hemi in ("left", "right"):
        fp = sub_dir / NOISE_CEILING_FILENAME_TEMPLATE.format(
            subject_id=1, split="test", hemi=hemi, n=10,
        )
        with fp.open("wb") as f:
            pkl.dump({"oops": "dict"}, f)     # neither array nor tuple
    with pytest.raises(TypeError, match="unexpected noise-ceiling payload"):
        load_noise_ceiling(subject_id=1, root=tmp_path)


# --------------------------------------------------------------------------- #
# integration: the array is usable by roi_summary                             #
# --------------------------------------------------------------------------- #

def test_loaded_ceiling_feeds_roi_summary_shape(tmp_path, tiny_vertices):
    """Sanity: the output is a flat 1-D array of the right length so
    downstream ``roi_summary(..., ceiling=...)`` calls won't shape-check-fail.
    """
    lh = np.full(tiny_vertices, 0.4, dtype=np.float32)
    rh = np.full(tiny_vertices, 0.6, dtype=np.float32)
    _write_ceiling(tmp_path, 1, "test", 10, lh, rh)
    ceiling = load_noise_ceiling(subject_id=1, root=tmp_path)
    assert ceiling.ndim == 1
    assert ceiling.size == 2 * tiny_vertices
