"""Tests for :mod:`cortexlab.data.parcellations`.

Synthetic annot-style arrays cover every branch of ``build_roi_indices``
and the HCP-MMP convenience loaders. No on-disk atlas required.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cortexlab.data import parcellations as pmod
from cortexlab.data.parcellations import (
    DEFAULT_HCP_MMP_ROIS,
    build_roi_indices,
    load_hcp_mmp_from_freesurfer,
    load_hcp_mmp_fsaverage,
)

# --------------------------------------------------------------------------- #
# default ROI set                                                             #
# --------------------------------------------------------------------------- #

def test_default_roi_set_covers_pillars():
    """The default set should include at least one ROI from each of the
    visual / auditory / language pillars a NeuroAI paper needs."""
    s = set(DEFAULT_HCP_MMP_ROIS)
    assert {"V1", "V2", "V4"} <= s            # early visual
    assert {"MT", "MST"} <= s                  # motion
    assert {"FFC"} <= s                        # face complex
    assert {"A1"} <= s                         # auditory
    assert {"44", "45"} <= s                   # Broca
    assert {"PGs", "PGi"} <= s                 # angular gyrus


def test_default_roi_set_has_no_duplicates():
    assert len(DEFAULT_HCP_MMP_ROIS) == len(set(DEFAULT_HCP_MMP_ROIS))


# --------------------------------------------------------------------------- #
# build_roi_indices — happy paths                                             #
# --------------------------------------------------------------------------- #

def _fake_lh_rh():
    """A minimal synthetic bihemispheric annot.

    Left hemisphere has 4 vertices labeled [0, 1, 1, 2].
    Right hemisphere has 4 vertices labeled [0, 2, 1, 1].

    Region 0 is the unknown/???
    Region 1 is V1
    Region 2 is V2
    """
    labels_lh = np.array([0, 1, 1, 2], dtype=np.int32)
    names_lh = ["L_???", "L_V1_ROI", "L_V2_ROI"]
    labels_rh = np.array([0, 2, 1, 1], dtype=np.int32)
    names_rh = ["R_???", "R_V1_ROI", "R_V2_ROI"]
    return labels_lh, names_lh, labels_rh, names_rh


def test_build_roi_indices_happy_path():
    labels_lh, names_lh, labels_rh, names_rh = _fake_lh_rh()
    out = build_roi_indices(
        labels_lh, names_lh, labels_rh, names_rh,
        rois=["V1", "V2"],
    )
    # V1 is label 1 in LH (vertices 1, 2) and label 1 in RH (vertices 2, 3).
    # Right hemisphere indices are offset by n_lh = 4.
    assert sorted(out["V1"].tolist()) == [1, 2, 6, 7]
    # V2 is label 2 in LH (vertex 3) and label 2 in RH (vertex 1).
    assert sorted(out["V2"].tolist()) == [3, 5]


def test_build_roi_indices_dtype_is_int64():
    labels_lh, names_lh, labels_rh, names_rh = _fake_lh_rh()
    out = build_roi_indices(labels_lh, names_lh, labels_rh, names_rh, rois=["V1"])
    assert out["V1"].dtype == np.int64


def test_build_roi_indices_is_sorted():
    """Indices should be sorted so downstream slicing has no surprises."""
    labels_lh, names_lh, labels_rh, names_rh = _fake_lh_rh()
    out = build_roi_indices(labels_lh, names_lh, labels_rh, names_rh, rois=["V1"])
    arr = out["V1"]
    assert np.all(arr[:-1] <= arr[1:])


def test_build_roi_indices_right_offset_correct():
    """Right hemisphere vertex K must land at index K + n_lh in the output."""
    # LH has 10 vertices, all background.
    labels_lh = np.zeros(10, dtype=np.int32)
    names_lh = ["L_???", "L_V1_ROI"]
    # RH has 4 vertices, last one is V1.
    labels_rh = np.array([0, 0, 0, 1], dtype=np.int32)
    names_rh = ["R_???", "R_V1_ROI"]
    out = build_roi_indices(labels_lh, names_lh, labels_rh, names_rh, rois=["V1"])
    assert out["V1"].tolist() == [13]      # 10 + 3


# --------------------------------------------------------------------------- #
# name normalization: prefix / suffix / case                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("roi_name", ["V1", "v1", "L_V1_ROI", "R_V1_ROI", " V1 "])
def test_build_roi_indices_matches_regardless_of_decoration(roi_name):
    labels_lh, names_lh, labels_rh, names_rh = _fake_lh_rh()
    out = build_roi_indices(
        labels_lh, names_lh, labels_rh, names_rh, rois=[roi_name],
    )
    # Should match V1 in both hemispheres regardless of how the caller spelled it.
    assert len(out) == 1
    only_key = next(iter(out))
    assert sorted(out[only_key].tolist()) == [1, 2, 6, 7]


def test_build_roi_indices_tolerates_bytes_names():
    """nibabel < 5 returns bytes; build_roi_indices should not care."""
    labels_lh = np.array([0, 1], dtype=np.int32)
    names_lh = [b"L_???", b"L_V1_ROI"]
    labels_rh = np.array([0, 1], dtype=np.int32)
    names_rh = [b"R_???", b"R_V1_ROI"]
    out = build_roi_indices(labels_lh, names_lh, labels_rh, names_rh, rois=["V1"])
    assert out["V1"].tolist() == [1, 3]    # LH vertex 1, RH vertex 1 + n_lh(2)


# --------------------------------------------------------------------------- #
# missing / zero-vertex ROIs                                                  #
# --------------------------------------------------------------------------- #

def test_build_roi_indices_missing_strict_raises():
    labels_lh, names_lh, labels_rh, names_rh = _fake_lh_rh()
    with pytest.raises(KeyError, match="not found"):
        build_roi_indices(
            labels_lh, names_lh, labels_rh, names_rh,
            rois=["V1", "FFC"], strict=True,
        )


def test_build_roi_indices_missing_non_strict_skips_and_warns(caplog):
    labels_lh, names_lh, labels_rh, names_rh = _fake_lh_rh()
    with caplog.at_level(logging.WARNING, logger="cortexlab.data.parcellations"):
        out = build_roi_indices(
            labels_lh, names_lh, labels_rh, names_rh,
            rois=["V1", "FFC"], strict=False,
        )
    assert "V1" in out
    assert "FFC" not in out
    assert any("FFC" in r.message for r in caplog.records)


def test_build_roi_indices_one_hemisphere_only_is_fine():
    """If an ROI exists only in LH (unusual but possible), the returned
    indices cover just that hemisphere."""
    labels_lh = np.array([0, 1, 1], dtype=np.int32)
    names_lh = ["L_???", "L_V1_ROI"]
    labels_rh = np.array([0, 0], dtype=np.int32)
    names_rh = ["R_???"]
    out = build_roi_indices(labels_lh, names_lh, labels_rh, names_rh, rois=["V1"])
    assert out["V1"].tolist() == [1, 2]


def test_build_roi_indices_zero_vertex_roi_is_skipped(caplog):
    """A region name that's listed but has no vertices labeled with it
    should be skipped rather than returning an empty array (downstream
    code treats zero-size arrays as errors)."""
    labels_lh = np.array([0, 0, 0], dtype=np.int32)
    names_lh = ["L_???", "L_V1_ROI"]   # V1 declared but never labeled
    labels_rh = np.array([0, 0, 0], dtype=np.int32)
    names_rh = ["R_???", "R_V1_ROI"]
    with caplog.at_level(logging.WARNING, logger="cortexlab.data.parcellations"):
        out = build_roi_indices(labels_lh, names_lh, labels_rh, names_rh, rois=["V1"])
    assert "V1" not in out
    assert any("zero vertices" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# shape / input validation                                                    #
# --------------------------------------------------------------------------- #

def test_build_roi_indices_rejects_2d_labels():
    labels_lh = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="1-D"):
        build_roi_indices(
            labels_lh, ["L_???"], np.zeros(3, dtype=np.int32), ["R_???"],
            rois=["V1"],
        )


def test_build_roi_indices_empty_rois_returns_empty_dict():
    labels_lh, names_lh, labels_rh, names_rh = _fake_lh_rh()
    out = build_roi_indices(labels_lh, names_lh, labels_rh, names_rh, rois=[])
    assert out == {}


# --------------------------------------------------------------------------- #
# integration with load_subject's contract                                    #
# --------------------------------------------------------------------------- #

def test_result_is_loadable_as_parcellation_kwarg():
    """The dict shape must be {str: int64 ndarray} so load_subject accepts it."""
    labels_lh, names_lh, labels_rh, names_rh = _fake_lh_rh()
    out = build_roi_indices(
        labels_lh, names_lh, labels_rh, names_rh, rois=["V1", "V2"],
    )
    assert isinstance(out, dict)
    for key, val in out.items():
        assert isinstance(key, str)
        assert isinstance(val, np.ndarray)
        assert val.dtype == np.int64
        assert val.ndim == 1


# --------------------------------------------------------------------------- #
# load_hcp_mmp_fsaverage (monkeypatched read_annot)                           #
# --------------------------------------------------------------------------- #

def _install_fake_nibabel(monkeypatch, annots: dict[Path, tuple]):
    """Install a fake nibabel.freesurfer.io.read_annot that returns the
    annot tuple for whichever path the caller gives."""
    calls: list[str] = []

    def fake_read_annot(path: str):
        calls.append(path)
        return annots[Path(path)]

    fake_fsio = SimpleNamespace(read_annot=fake_read_annot)
    fake_freesurfer = SimpleNamespace(io=fake_fsio)
    fake_nibabel = SimpleNamespace(freesurfer=fake_freesurfer)

    monkeypatch.setitem(sys.modules, "nibabel", fake_nibabel)
    monkeypatch.setitem(sys.modules, "nibabel.freesurfer", fake_freesurfer)
    monkeypatch.setitem(sys.modules, "nibabel.freesurfer.io", fake_fsio)
    return calls


def test_load_hcp_mmp_fsaverage_happy(tmp_path, monkeypatch):
    labels_lh, names_lh, labels_rh, names_rh = _fake_lh_rh()
    lh_path = tmp_path / "lh.HCPMMP1.annot"
    rh_path = tmp_path / "rh.HCPMMP1.annot"
    lh_path.write_bytes(b"")
    rh_path.write_bytes(b"")

    ctab = np.zeros((3, 5), dtype=np.int32)
    annots = {
        lh_path: (labels_lh, ctab, [n.encode() for n in names_lh]),
        rh_path: (labels_rh, ctab, [n.encode() for n in names_rh]),
    }
    _install_fake_nibabel(monkeypatch, annots)

    out = load_hcp_mmp_fsaverage(lh_path, rh_path, rois=["V1", "V2"])
    assert set(out.keys()) == {"V1", "V2"}
    assert sorted(out["V1"].tolist()) == [1, 2, 6, 7]


def test_load_hcp_mmp_fsaverage_missing_lh_raises(tmp_path, monkeypatch):
    rh_path = tmp_path / "rh.HCPMMP1.annot"
    rh_path.write_bytes(b"")
    _install_fake_nibabel(monkeypatch, {})
    with pytest.raises(FileNotFoundError, match="left"):
        load_hcp_mmp_fsaverage(tmp_path / "missing.annot", rh_path)


def test_load_hcp_mmp_fsaverage_missing_rh_raises(tmp_path, monkeypatch):
    lh_path = tmp_path / "lh.HCPMMP1.annot"
    lh_path.write_bytes(b"")
    _install_fake_nibabel(monkeypatch, {})
    with pytest.raises(FileNotFoundError, match="right"):
        load_hcp_mmp_fsaverage(lh_path, tmp_path / "missing.annot")


def test_load_hcp_mmp_fsaverage_default_rois(tmp_path, monkeypatch):
    """When rois=None, the loader tries every DEFAULT_HCP_MMP_ROIS name.
    Absent regions are skipped (non-strict)."""
    # Build a minimal annot that contains V1 only.
    labels_lh = np.array([0, 1], dtype=np.int32)
    names_lh = ["L_???", "L_V1_ROI"]
    labels_rh = np.array([0, 1], dtype=np.int32)
    names_rh = ["R_???", "R_V1_ROI"]

    lh_path = tmp_path / "lh.HCPMMP1.annot"
    rh_path = tmp_path / "rh.HCPMMP1.annot"
    lh_path.write_bytes(b"")
    rh_path.write_bytes(b"")

    ctab = np.zeros((2, 5), dtype=np.int32)
    annots = {
        lh_path: (labels_lh, ctab, [n.encode() for n in names_lh]),
        rh_path: (labels_rh, ctab, [n.encode() for n in names_rh]),
    }
    _install_fake_nibabel(monkeypatch, annots)

    out = load_hcp_mmp_fsaverage(lh_path, rh_path)      # rois=None
    assert "V1" in out
    # Everything else in DEFAULT_HCP_MMP_ROIS is absent in our tiny annot.
    assert set(out.keys()) == {"V1"}


# --------------------------------------------------------------------------- #
# load_hcp_mmp_from_freesurfer (SUBJECTS_DIR)                                 #
# --------------------------------------------------------------------------- #

def test_load_hcp_mmp_from_freesurfer_uses_subjects_dir(tmp_path, monkeypatch):
    subj_root = tmp_path / "subjects" / "fsaverage" / "label"
    subj_root.mkdir(parents=True)
    lh_path = subj_root / "lh.HCPMMP1.annot"
    rh_path = subj_root / "rh.HCPMMP1.annot"
    lh_path.write_bytes(b"")
    rh_path.write_bytes(b"")

    labels_lh, names_lh, labels_rh, names_rh = _fake_lh_rh()
    ctab = np.zeros((3, 5), dtype=np.int32)
    annots = {
        lh_path: (labels_lh, ctab, [n.encode() for n in names_lh]),
        rh_path: (labels_rh, ctab, [n.encode() for n in names_rh]),
    }
    _install_fake_nibabel(monkeypatch, annots)

    out = load_hcp_mmp_from_freesurfer(
        subjects_dir=tmp_path / "subjects", rois=["V1"],
    )
    assert "V1" in out


def test_load_hcp_mmp_from_freesurfer_falls_back_to_env(tmp_path, monkeypatch):
    subj_root = tmp_path / "subjects" / "fsaverage" / "label"
    subj_root.mkdir(parents=True)
    lh_path = subj_root / "lh.HCPMMP1.annot"
    rh_path = subj_root / "rh.HCPMMP1.annot"
    lh_path.write_bytes(b"")
    rh_path.write_bytes(b"")

    labels_lh, names_lh, labels_rh, names_rh = _fake_lh_rh()
    ctab = np.zeros((3, 5), dtype=np.int32)
    annots = {
        lh_path: (labels_lh, ctab, [n.encode() for n in names_lh]),
        rh_path: (labels_rh, ctab, [n.encode() for n in names_rh]),
    }
    _install_fake_nibabel(monkeypatch, annots)

    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path / "subjects"))
    out = load_hcp_mmp_from_freesurfer(rois=["V1"])
    assert "V1" in out


def test_load_hcp_mmp_from_freesurfer_no_env_raises(monkeypatch):
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    with pytest.raises(RuntimeError, match="SUBJECTS_DIR"):
        load_hcp_mmp_from_freesurfer()


# --------------------------------------------------------------------------- #
# internal helpers (exercised indirectly above, but pinning behavior)         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,expected", [
    ("V1", "v1"),
    ("L_V1_ROI", "v1"),
    ("R_V1_ROI", "v1"),
    ("  v1  ", "v1"),
    ("L_FFC_ROI", "ffc"),
    ("44", "44"),
    (b"L_V1_ROI", "v1"),
])
def test_canonical_normalizes(name, expected):
    assert pmod._canonical(name) == expected


@pytest.mark.parametrize("name,expected", [
    ("V1", "V1"),
    ("L_V1_ROI", "V1"),
    ("R_FFC_ROI", "FFC"),
    ("44", "44"),
])
def test_friendly_strips_decoration_preserves_case(name, expected):
    assert pmod._friendly(name) == expected
