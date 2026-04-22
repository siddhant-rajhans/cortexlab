"""Tests for ``load_captions`` and ``middle_frame_paths``.

Both read auxiliary data files produced by the BOLD Moments CSAIL
stimulus archive and OpenNeuro metadata. Tests stub a tiny mirror of
the on-disk layout; no real dataset access required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cortexlab.data.studies.lahner2024bold import (
    CAPTIONS_SUBPATH,
    STIMULI_SUBPATH,
    load_captions,
    middle_frame_paths,
)


def _make_stimuli_tree(root: Path, n_train: int = 3, n_test: int = 2):
    stim = root / STIMULI_SUBPATH
    (stim / "train").mkdir(parents=True, exist_ok=True)
    (stim / "test").mkdir(parents=True, exist_ok=True)
    # BOLD Moments uses zero-padded 4-digit stems across the combined
    # 1102 clips. For tests we mimic a sortable subset.
    for i in range(n_train):
        (stim / "train" / f"{i + 1:04d}.mp4").write_bytes(b"")
    for j in range(n_test):
        (stim / "test" / f"{n_train + j + 1:04d}.mp4").write_bytes(b"")


def _make_captions(
    root: Path, stems: list[str],
    generator: str = "GIT-git-large-coco",
    n_captions: int = 5,
):
    path = root / CAPTIONS_SUBPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    ann = {
        stem: {generator: [f"caption {k} for {stem}" for k in range(n_captions)]}
        for stem in stems
    }
    path.write_text(json.dumps(ann))


def _make_middle_frames(root: Path, stems: list[str]):
    frames_root = root / "stimulus_set" / "frames_middle"
    frames_root.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (frames_root / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # empty JPEG marker


# --------------------------------------------------------------------------- #
# load_captions                                                               #
# --------------------------------------------------------------------------- #

def test_load_captions_default_picks_first(tmp_path: Path):
    _make_stimuli_tree(tmp_path)
    stems = [f"{i:04d}" for i in range(1, 6)]
    _make_captions(tmp_path, stems)
    caps = load_captions(tmp_path)
    assert len(caps) == 5
    assert caps == [f"caption 0 for {s}" for s in stems]


def test_load_captions_with_index(tmp_path: Path):
    _make_stimuli_tree(tmp_path)
    stems = [f"{i:04d}" for i in range(1, 6)]
    _make_captions(tmp_path, stems)
    caps = load_captions(tmp_path, caption_index=3)
    assert caps == [f"caption 3 for {s}" for s in stems]


def test_load_captions_join_concatenates_all(tmp_path: Path):
    _make_stimuli_tree(tmp_path)
    stems = [f"{i:04d}" for i in range(1, 6)]
    _make_captions(tmp_path, stems, n_captions=3)
    caps = load_captions(tmp_path, join=True)
    assert len(caps) == 5
    assert caps[0] == "caption 0 for 0001 caption 1 for 0001 caption 2 for 0001"


def test_load_captions_alignment_matches_stimulus_paths(tmp_path: Path):
    """Captions align with list_stimulus_paths row-by-row."""
    _make_stimuli_tree(tmp_path, n_train=3, n_test=2)
    stems = [f"{i:04d}" for i in range(1, 6)]
    _make_captions(tmp_path, stems)

    from cortexlab.data.studies.lahner2024bold import list_stimulus_paths
    paths = list_stimulus_paths(tmp_path)
    caps = load_captions(tmp_path)
    for p, c in zip(paths, caps):
        assert p.stem in c, f"caption for {p.stem} should mention the stem, got {c!r}"


def test_load_captions_missing_file(tmp_path: Path):
    _make_stimuli_tree(tmp_path)
    with pytest.raises(FileNotFoundError, match="captions"):
        load_captions(tmp_path)


def test_load_captions_missing_stimulus_raises(tmp_path: Path):
    _make_stimuli_tree(tmp_path)
    # Captions only for 3 of the 5 stimuli.
    _make_captions(tmp_path, ["0001", "0002", "0003"])
    with pytest.raises(KeyError, match="no caption entry"):
        load_captions(tmp_path)


def test_load_captions_missing_generator(tmp_path: Path):
    _make_stimuli_tree(tmp_path)
    stems = [f"{i:04d}" for i in range(1, 6)]
    _make_captions(tmp_path, stems, generator="other-generator")
    with pytest.raises(KeyError, match="generator"):
        load_captions(tmp_path)  # asks for default GIT generator


def test_load_captions_caption_index_out_of_range(tmp_path: Path):
    _make_stimuli_tree(tmp_path)
    stems = [f"{i:04d}" for i in range(1, 6)]
    _make_captions(tmp_path, stems, n_captions=3)
    with pytest.raises(IndexError, match="caption_index"):
        load_captions(tmp_path, caption_index=10)


# --------------------------------------------------------------------------- #
# middle_frame_paths                                                          #
# --------------------------------------------------------------------------- #

def test_middle_frame_paths_happy(tmp_path: Path):
    _make_stimuli_tree(tmp_path, n_train=3, n_test=2)
    stems = [f"{i:04d}" for i in range(1, 6)]
    _make_middle_frames(tmp_path, stems)
    frames = middle_frame_paths(tmp_path)
    assert len(frames) == 5
    assert [f.stem for f in frames] == stems


def test_middle_frame_paths_missing_directory(tmp_path: Path):
    _make_stimuli_tree(tmp_path)
    with pytest.raises(FileNotFoundError, match="middle-frame JPGs"):
        middle_frame_paths(tmp_path)


def test_middle_frame_paths_missing_one_frame(tmp_path: Path):
    _make_stimuli_tree(tmp_path, n_train=3, n_test=2)
    stems = [f"{i:04d}" for i in range(1, 6)]
    _make_middle_frames(tmp_path, stems[:-1])  # skip the last stem
    with pytest.raises(FileNotFoundError, match="missing middle frame"):
        middle_frame_paths(tmp_path)


def test_middle_frame_paths_resolves_csail_suffix(tmp_path: Path):
    """CSAIL ships frames named ``<stem>_<frame_idx>_<total>.jpg`` rather
    than the bare ``<stem>.jpg`` the docstring implies. The helper should
    glob for the CSAIL convention when the bare filename is absent.
    """
    _make_stimuli_tree(tmp_path, n_train=3, n_test=2)
    frames_root = tmp_path / "stimulus_set" / "frames_middle"
    frames_root.mkdir(parents=True, exist_ok=True)
    for i in range(1, 6):
        (frames_root / f"{i:04d}_45_90.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    frames = middle_frame_paths(tmp_path)
    assert len(frames) == 5
    assert [f.name for f in frames] == [f"{i:04d}_45_90.jpg" for i in range(1, 6)]


def test_middle_frame_paths_prefers_bare_over_glob(tmp_path: Path):
    """When both a symlink named ``<stem>.jpg`` and the CSAIL-style
    ``<stem>_<idx>.jpg`` exist, the bare file wins so user-built symlink
    trees are respected and no warning fires.
    """
    # Use 1 train + 1 test to satisfy list_stimulus_paths' requirement
    # that both split directories are populated.
    _make_stimuli_tree(tmp_path, n_train=1, n_test=1)
    frames_root = tmp_path / "stimulus_set" / "frames_middle"
    frames_root.mkdir(parents=True, exist_ok=True)
    (frames_root / "0001_45_90.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (frames_root / "0001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (frames_root / "0002_45_90.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    frames = middle_frame_paths(tmp_path)
    assert len(frames) == 2
    # Stem 0001 has a bare form; the helper must prefer it over the glob.
    assert frames[0].name == "0001.jpg"
    assert frames[1].name == "0002_45_90.jpg"


def test_middle_frame_paths_warns_on_ambiguous_glob(tmp_path: Path, caplog):
    """Two CSAIL-style files for the same stem is a data bug; pick the
    first sorted match and warn so the surprise is visible.
    """
    import logging as stdlib_logging
    _make_stimuli_tree(tmp_path, n_train=1, n_test=1)
    frames_root = tmp_path / "stimulus_set" / "frames_middle"
    frames_root.mkdir(parents=True, exist_ok=True)
    (frames_root / "0001_30_60.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (frames_root / "0001_45_90.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (frames_root / "0002_45_90.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    with caplog.at_level(stdlib_logging.WARNING,
                         logger="cortexlab.data.studies.lahner2024bold"):
        frames = middle_frame_paths(tmp_path)
    assert frames[0].name == "0001_30_60.jpg"   # sorted pick for stem 0001
    assert any("multiple middle-frame" in r.message for r in caplog.records)
