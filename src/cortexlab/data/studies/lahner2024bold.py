# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""BOLD Moments: 3T fMRI responses to short naturalistic videos.

This study provides 3T BOLD fMRI data from 10 participants viewing brief (3-second) 
naturalistic video clips. The dataset is designed to study neural responses to 
dynamic visual events and includes rich metadata and annotations. The test set's high 
repetition count (10 reps) enables reliability analysis and within-subject 
generalization studies.

Experimental Design:
    - 3T fMRI recordings (TR = 1.75 seconds)
    - 10 participants
    - 4 functional scanning sessions per subject (sessions 2-5)
    - Two sets of stimuli:
        * Training set: 1,000 unique 3-second video clips (10 runs)
        * Test set: 102 unique 3-second video clips (3 runs, 10 repetitions each)
    - Paradigm: passive viewing of naturalistic video clips
        - Oddball trials included for attention monitoring (excluded from analysis)

Data Format:
    - BIDS-compliant dataset structure
    - fMRIPrep preprocessed data (version B recommended by authors)
    - Available in multiple spaces:
        * MNI152NLin2009cAsym (volumetric)
        * T1w (subject-native volumetric)
        * fsaverage (cortical surface, 163842 vertices per hemisphere)
        * fsnative (subject-specific cortical surface)
    - Pre-computed GLM betas available for fsaverage space
    - Video stimuli
    - Event annotations:
        *  LLM-generated captions for middle frames of each video

Download Requirements:
    - openneuro-py for fMRI data download
    - Stimuli downloaded from boldmomentsdataset.csail.mit.edu
    - Moderate dataset size (~several GB)
    - moviepy required for video processing
"""

import json
import os
import pickle as pkl
import typing as tp
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
from neuralset.events import study
from neuralset.utils import get_bids_filepath, get_masked_bold_image, read_bids_events


STIMULI_SUBPATH: tp.Final[str] = "stimuli/stimulus_set/stimuli"
"""Path inside a BOLD Moments data root that holds the ``train/`` and ``test/`` video directories."""


def list_stimulus_paths(
    root: str | os.PathLike | None = None,
    split: str | None = None,
    suffix: str = ".mp4",
) -> list[Path]:
    """Return stimulus video paths for the BOLD Moments dataset.

    The returned list is the canonical order used throughout CortexLab
    for stimulus indexing: training clips first (sorted by filename),
    then test clips (sorted by filename). Downstream code (feature
    extraction, encoding, alignment benchmarks) assumes this order.

    Parameters
    ----------
    root
        Path to the BOLD Moments data root, i.e. the directory that
        contains ``stimuli/stimulus_set/stimuli/{train,test}/*.mp4``.
        When ``None``, falls back to the ``CORTEXLAB_DATA`` environment
        variable, which should point at the dataset root on the target
        cluster (see ``scripts/slurm/env_setup.sh``).
    split
        ``"train"``, ``"test"``, or ``None`` for both (default).
    suffix
        File suffix to match. BOLD Moments ships with ``.mp4`` clips;
        expose the parameter so callers can adapt if they re-encode
        the stimuli.

    Returns
    -------
    list[Path]
        Sorted stimulus paths.

    Raises
    ------
    FileNotFoundError
        If the expected ``stimuli/stimulus_set/stimuli`` directory or a
        requested split directory is missing, or if no files match the
        suffix. The error message includes the resolved root so the
        underlying cluster-path typo is easy to spot.

    Examples
    --------
    >>> paths = list_stimulus_paths("/scratch/me/bold_moments")  # doctest: +SKIP
    >>> len(paths)                                                # doctest: +SKIP
    1102
    >>> paths[0].name                                             # doctest: +SKIP
    '0001_aerobics.mp4'
    """
    root = _resolve_root(root)
    stim_root = root / STIMULI_SUBPATH
    if not stim_root.is_dir():
        raise FileNotFoundError(
            f"expected BOLD Moments stimuli under {stim_root} (resolved from {root}); "
            "is the dataset staged in CORTEXLAB_DATA?"
        )

    if split is not None and split not in ("train", "test"):
        raise ValueError(f"split must be 'train', 'test', or None; got {split!r}")

    splits = ("train", "test") if split is None else (split,)
    out: list[Path] = []
    for sp in splits:
        sp_dir = stim_root / sp
        if not sp_dir.is_dir():
            raise FileNotFoundError(
                f"missing split directory {sp_dir}; staging for BOLD Moments incomplete."
            )
        paths = sorted(sp_dir.glob(f"*{suffix}"))
        if not paths:
            raise FileNotFoundError(
                f"no *{suffix} files in {sp_dir}. If stimuli were re-encoded, "
                "pass the new suffix via list_stimulus_paths(..., suffix=...)"
            )
        out.extend(paths)
    return out


def _resolve_root(root: str | os.PathLike | None) -> Path:
    if root is None:
        env_root = os.environ.get("CORTEXLAB_DATA")
        if not env_root:
            raise RuntimeError(
                "root was not provided and CORTEXLAB_DATA is unset. "
                "Either pass root=/path/to/bold_moments or export CORTEXLAB_DATA."
            )
        root = env_root
    p = Path(root)
    if not p.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {p}")
    # Allow callers to pass either the dataset root or the parent folder
    # holding a `bold_moments/` subdirectory.
    candidate = p / "bold_moments"
    if candidate.is_dir() and not (p / STIMULI_SUBPATH).is_dir():
        return candidate
    return p


BETAS_SUBPATH: tp.Final[str] = "derivatives/versionB/fsaverage/GLM"
"""Relative path under the dataset root containing the per-subject prepared betas."""

N_TRAIN_STIMULI: tp.Final[int] = 1000
N_TEST_STIMULI: tp.Final[int] = 102
N_VERTICES_PER_HEMI: tp.Final[int] = 163842
"""fsaverage7 surface vertex count per hemisphere."""


def load_subject(
    subject_id: int,
    root: str | os.PathLike | None = None,
    feature_cache: str | os.PathLike | None = None,
    modalities: tp.Sequence[str] = ("vision", "text"),
    parcellation: tp.Mapping[str, np.ndarray] | None = None,
    n_trimmed_stimuli: int | None = None,
) -> dict[str, tp.Any]:
    """Load one subject's BOLD Moments betas together with matching features.

    This is the thin glue between the on-disk GLMsingle prepared-betas pickle
    format and the dict interface consumed by
    :func:`cortexlab.analysis.lesion.run_modality_lesion` and other encoding
    experiments in the repository.

    Parameters
    ----------
    subject_id
        Subject index, 1 through 10.
    root
        Path to the BOLD Moments dataset root. Falls back to ``CORTEXLAB_DATA``
        when None. See :func:`list_stimulus_paths` for the resolution rules.
    feature_cache
        Directory containing one ``<modality>.npz`` per entry in ``modalities``.
        Each file must have a ``features`` array of shape ``(1102, d_m)`` whose
        row order matches :func:`list_stimulus_paths` (train clips 0001-1000
        then test clips 0001-0102, both sorted by filename). When None, the
        returned dict carries empty feature dicts and only the fMRI betas are
        populated; callers can then attach their own features.
    modalities
        Which modality keys to load from ``feature_cache``. Each name must
        correspond to a ``<name>.npz`` file when ``feature_cache`` is set.
    parcellation
        Optional mapping from ROI name to a numpy array of vertex indices into
        the concatenated hemispheres (left hemisphere first, 0 through 163841;
        right hemisphere second, 163842 through 327683). When None the function
        returns a single ``{"all_cortex": arange(n_voxels)}`` entry, which lets
        downstream aggregation work even without a parcellation. Users with an
        HCP-MMP atlas or custom ROIs should supply it here.
    n_trimmed_stimuli
        For pilot runs, keep only the first N training stimuli to cut fit
        time. ``None`` keeps all 1000. Test split is never trimmed.

    Returns
    -------
    dict
        Keys:

        * ``subject_id`` (int)
        * ``y_train`` ``(n_train, 2*163842)``, mean across repetitions
        * ``y_test``  ``(102, 2*163842)``, mean across 10 repetitions
        * ``features_train`` ``{modality: (n_train, d_m)}``
        * ``features_test``  ``{modality: (102, d_m)}``
        * ``stimulus_ids_train`` list of str, from the GLMsingle manifest
        * ``stimulus_ids_test``  list of str
        * ``roi_indices`` ``{roi_name: int_array}``

    Raises
    ------
    FileNotFoundError
        If a required beta file or feature cache file is missing.
    ValueError
        If the betas and features disagree about stimulus count.

    Notes
    -----
    The pickle layout shipped by the BOLD Moments authors is a 2-tuple
    ``(betas, stim_names)`` where ``betas`` has shape
    ``(n_trials, n_reps, 163842)``. We average across the repetition axis
    because downstream encoders regress on per-stimulus responses. If you want
    per-trial analysis (e.g. reliability estimation), read the pickles
    directly with :mod:`pickle` rather than this helper.
    """
    root_path = _resolve_root(root)
    betas_root = root_path / BETAS_SUBPATH / f"sub-{subject_id:02d}" / "prepared_betas"
    if not betas_root.is_dir():
        raise FileNotFoundError(
            f"expected prepared betas under {betas_root}. "
            "Run the OpenNeuro download for ds005165 (versionB/fsaverage/GLM) first."
        )

    splits = [("train", N_TRAIN_STIMULI), ("test", N_TEST_STIMULI)]
    y: dict[str, np.ndarray] = {}
    stimulus_ids: dict[str, list[str]] = {}

    for split, n_expected in splits:
        hemi_arrays = []
        stim_ref: list[str] | None = None
        for hemi in ("left", "right"):
            fp = (
                betas_root
                / f"sub-{subject_id:02d}_organized_betas_task-{split}_hemi-{hemi}_normalized.pkl"
            )
            if not fp.exists():
                raise FileNotFoundError(f"missing beta file {fp}")
            with fp.open("rb") as f:
                obj = pkl.load(f)
            betas, stims = obj[0], obj[1]
            if betas.shape[0] != n_expected:
                raise ValueError(
                    f"{fp.name}: expected {n_expected} trials, got {betas.shape[0]}"
                )
            if betas.shape[-1] != N_VERTICES_PER_HEMI:
                raise ValueError(
                    f"{fp.name}: expected {N_VERTICES_PER_HEMI} vertices, "
                    f"got {betas.shape[-1]}"
                )
            # Mean across repetition axis -> (n_trials, n_vertices).
            hemi_arrays.append(np.asarray(betas, dtype=np.float32).mean(axis=1))
            stim_list = [str(s) for s in stims]
            if stim_ref is None:
                stim_ref = stim_list
            elif stim_list != stim_ref:
                raise ValueError(
                    f"{fp.name}: stimulus order disagrees between hemispheres"
                )
        assert stim_ref is not None
        y[split] = np.concatenate(hemi_arrays, axis=1)
        stimulus_ids[split] = stim_ref

    # Optional pilot truncation on training split.
    if n_trimmed_stimuli is not None and n_trimmed_stimuli < y["train"].shape[0]:
        y["train"] = y["train"][:n_trimmed_stimuli]
        stimulus_ids["train"] = stimulus_ids["train"][:n_trimmed_stimuli]

    # Feature loading. The canonical stimulus order for feature caches is
    # `list_stimulus_paths()` (train 0001-1000 sorted, then test 0001-0102
    # sorted). Load once, split by the row counts dictated by the betas.
    features_train: dict[str, np.ndarray] = {}
    features_test: dict[str, np.ndarray] = {}
    if feature_cache is not None:
        cache = Path(feature_cache)
        if not cache.is_dir():
            raise FileNotFoundError(f"feature cache directory not found: {cache}")
        n_train = y["train"].shape[0]
        n_test = y["test"].shape[0]
        for modality in modalities:
            fp = cache / f"{modality}.npz"
            if not fp.exists():
                raise FileNotFoundError(
                    f"feature file {fp} missing; expected an npz with key "
                    "'features' of shape (1102, d) in list_stimulus_paths order."
                )
            arr = np.load(fp)["features"]
            if arr.shape[0] != N_TRAIN_STIMULI + N_TEST_STIMULI:
                raise ValueError(
                    f"{fp.name}: expected {N_TRAIN_STIMULI + N_TEST_STIMULI} rows, "
                    f"got {arr.shape[0]}"
                )
            # Training split was possibly trimmed; test is always the full 102.
            features_train[modality] = np.asarray(arr[:n_train], dtype=np.float32)
            features_test[modality] = np.asarray(
                arr[N_TRAIN_STIMULI : N_TRAIN_STIMULI + n_test], dtype=np.float32,
            )

    n_voxels = y["train"].shape[1]
    if parcellation is None:
        roi_indices: dict[str, np.ndarray] = {
            "all_cortex": np.arange(n_voxels, dtype=np.int64),
        }
    else:
        roi_indices = {k: np.asarray(v, dtype=np.int64) for k, v in parcellation.items()}

    return {
        "subject_id": int(subject_id),
        "y_train": y["train"],
        "y_test": y["test"],
        "features_train": features_train,
        "features_test": features_test,
        "stimulus_ids_train": stimulus_ids["train"],
        "stimulus_ids_test": stimulus_ids["test"],
        "roi_indices": roi_indices,
    }


class Lahner2024Bold(study.Study):
    device: tp.ClassVar[str] = "Fmri"
    dataset_name: tp.ClassVar[str] = "BOLD Moments"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{Lahner2024,
        title = {Modeling short visual events through the BOLD moments video fMRI dataset and metadata},
        volume = {15},
        ISSN = {2041-1723},
        url = {http://dx.doi.org/10.1038/s41467-024-50310-3},
        DOI = {10.1038/s41467-024-50310-3},
        number = {1},
        journal = {Nature Communications},
        publisher = {Springer Science and Business Media LLC},
        author = {Lahner,  Benjamin and Dwivedi,  Kshitij and Iamshchinina,  Polina and Graumann,  Monika and Lascelles,  Alex and Roig,  Gemma and Gifford,  Alessandro Thomas and Pan,  Bowen and Jin,  SouYoung and Ratan Murty,  N. Apurva and Kay,  Kendrick and Oliva,  Aude and Cichy,  Radoslaw},
        year = {2024},
        month = jul 
    }
    """
    licence: tp.ClassVar[str] = "CC0"
    description: tp.ClassVar[str] = (
        "BOLD Moments: 3T fMRI from 10 participants viewing 1,000+ brief "
        "(3-second) naturalistic videos"
    )

    requirements: tp.ClassVar[tuple[str, ...]] = ("moviepy==2.0.0.dev2",)

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=520,
        num_subjects=10,
        num_events_in_query=76,
        event_types_in_query={"Fmri", "Video"},
        data_shape=(62, 77, 61, 238),
        frequency=0.571,
        fmri_spaces=("custom",),
    )

    NUM_SUBJECTS: tp.ClassVar[int] = 10
    NUM_RUNS_PER_SPLIT: tp.ClassVar[dict[str, int]] = {"train": 10, "test": 3}

    DERIVATIVES_FOLDER: tp.ClassVar[str] = "download/derivatives/versionB/fmriprep"
    SPACES: tp.ClassVar[tuple[str, ...]] = (
        "MNI152NLin2009cAsym",
        "T1w",
        "fsaverage",
        "fsnative",
    )

    N_TRIALS_TRAIN: tp.ClassVar[int] = 1000
    N_TRIALS_TEST: tp.ClassVar[int] = 102
    N_VOLUMES_TRAIN: tp.ClassVar[int] = 238
    N_VOLUMES_TEST: tp.ClassVar[int] = 268
    TR_FMRI_S: tp.ClassVar[float] = 1.75

    def _download(self) -> None:
        raise NotImplementedError("Download method not implemented yet")

    def _validate_downloaded_data(self) -> None:
        postfixs = [
            "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            "_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
            "_hemi-R_space-fsaverage_bold.func.gii",
            "_hemi-L_space-fsaverage_bold.func.gii",
        ]

        for tl in self.iter_timelines():
            subj, ses, split, run = tl["subject"], tl["session"], tl["split"], tl["run"]
            for postfix in postfixs:
                fp = self.path / (
                    f"sub-{subj:02d}/ses-{ses:02d}/func/sub-{subj:02d}"
                    f"_ses-{ses:02d}_task-{split}_run-{run:01d}{postfix}"
                )
                if not fp.exists():
                    msg = f"{fp} is missing. Please download again"
                    raise RuntimeError(msg)

        for subj in range(1, self.NUM_SUBJECTS + 1):
            betas_root = (
                self.path / "download/derivatives/versionB/fsaverage/GLM/"
                f"sub-{subj:02}/prepared_betas/"
            )
            for split in ("train", "test"):
                for hemi in ("left", "right"):
                    fp = (
                        betas_root / f"sub-{subj:02}_organized_betas_task-{split}"
                        f"_hemi-{hemi}_normalized.pkl"
                    )
                    if not fp.exists():
                        msg = f"{fp} is missing. Please download again"
                        raise RuntimeError(msg)
                    with fp.open("rb") as f:
                        prepared_betas = pkl.load(f)
                        betas = prepared_betas[0]
                        n_trials = (
                            self.N_TRIALS_TEST
                            if split == "test"
                            else self.N_TRIALS_TRAIN
                        )
                        n_reps = 10 if split == "test" else 3
                        betas_shape = (n_trials, n_reps, 163842)
                        if betas.shape != betas_shape:
                            msg = f"Expected {betas_shape}, got {betas.shape}"
                            raise RuntimeError(msg)
                        stims = prepared_betas[1]
                        if len(stims) != n_trials:
                            msg = f"Expected {n_trials} stimuli, got {len(stims)}"
                            raise RuntimeError(msg)

        root = self.path / "stimuli/stimulus_set/stimuli/"
        for split in ("train", "test"):
            num_expected = (
                self.N_TRIALS_TRAIN if split == "train" else self.N_TRIALS_TEST
            )
            num_found = len(list((root / split).iterdir()))
            if num_found != num_expected:
                msg = f"Expecting {num_expected} stimuli for split {split}"
                msg += f" but found {num_found}. Please download again"
                raise RuntimeError(msg)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for subj in range(1, self.NUM_SUBJECTS + 1):
            for ses in (2, 3, 4, 5):
                for split, n_runs in self.NUM_RUNS_PER_SPLIT.items():
                    for run in range(1, n_runs + 1):
                        yield dict(subject=subj, session=ses, split=split, run=run)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        tl = dict(timeline)
        split = tl.pop("split")
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        n_vols = self.N_VOLUMES_TRAIN if split == "train" else self.N_VOLUMES_TEST
        fmri = {
            "filepath": info,
            "type": "Fmri",
            "start": 0.0,
            "frequency": 1.0 / self.TR_FMRI_S,
            "duration": n_vols * self.TR_FMRI_S,
        }
        bids_events_df_fp = get_bids_filepath(
            root_path=self.path / "download",
            filetype="events",
            data_type="Fmri",
            run_padding="01",
            task=split,
            **tl,
        )
        bids_events_df = read_bids_events(bids_events_df_fp)

        bids_events_df = bids_events_df[bids_events_df.trial_type != "oddball"]
        ns_events_df = self._get_ns_img_events_df(bids_events_df, timeline)
        return pd.concat([pd.DataFrame([fmri]), ns_events_df], axis=0)

    def _load_raw(
        self, timeline: dict[str, tp.Any], space: str = "MNI152NLin2009cAsym"
    ) -> nibabel.Nifti2Image | nibabel.Nifti1Image:
        if space in ["MNI152NLin2009cAsym", "T1w"]:
            return get_masked_bold_image(*self._get_bold_images(timeline, space))
        elif space in ["fsnative", "fsaverage"]:
            return self._get_fs(timeline, space)
        msg = f"{space} is not supported."
        raise ValueError(msg)

    def _get_ns_img_events_df(
        self, bids_events_df: pd.DataFrame, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        path_to_stimuli = self.path / "stimuli/stimulus_set/stimuli"

        annot_path = (
            self.path
            / "download/derivatives/stimuli_metadata/llm_frame_annotations.json"
        )
        with annot_path.open("r", encoding="utf8") as f:
            middle_frame_captions = json.load(f)

        bids_events = bids_events_df.to_dict("records")
        ns_events = []
        for bids_event in bids_events:
            fp = Path(bids_event["stim_file"])
            filepath = str(path_to_stimuli / fp)
            captions = "\n".join(next(iter(middle_frame_captions[fp.stem].values())))
            ns_event = dict(
                type="Video",
                start=bids_event["onset"],
                filepath=filepath,
                middle_frame_captions=captions,
            )
            ns_events.append(ns_event)
        return pd.DataFrame(ns_events)

    def _get_bold_images(self, timeline: dict[str, tp.Any], space: str):
        timeline = dict(timeline)
        timeline["task"] = timeline.pop("split")
        kwargs = {
            "root_path": self.path / self.DERIVATIVES_FOLDER,
            "data_type": "Fmri",
            "space": space,
            "run_padding": "01",
            **timeline,
        }
        bold = nibabel.load(get_bids_filepath(**kwargs, filetype="bold"), mmap=True)
        mask = nibabel.load(
            get_bids_filepath(**kwargs, filetype="bold_mask"), mmap=True
        )
        return (bold, mask)

    def _get_fs(
        self, timeline: dict[str, tp.Any], space: str = "fsaverage"
    ) -> nibabel.Nifti2Image:
        tl = timeline
        if space not in ["fsaverage", "fsnative"]:
            msg = f"{space} is not supported. " "Only surfaces 'fsaverage' "
            msg += "and 'fsnative' are supported for Lahner2024Bold."
            raise ValueError(msg)

        data = []
        n_volumes = (
            self.N_VOLUMES_TRAIN if tl["split"] == "train" else self.N_VOLUMES_TEST
        )
        for hemi in ("L", "R"):
            fp = (
                self.path
                / self.DERIVATIVES_FOLDER
                / f"sub-{int(tl['subject']):02}/ses-{tl['session']:02}"
                / f"func/sub-{int(tl['subject']):02}_ses-{tl['session']:02}_task-{tl['split']}"
                f"_run-{tl['run']}_hemi-{hemi}_space-{space}_bold.func.gii"
            )
            hemi_data = nibabel.load(fp, mmap=True).darrays  # type: ignore
            if len(hemi_data) != n_volumes:
                msg = f"Expected {n_volumes} volumes, got {len(hemi_data)}"
                raise RuntimeError(msg)
            if space == "fsaverage" and hemi_data[0].data.shape != (163842,):
                msg = f"Expected shape (163842,), got {hemi_data[0].data.shape}"
                raise RuntimeError(msg)
            np_data = np.stack([darray.data for darray in hemi_data], -1)
            data.append(np_data)
        data = np.concatenate(data, axis=0)
        return nibabel.Nifti2Image(data, np.eye(4))
