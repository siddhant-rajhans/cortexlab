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
import pickle as pkl
import typing as tp
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
from neuralset.events import study
from neuralset.utils import get_bids_filepath, get_masked_bold_image, read_bids_events


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
