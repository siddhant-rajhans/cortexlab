# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Algonauts Project 2025 Challenge: fMRI responses to multimodal movie stimuli.

This study is part of the Algonauts Project 2025 Challenge, using a subset of the
Courtois NeuroMod dataset (https://www.cneuromod.ca/). Participants watched naturalistic
video stimuli including episodes from the TV sitcom "Friends" and extractor films while
undergoing fMRI scanning.

Experimental Design:
    - 4 participants (sub-01, sub-02, sub-03, sub-05)
    - Two stimulus types:
        * "Friends" sitcom: 7 seasons, ~175 episodes, segmented into ~5min chunks (a,b,c,d)
        * "movie10": 4 extractor films (Bourne, Wolf, Life, Figures) in ~5min chunks
    - TR = 1.49 seconds
    - Training data: Friends seasons 1-6, all movies
    - Test data: Friends season 7
    - Some movies shown twice (Life, Figures) for reliability analysis

Data Format:
    - Preprocessed fMRI in MNI152NLin2009cAsym space
    - Parcellated using Schaefer-1000 atlas (1000 parcels, 7 networks)
    - HDF5 format
    - Video stimuli provided as .mkv files
    - Word-level transcripts with timestamps (.tsv format)
    - Includes rich multimodal annotations (speech, text, visual extractors)

Download Requirements:
    - Datalad must be installed (pip install datalad)
    - Git must be configured
    - Dataset cloned from: https://github.com/courtois-neuromod/algonauts_2025.competitors.git
    - Moderate dataset size (~several GB)

Note:
    This dataset is designed for the Algonauts 2025 Challenge focused on predicting
    brain responses to complex, naturalistic multimodal stimuli.
    See: https://algonautsproject.com/2025/index.html
"""

import ast
import logging
import typing as tp
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from neuralset.events import study

logger = logging.getLogger(__name__)


class Algonauts2025(study.Study):
    _SUBJECTS: tp.ClassVar[list[str]] = ["sub-01", "sub-02", "sub-03", "sub-05"]
    _TASKS: tp.ClassVar[list[str]] = ["friends", "movie10"]
    _SPACE: tp.ClassVar[str] = "space-MNI152NLin2009cAsym"
    _ATLAS: tp.ClassVar[str] = "atlas-Schaefer18_parcel-1000Par7Net"
    _FREQUENCY: tp.ClassVar[float] = 1 / 1.49

    device: tp.ClassVar[str] = "Fmri"
    dataset_name: tp.ClassVar[str] = "Algonauts 2025 Challenge"
    url: tp.ClassVar[str] = "https://algonautsproject.com/"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{algonauts2025,
        url = {https://arxiv.org/abs/2501.00504},
        author = {Gifford,  Alessandro T. and Bersch,  Domenic and St-Laurent,  Marie and Pinsard,  Basile and Boyle,  Julie and Bellec,  Lune and Oliva,  Aude and Roig,  Gemma and Cichy,  Radoslaw M.},
        keywords = {Neurons and Cognition (q-bio.NC),  FOS: Biological sciences,  FOS: Biological sciences},
        title = {The Algonauts Project 2025 Challenge: How the Human Brain Makes Sense of Multimodal Movies},
        publisher = {arXiv},
        year = {2025},
        copyright = {Creative Commons Attribution 4.0 International},
        doi={https://doi.org/10.48550/arXiv.2501.00504},
        url={https://arxiv.org/abs/2501.00504}
    }
    """
    description: tp.ClassVar[str] = (
        'Subset of Courtois NeuroMod dataset (boyle2020) with fMRI recordings of subjects watching videos of a popular sitcom ("Friends") for Algonauts 2025'
    )
    requirements: tp.ClassVar[tuple[str, ...]] = (
        "datalad>=0.19.5",
        "moviepy",
    )

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1588,
        num_subjects=4,
        num_events_in_query=1700,
        event_types_in_query={"Fmri", "Video", "Word", "Text"},
        data_shape=(1000, 592),
        frequency=0.671,
        fmri_spaces=("custom",),
    )

    def _download(self) -> None:
        raise NotImplementedError("Download method not implemented yet")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for subject in self._SUBJECTS:
            for task in self._TASKS:
                if task == "friends":
                    season_episode_chunk = range(1, 8), range(1, 26), "abcd"
                    for season, episode, chunk in product(*season_episode_chunk):
                        tl = dict(
                            subject=subject,
                            task=task,
                            movie=f"s{season:02d}",
                            chunk=f"e{episode:02d}{chunk}",
                            run=0,
                        )
                        stim_path = self._get_transcript_filepath(tl)
                        if (
                            (season == 5 and episode == 20 and chunk == "a")
                            or (season == 4 and episode == 1 and chunk == "a")
                            or (season == 6 and episode == 3 and chunk == "a")
                            or (season == 4 and episode == 13 and chunk == "b")
                            or (season == 4 and episode == 1 and chunk == "b")
                        ):
                            continue
                        if stim_path.exists():
                            yield tl
                elif task == "movie10":
                    movie_chunk_run = (
                        ["bourne", "wolf", "life", "figures"],
                        range(1, 18),
                        [1, 2],
                    )
                    for movie, chunk, run in product(*movie_chunk_run):  # type: ignore
                        if movie in ["bourne", "wolf"] and run == 2:
                            continue
                        tl = dict(
                            subject=subject,
                            task=task,
                            movie=movie,
                            chunk=str(chunk),
                            run=run,
                        )
                        stim_path = self._get_transcript_filepath(tl)
                        if stim_path.exists():
                            yield tl

    def _get_transcript_filepath(self, timeline: dict[str, tp.Any]) -> Path:
        tl = timeline
        base = (
            self.path
            / "download/algonauts_2025.competitors/stimuli/transcripts"
            / tl["task"]
        )
        if tl["task"] == "friends":
            return base / f"s{tl['movie'][-1]}/friends_{tl['movie']}{tl['chunk']}.tsv"
        elif tl["task"] == "movie10":
            return (
                base / f"{tl['movie']}/movie10_{tl['movie']}{int(tl['chunk']):02d}.tsv"
            )
        raise ValueError(f"Unknown task: {tl['task']}")

    def _get_movie_filepath(self, timeline: dict[str, tp.Any]) -> Path:
        tl = timeline
        base = (
            self.path
            / "download/algonauts_2025.competitors/stimuli/movies"
            / tl["task"]
        )
        if tl["task"] == "friends":
            return base / f"s{tl['movie'][-1]}/friends_{tl['movie']}{tl['chunk']}.mkv"
        elif tl["task"] == "movie10":
            return base / f"{tl['movie']}/{tl['movie']}{int(tl['chunk']):02d}.mkv"
        raise ValueError(f"Unknown task: {tl['task']}")

    def _get_fmri_filepath(self, timeline: dict[str, tp.Any]) -> Path:
        tl = timeline
        subj_dir = (
            self.path
            / "download/algonauts_2025.competitors/fmri"
            / tl["subject"]
            / "func"
        )
        stem = f"{tl['subject']}_task-{tl['task']}_{self._SPACE}_{self._ATLAS}"
        suffix = "_desc-s123456_bold.h5" if tl["task"] == "friends" else "_bold.h5"
        return subj_dir / f"{stem}{suffix}"

    def _load_fmri(self, timeline: dict[str, tp.Any]) -> tp.Any:
        import h5py

        tl = timeline
        fmri_file = self._get_fmri_filepath(timeline)
        fmri = h5py.File(fmri_file, "r")
        if tl["task"] == "friends":
            key = f"{tl['movie'][1:]}{tl['chunk']}"
        else:
            key = f"{tl['movie']}{int(tl['chunk']):02d}"
            if tl["movie"] in ["life", "figures"]:
                key += f"_run-{tl['run']}"
        selected_key = [key_ for key_ in fmri.keys() if key in key_]
        if len(selected_key) != 1:
            logger.error(
                "key=%s, selected=%s, available=%s",
                key,
                selected_key,
                list(fmri.keys()),
            )
            raise ValueError(f"Multiple or no keys found, {key}, {list(fmri.keys())}")
        fmri = fmri[selected_key[0]]
        data = fmri[:].astype(np.float32)
        import nibabel

        obj = nibabel.Nifti2Image(data.T, affine=np.eye(4))
        return obj

    def _get_split(self, timeline: dict[str, tp.Any]) -> str:
        tl = timeline
        if tl["task"] == "friends":
            if int(tl["movie"][-1]) in range(1, 7):
                return "train"
            elif int(tl["movie"][-1]) == 7:
                return "test"
        return "train"

    def _get_fmri_event(self, timeline: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Return fmri event dict"""
        info = study.SpecialLoader(method=self._load_fmri, timeline=timeline).to_json()
        return dict(type="Fmri", filepath=info, start=0, frequency=self._FREQUENCY)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        all_events = []
        if (timeline["task"], timeline["movie"]) != ("friends", "s07"):
            all_events.append(self._get_fmri_event(timeline))

        movie_filepath = self._get_movie_filepath(timeline)
        movie_event = dict(type="Video", filepath=str(movie_filepath), start=0)
        all_events.append(movie_event)

        transcript_path = self._get_transcript_filepath(timeline)
        transcript_df = pd.read_csv(transcript_path, sep="\t")
        word_events = []
        for _, row in transcript_df.iterrows():
            words = ast.literal_eval(row["words_per_tr"])
            starts = ast.literal_eval(row["onsets_per_tr"])
            durations = ast.literal_eval(row["durations_per_tr"])
            for word, start, duration in zip(words, starts, durations):
                event = dict(
                    type="Word",
                    text=word,
                    start=start,
                    duration=duration,
                    stop=start + duration,
                    language="english",
                )
                word_events.append(event)
        if word_events:
            word_df = pd.DataFrame(word_events)
            text = " ".join(word_df["text"].tolist())
            text_event = dict(
                type="Text",
                text=text,
                start=word_df["start"].min(),
                duration=word_df["stop"].max() - word_df["start"].min(),
                stop=word_df["stop"].max(),
                language="english",
            )
            all_events.append(text_event)
        all_events.extend(word_events)

        events_df = pd.DataFrame(all_events)
        events_df["split"] = self._get_split(timeline)

        events_df.loc[events_df.type.isin(["Word", "Sentence", "Text"]), "modality"] = (
            "heard"
        )

        return events_df


class Algonauts2025Bold(Algonauts2025):

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1588,
        num_subjects=4,
        num_events_in_query=1700,
        event_types_in_query={"Fmri", "Video", "Word", "Text"},
        data_shape=(76, 90, 71, 592),
        frequency=0.671,
        fmri_spaces=("T1w", "MNI152NLin2009cAsym"),
    )

    def _download(self) -> None:
        raise NotImplementedError("Download method not implemented yet")

    def _get_fmri_event(self, timeline: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Return fmri event dict using fmriprep finder"""
        tl = timeline
        if tl["task"] == "friends":
            task_str = f"{tl['movie']}{tl['chunk']}"
        else:
            task_str = f"{tl['movie']}{int(tl['chunk']):02d}"
        subj_dir = self.path / "download" / f"{tl['task']}.fmriprep" / tl["subject"]
        task_pattern = f"*_task-{task_str}_*"
        for session_dir in sorted(subj_dir.iterdir()):
            if not session_dir.name.startswith("ses-"):
                continue
            func_dir = session_dir / "func"
            if func_dir.exists() and list(func_dir.glob(task_pattern + ".nii.gz")):
                fp = func_dir / task_pattern
                return dict(
                    type="Fmri",
                    filepath=fp,
                    layout="fmriprep",
                    start=0,
                    frequency=self._FREQUENCY,
                )
        raise FileNotFoundError(f"No fMRI file found for {tl}")
