# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Natural language fMRI dataset: 3T fMRI responses to spoken narrative stories.

This dataset provides fMRI data from participants listening to natural spoken 
narratives (stories) during 3T scanning. The stimuli include various narrative 
audio stories with detailed word-level and phoneme-level annotations. The dataset 
is designed for studying natural language processing in the brain.

Experimental Design:
    - 3T fMRI recordings (TR = 2.0 seconds)
    - 8 subjects (UTS01-UTS08)
    - Subjects 1-3: 82 stories across 20 sessions (extended dataset)
    - Subjects 4-8: 26-27 stories across 6 sessions
    - Paradigm: passive listening to naturalistic spoken narratives
        * Audio narratives with 10-second blank period before story onset
        * Test story: "wheretheressmoke" (with 10 runs)
        * Training stories: diverse narrative content
    - Localizer tasks included: AudioMotorLocalizer, AuditoryLocalizer, 
      CategoryLocalizer, MotorLocalizer

Data Format:
    - BIDS-compliant dataset structure (OpenNeuro ds003020)
    - Two preprocessing versions available (see Study Classes below)
    - Audio files: WAV format
    - Event annotations (from TextGrid files)
        * Word-level timing and text
        * Phoneme-level timing and text
        * Audio file paths

Study Classes:
    1. **Lebel2023Bold**: Uses deepprep preprocessing pipeline
        - Available spaces: T1w, MNI152NLin6Asym, fsaverage, fsnative
        - 432 timelines (all sessions/runs)
        - Full BIDS structure with multiple space outputs
        
    2. **LebelProcessed2023Bold**: Uses authors' custom HDF5 preprocessing
        - Custom cortical surface registration
        - 200 timelines (aggregated by subject x task)
        - Data stored in HDF5 format (.hf5 files)
        - Custom voxel selection and masking

Download Requirements:
    - OpenNeuro dataset: ds003020
    - Dataset includes both raw fMRI data and preprocessed derivatives
    - Audio stimuli (.wav files) and TextGrid annotations included
    - Deepprep derivatives for Lebel2023Bold
    - HDF5 preprocessed data for LebelProcessed2023Bold
    - Python packages:
        * nltk (v3.8.1) for TextGrid parsing
        * nltk_contrib (from GitHub) for TextGrid file format
        * soundfile (>=0.13.1) for audio handling
        * h5py (>=3.10.0) for HDF5 files (LebelProcessed2023Bold only)
        * pycortex (for cortical surface visualization, LebelProcessed2023Bold only)

Issues and Considerations:
    - Subject UTS02: Different scan location and protocol, no localizer data
    - Subject UTS04: Missing "life.hf5" story scan
    - Subject UTS05: Low visual acuity, presented auditory cues
    - UTS01/ses-7/treasureisland: Corrupted NIfTI file, automatically skipped
    - Preprocessed data has additional 20s removed from beginning
    - Original preprocessing: https://github.com/HuthLab/deep-fMRI-dataset
"""

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
from neuralset.events import study

logger = logging.getLogger(__name__)

_DEFAULT_BAD_WORDS = frozenset(
    [
        "sentence_start",
        "sentence_end",
        "br",
        "lg",
        "ls",
        "ns",
        "sp",
        "{BR}",
        "{LG}",
        "{LS}",
        "{NS}",
        "{SP}",
    ]
)

_ANAT_TASKS = [
    "AudioMotorLocalizer",
    "AuditoryLocalizer",
    "CategoryLocalizer",
    "MotorLocalizer",
]

SUBJECTS = [f"UTS{i:02d}" for i in range(1, 9)]


def _get_audio_file(path: Path | str, task: str) -> Path:
    path = Path(path)
    return path / f"stimuli/{task}.wav"


def _get_audio_text_file(path: Path | str, task: str) -> Path:
    path = Path(path)
    return path / f"derivative/TextGrids/{task}.TextGrid"


def _create_audio_events(path: Path | str, task: str) -> list[dict]:
    events = []
    dl_path = Path(path)
    audio_text_file_name = _get_audio_text_file(dl_path, task)
    audio_wav_file_name = _get_audio_file(dl_path, task)

    split = "train" if task != "wheretheressmoke" else "test"

    events.append(
        dict(
            start=0.0,
            type="Audio",
            language="english",
            filepath=str(audio_wav_file_name),
            split=split,
        )
    )

    from nltk_contrib.textgrid import TextGrid

    data = audio_text_file_name.read_text(encoding="utf-8")
    fid = TextGrid(data)

    for _, tier in enumerate(fid):
        for recording in tier.simple_transcript:
            start, stop, text = recording
            if text != "" and text not in _DEFAULT_BAD_WORDS:
                if tier.nameid == "phone":
                    tier_type = "Phoneme"
                elif tier.nameid == "word":
                    tier_type = "Word"
                else:
                    msg = "Tier must either be phone or word but tier.nameid is %s"
                    logger.warning(msg, tier.nameid)
                events.append(
                    dict(
                        start=float(start),
                        text=text.lower(),
                        duration=float(stop) - float(start),
                        type=tier_type,
                        language="english",
                        filepath=str(audio_wav_file_name),
                        split=split,
                    )
                )

    return events


def _get_preprocessed_responses(
    path: Path | str, task: str, subject: str
) -> np.ndarray:
    output = _get_response(Path(path), [task], subject)
    return output


def _get_hf5_path(path: Path | str, subject: str, task: str) -> Path | None:
    path = Path(path).resolve()
    hf5_path = path / "derivative" / "preprocessed_data" / subject / f"{task}.hf5"
    if hf5_path.exists():
        return hf5_path
    return None


def _get_tasks(path: Path) -> list[str]:
    path = Path(path).resolve()
    dl_path = path / "stimuli"
    tasks = []
    for fp in dl_path.glob("*.wav"):
        tasks.append(fp.stem)
    return tasks


def _get_response(path: Path | str, stories, subject) -> np.ndarray:
    """Get the subject"s fMRI response for stories."""
    import h5py

    path = Path(path).resolve()
    base_path = path / f"download/ds003020/derivative/preprocessed_data/{subject}"
    resp = []
    for story in stories:
        resp_path = base_path / f"{story}.hf5"
        hf = h5py.File(resp_path, "r")
        resp.extend(hf["data"][:])
        hf.close()
    return np.array(resp)


class Lebel2023Bold(study.Study):
    device: tp.ClassVar[str] = "Fmri"
    licence: tp.ClassVar[str] = "CC0"
    description: tp.ClassVar[str] = (
        "Natural language fMRI: 3T fMRI responses from 8 subjects listening to "
        "spoken narrative stories. Deepprep preprocessing with multiple output spaces "
        "(T1w, MNI152NLin6Asym, fsaverage, fsnative). 432 timelines with word and "
        "phoneme-level annotations. Test story: 'wheretheressmoke'."
    )
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{lebel2023natural,
        title={A natural language fMRI dataset for voxelwise encoding models},
        author={LeBel, Amanda and Wagner, Lauren and Jain, Shailee and Adhikari-Desai, Aneesh and Gupta, Bhavin and Morgenthal, Allyson and Tang, Jerry and Xu, Lixiang and Huth, Alexander G},
        journal={Scientific Data},
        volume={10},
        number={1},
        pages={555},
        year={2023},
        publisher={Nature Publishing Group UK London},
        doi={https://doi.org/10.1038/s41597-023-02437-z},
        url={https://www.nature.com/articles/s41597-023-02437-z}
    }

    @dataset{lebel2023bold,
        title={A natural language fMRI dataset for voxelwise encoding models},
        author={LeBel, Amanda and Wagner, Lauren and Jain, Shailee and Adhikari-Desai, Aneesh and
                Gupta, Bhavin and Morgenthal, Alyssa and Tang, Jerry and Xu, Lixiang and Huth, Alexander G},
        year={2023},
        publisher={OpenNeuro},
        doi={10.18112/openneuro.ds003020.v2.2.0},
        url={https://openneuro.org/datasets/ds003020}
    }
    """
    requirements: tp.ClassVar[tuple[str, ...]] = (
        "nltk==3.8.1",
        "git+https://github.com/nltk/nltk_contrib.git@683961c53f0c122b90fe2d039fe795e0a2b3e997",
        "soundfile>=0.13.1",
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=432,
        num_subjects=8,
        num_events_in_query=9199,
        event_types_in_query={"Fmri", "Audio", "Word", "Phoneme"},
        data_shape=(57, 65, 56, 363),
        frequency=0.5,
        fmri_spaces=("T1w", "MNI152NLin6Asym", "fsaverage", "fsnative"),
    )
    TR_FMRI_S: tp.ClassVar[float] = 2.0
    DERIVATIVES_FOLDER: tp.ClassVar[str] = "download/ds003020-fmriprep"

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        self.infra_timelines.version = "v3.4"

    def _download(self) -> None:
        raise NotImplementedError("Download method not implemented yet")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """
        Iterate over the different recording timelines:
        e.g. subjects x sessions in order with fmri runs
        """
        dl_dir = self.path / "download/ds003020"
        if not dl_dir.exists():
            raise RuntimeError(f"Missing folder {dl_dir}")

        for subject in SUBJECTS:
            sessions = 20 if subject in ["UTS01", "UTS02", "UTS03"] else 6

            for sess in range(1, sessions + 1):
                sess_dir = dl_dir / f"sub-{subject}" / f"ses-{sess}" / "func"
                tasks = [task.name for task in sess_dir.glob("*_bold.nii.gz")]
                tasks = sorted({task.split("_")[2].split("-")[1] for task in tasks})
                for task in tasks:
                    if task.startswith(tuple(_ANAT_TASKS)):
                        continue
                    if subject == "UTS01" and sess == 7 and task == "treasureisland":
                        msg = "Skipping subject=UTS01, session=7, task=treasureisland as nii.gz is corrupted."
                        logger.warning(msg)
                        continue

                    runs = (
                        list(range(1, 11)) + [None]
                        if task == "wheretheressmoke"
                        else [None]
                    )
                    for run in runs:
                        run_infix = f"_run-{run}" if run is not None else ""
                        filename = f"sub-{subject}_ses-{sess}_task-{task}{run_infix}_bold.nii.gz"
                        bids_path = sess_dir / filename
                        if not bids_path.exists():
                            continue

                        audio_text_file = _get_audio_text_file(path=dl_dir, task=task)
                        if not audio_text_file.exists():
                            raise RuntimeError(
                                f"Missing audio text file: {audio_text_file}"
                            )
                        audio_file = _get_audio_file(path=dl_dir, task=task)
                        if not audio_file.exists():
                            raise RuntimeError(f"Missing audio file: {audio_file}")

                        yield dict(
                            subject=subject, session=str(sess), task=task, run=run
                        )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Reads the events of a given timeline"""

        task = timeline["task"]
        freq = 1.0 / self.TR_FMRI_S
        events = _create_audio_events(self.path / "download/ds003020", task)
        subject, session, task, run = (
            timeline["subject"],
            timeline["session"],
            timeline["task"],
            timeline["run"],
        )
        run_substr = f"_run-{run}" if run is not None else ""
        fp = (
            self.path
            / self.DERIVATIVES_FOLDER
            / f"sub-{subject}/ses-{session}/func"
            / f"sub-{subject}_ses-{session}_task-{task}{run_substr}_*"
        )
        events.append(
            dict(
                type="Fmri",
                start=0.0,
                filepath=fp,
                layout="fmriprep",
                frequency=freq,
                split="train" if task != "wheretheressmoke" else "test",
            )
        )
        out = pd.DataFrame(events)
        out.loc[out.type != "Fmri", "start"] += 10
        out["task"] = task
        out.loc[out.type != "Fmri", "modality"] = "heard"
        return out
