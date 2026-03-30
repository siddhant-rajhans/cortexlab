# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import pandas as pd
from neuralset.events import study


def _get_nii_file(path: Path | str, subject: str, seg: str, fmri_run: int) -> Path:
    path = Path(path)
    seg_dir = path / subject / "fmri" / seg
    nii = seg_dir / "mni" / f"{seg}_{fmri_run}_mni.nii.gz"
    # Outrageously, some test files have a different
    # naming convention...
    if not nii.exists():
        nii = seg_dir / "mni" / f"{seg}_{fmri_run}.mni.nii.gz"
    assert nii.exists(), f"Missing file {nii} for {subject!r} and {seg!r}"
    return nii


def _get_video_file(path: Path | str, seg: str) -> Path:
    path = Path(path)
    return path / f"stimuli/{seg}.mp4"


class Wen2017(study.Study):
    device: tp.ClassVar[str] = "Fmri"
    licence: tp.ClassVar[str] = "CC-BY 0"
    url: tp.ClassVar[str] = "https://academic.oup.com/cercor/article/28/12/4136/4560155"
    TR_FMRI_S: tp.ClassVar[float] = 2.0  # don't rely on nifti header

    def _download(self) -> None:
        raise NotImplementedError("Download method not implemented yet")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        base = self.path / "download" / "video_fmri_dataset"
        for subject_dir in base.iterdir():
            subject = subject_dir.name
            if not subject.startswith("subject") or not subject_dir.is_dir():
                continue

            for seg_dir in (subject_dir / "fmri").iterdir():
                seg = seg_dir.name
                is_train = seg.startswith("seg")
                is_test = seg.startswith("test")
                if not (is_train or is_test):
                    continue
                file = _get_video_file(base, seg)
                if not file.exists():
                    raise FileNotFoundError(f"Missing video file: {file}")

                fmri_runs = range(1, 3) if is_train else range(1, 11)
                for run_ in fmri_runs:
                    nii = _get_nii_file(base, subject, seg, run_)
                    if not nii.exists():
                        raise FileNotFoundError(f"Missing nii file: {nii}")

                    yield dict(subject=subject, seg=seg, run=run_)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        import nibabel

        tl = timeline
        base = self.path / "download" / "video_fmri_dataset"
        video_file = _get_video_file(base, tl["seg"])
        nii_file = _get_nii_file(base, tl["subject"], tl["seg"], tl["run"])
        nii: tp.Any = nibabel.load(nii_file, mmap=True)
        freq = 1.0 / self.TR_FMRI_S
        dur = nii.shape[-1] / freq
        fmri = dict(
            type="Fmri", start=0, filepath=nii_file, frequency=freq, duration=dur
        )
        return pd.DataFrame([dict(type="Video", start=0, filepath=video_file), fmri])
