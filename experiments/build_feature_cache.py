"""Build the per-modality feature cache consumed by the lesion experiment.

Given the BOLD Moments dataset on disk and a HuggingFace token, this
script produces two ``.npz`` files:

* ``<cache>/vision.npz`` — vision features for each middle frame, pooled
  from a preset registered in :mod:`cortexlab.features.extractors`.
* ``<cache>/text.npz`` — text features for each LLM caption, pooled
  from a preset registered in :mod:`cortexlab.features.text`.

Row order in both files is :func:`list_stimulus_paths` exactly (train
clips 0001-1000 then test clips 0001-0102, sorted by filename). That
order is what
:func:`cortexlab.data.studies.lahner2024bold.load_subject` expects.

Usage
-----

Full 1,102-stimulus build, using DINOv2 for vision and CLIP-text for text::

    python -m experiments.build_feature_cache \
        --cache-dir $CORTEXLAB_RESULTS/features \
        --vision-preset dinov2-vit-l \
        --text-preset clip-text-vit-l-14

Smoke test on 5 stimuli (no GPU needed for image extractors running on CPU)::

    python -m experiments.build_feature_cache \
        --cache-dir /tmp/smoke_cache \
        --vision-preset clip-vit-l-14 \
        --text-preset clip-text-vit-l-14 \
        --n-stimuli 5 \
        --device cpu
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from cortexlab.data.studies.lahner2024bold import (
    list_stimulus_paths,
    load_captions,
    middle_frame_paths,
)
from cortexlab.features.extractors import FoundationFeatureExtractor, StimulusSpec
from cortexlab.features.text import TextFeatureExtractor

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=str, required=True,
                    help="Output directory; writes vision.npz and text.npz.")
    ap.add_argument("--data-root", type=str, default=None,
                    help="BOLD Moments root; falls back to CORTEXLAB_DATA.")
    ap.add_argument("--vision-preset", type=str, default="dinov2-vit-l")
    ap.add_argument("--text-preset", type=str, default="clip-text-vit-l-14")
    ap.add_argument("--device", type=str,
                    default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--batch-size-vision", type=int, default=16)
    ap.add_argument("--batch-size-text", type=int, default=64)
    ap.add_argument("--n-stimuli", type=int, default=None,
                    help="Pilot: use only the first N stimuli.")
    ap.add_argument("--caption-generator", type=str,
                    default="GIT-git-large-coco")
    ap.add_argument("--caption-index", type=int, default=0)
    ap.add_argument("--join-captions", action="store_true",
                    help="Concat all 5 captions instead of picking one.")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-text", action="store_true")
    return ap.parse_args()


def _load_image(path: Path) -> np.ndarray:
    """PIL-free image loader. Uses PIL because torchvision.io struggles
    with the CSAIL JPEGs that do not carry an explicit color profile.
    """
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


def build_vision_cache(
    preset: str,
    frame_paths: list[Path],
    device: str,
    batch_size: int,
    out_path: Path,
) -> np.ndarray:
    logger.info("vision: %d frames, preset=%s, device=%s",
                len(frame_paths), preset, device)
    ext = FoundationFeatureExtractor.from_preset(
        preset, device=device, batch_size=batch_size,
    )
    specs = [
        StimulusSpec(stimulus_id=p.stem, image=_load_image(p))
        for p in frame_paths
    ]
    t0 = time.perf_counter()
    feats = ext.extract(specs)
    logger.info("vision features: %s in %.1f s", feats.shape, time.perf_counter() - t0)
    ext.save_cache(feats, out_path)
    return feats


def build_text_cache(
    preset: str,
    captions: list[str],
    device: str,
    batch_size: int,
    out_path: Path,
) -> np.ndarray:
    logger.info("text: %d captions, preset=%s, device=%s",
                len(captions), preset, device)
    ext = TextFeatureExtractor.from_preset(
        preset, device=device, batch_size=batch_size,
    )
    t0 = time.perf_counter()
    feats = ext.extract(captions)
    logger.info("text features: %s in %.1f s", feats.shape, time.perf_counter() - t0)
    ext.save_cache(feats, out_path)
    return feats


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    video_paths = list_stimulus_paths(args.data_root)
    if args.n_stimuli is not None:
        video_paths = video_paths[: args.n_stimuli]
    logger.info("target stimuli: %d (first %s, last %s)",
                len(video_paths), video_paths[0].name, video_paths[-1].name)

    if not args.skip_vision:
        frames = middle_frame_paths(args.data_root)
        if args.n_stimuli is not None:
            frames = frames[: args.n_stimuli]
        build_vision_cache(
            preset=args.vision_preset,
            frame_paths=frames,
            device=args.device,
            batch_size=args.batch_size_vision,
            out_path=cache_dir / "vision.npz",
        )

    if not args.skip_text:
        captions = load_captions(
            args.data_root,
            generator_key=args.caption_generator,
            caption_index=args.caption_index,
            join=args.join_captions,
        )
        if args.n_stimuli is not None:
            captions = captions[: args.n_stimuli]
        build_text_cache(
            preset=args.text_preset,
            captions=captions,
            device=args.device,
            batch_size=args.batch_size_text,
            out_path=cache_dir / "text.npz",
        )

    logger.info("cache ready at %s", cache_dir)
    for f in cache_dir.glob("*.npz"):
        arr = np.load(f)["features"]
        logger.info("  %s : %s %s", f.name, arr.shape, arr.dtype)


if __name__ == "__main__":
    main()
