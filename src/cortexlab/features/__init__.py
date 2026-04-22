"""Feature extractors for representational-alignment baselines.

Provides a single class :class:`FoundationFeatureExtractor` that wraps any
HuggingFace vision / video / vision-language model and returns per-stimulus
embeddings suitable for voxelwise encoding. Five preset configurations
match the alignment baselines used in the CortexLab modality-attribution
study: CLIP ViT-L/14, SigLIP 2 ViT-L, DINOv2 ViT-L, V-JEPA 2 ViT-L, and
PaliGemma 2 3B.

Typical usage::

    from cortexlab.features import FoundationFeatureExtractor

    ext = FoundationFeatureExtractor.from_preset("clip-vit-l-14",
                                                 device="cuda")
    feats = ext.extract(video_paths)   # shape (n_stim, d)
    ext.save_cache(feats, "cache/clip.npz")

The module is import-light: the heavy transformers / torchvision imports
happen when an extractor is instantiated, not when the package is loaded.
"""

from cortexlab.features.extractors import (
    PRESETS,
    FoundationFeatureExtractor,
    StimulusSpec,
)
from cortexlab.features.text import (
    TEXT_PRESETS,
    TextExtractorConfig,
    TextFeatureExtractor,
)

__all__ = [
    "FoundationFeatureExtractor",
    "PRESETS",
    "StimulusSpec",
    "TextFeatureExtractor",
    "TextExtractorConfig",
    "TEXT_PRESETS",
]
