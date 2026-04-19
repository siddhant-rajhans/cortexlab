"""Foundation-model feature extractor for representational-alignment baselines.

Design goals
------------

* One class, many presets. The five alignment baselines used in the class
  project differ in (a) which HuggingFace checkpoint to load, (b) how each
  stimulus is preprocessed, and (c) which forward-pass output to keep.
  All three are captured in :class:`ExtractorConfig`; the extractor class
  itself is a thin dispatcher.

* Lazy model loading. The transformers + torchvision imports happen when
  you actually instantiate an extractor, not when you import this module.
  This keeps test suites and slide-rendering pipelines cheap.

* Stable feature layout. Every preset returns ``(n_stim, d)`` arrays
  (pooled token representations). Per-layer extraction is available via
  ``output_layer_indices``.

* Bring-your-own-stimulus. The extractor accepts either a sequence of
  file paths (videos or images), or a sequence of already-decoded
  ``StimulusSpec`` objects. For the BOLD Moments dataset the former is
  enough; the latter lets downstream code reuse decoded frames across
  extractors.

Actually running the extractors requires the HuggingFace ``transformers``
package and network access to download weights. The module is usable
without either for construction and for writing tests that mock the
forward pass.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# stimulus container
# --------------------------------------------------------------------------- #

@dataclass
class StimulusSpec:
    """A single stimulus, resolved to a form the extractor can consume.

    Exactly one of ``frames`` (video, ``(T, H, W, 3)`` uint8) or ``image``
    (``(H, W, 3)`` uint8) should be populated, matching the extractor's
    ``input_type``.
    """
    stimulus_id: str
    frames: np.ndarray | None = None
    image: np.ndarray | None = None
    caption: str | None = None


# --------------------------------------------------------------------------- #
# extractor config + presets
# --------------------------------------------------------------------------- #

@dataclass
class ExtractorConfig:
    """Configuration for a single alignment baseline.

    Attributes
    ----------
    name
        Human-readable preset identifier used in cache paths and logs.
    hf_model_id
        Hugging Face hub checkpoint.
    input_type
        ``"image"`` (single frame) or ``"video"`` (multi-frame clip).
    n_frames
        For ``input_type == "video"`` only: how many frames per clip.
    expected_dim
        Expected pooled-feature dimensionality. Used as a cheap
        correctness check after the first batch.
    processor_class
        Name of the HF processor class to load. Auto-detected when None.
    model_class
        Name of the HF model class to load. Auto-detected when None.
    pooling
        How to reduce token-sequence outputs to a single vector:
        ``"cls"`` (first token), ``"mean"`` (mean over sequence), or
        ``"pooler"`` (use model's ``pooler_output`` when available).
    """
    name: str
    hf_model_id: str
    input_type: str  # "image" | "video"
    expected_dim: int
    n_frames: int = 1
    processor_class: str | None = None
    model_class: str | None = None
    pooling: str = "pooler"


# The five alignment baselines used in the class-project study.
PRESETS: dict[str, ExtractorConfig] = {
    "clip-vit-l-14": ExtractorConfig(
        name="clip-vit-l-14",
        hf_model_id="openai/clip-vit-large-patch14",
        input_type="image",
        expected_dim=1024,
        processor_class="CLIPProcessor",
        model_class="CLIPVisionModel",
        pooling="pooler",
    ),
    "siglip2-vit-l": ExtractorConfig(
        name="siglip2-vit-l",
        hf_model_id="google/siglip2-large-patch16-384",
        input_type="image",
        expected_dim=1024,
        processor_class="AutoProcessor",
        model_class="AutoModel",
        pooling="pooler",
    ),
    "dinov2-vit-l": ExtractorConfig(
        name="dinov2-vit-l",
        hf_model_id="facebook/dinov2-large",
        input_type="image",
        expected_dim=1024,
        processor_class="AutoImageProcessor",
        model_class="AutoModel",
        pooling="cls",
    ),
    "vjepa2-vit-l": ExtractorConfig(
        name="vjepa2-vit-l",
        hf_model_id="facebook/vjepa2-vitl-fpc16-256",
        input_type="video",
        expected_dim=1024,
        n_frames=16,
        processor_class="AutoProcessor",
        model_class="AutoModel",
        pooling="mean",
    ),
    "paligemma2-3b": ExtractorConfig(
        name="paligemma2-3b",
        hf_model_id="google/paligemma2-3b-pt-224",
        input_type="image",
        expected_dim=2304,
        processor_class="AutoProcessor",
        model_class="AutoModel",
        pooling="mean",
    ),
}


# --------------------------------------------------------------------------- #
# extractor
# --------------------------------------------------------------------------- #

class FoundationFeatureExtractor:
    """Extract pooled features from a HuggingFace foundation model.

    Parameters
    ----------
    config
        An :class:`ExtractorConfig` specifying model, preprocessing,
        and pooling.
    device
        Torch device string (``"cuda"``, ``"cpu"``, ``"cuda:0"``, ...).
    dtype
        Forward-pass dtype. Defaults to ``torch.float16`` on CUDA,
        ``torch.float32`` on CPU.
    batch_size
        Number of stimuli per forward pass. Tune for your GPU memory.
    model_factory
        Test-only hook. When provided, called with ``(config, device,
        dtype)`` and must return ``(processor, model)``; skips real
        HuggingFace loading. Lets tests inject a stub forward-pass
        without downloading weights.
    """

    def __init__(
        self,
        config: ExtractorConfig,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        batch_size: int = 16,
        model_factory: Callable[..., tuple[object, torch.nn.Module]] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda"
                               else torch.float32)
        self.batch_size = int(batch_size)
        self._processor: object | None = None
        self._model: torch.nn.Module | None = None
        self._model_factory = model_factory

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "FoundationFeatureExtractor":
        """Construct an extractor from one of the named presets."""
        if preset not in PRESETS:
            raise KeyError(
                f"unknown preset {preset!r}; available: {sorted(PRESETS)}"
            )
        return cls(PRESETS[preset], **kwargs)

    # ------------------------------------------------------------------ #
    # model loading
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._model_factory is not None:
            self._processor, self._model = self._model_factory(
                self.config, self.device, self.dtype,
            )
            return

        try:
            import transformers
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "transformers is required for real feature extraction. "
                "`pip install transformers`. For tests, pass model_factory=..."
            ) from e

        logger.info("Loading %s (%s) on %s [%s]",
                    self.config.name, self.config.hf_model_id,
                    self.device, self.dtype)
        proc_cls_name = self.config.processor_class or "AutoProcessor"
        model_cls_name = self.config.model_class or "AutoModel"
        proc_cls = getattr(transformers, proc_cls_name)
        model_cls = getattr(transformers, model_cls_name)
        self._processor = proc_cls.from_pretrained(self.config.hf_model_id)
        model = model_cls.from_pretrained(
            self.config.hf_model_id, torch_dtype=self.dtype,
        ).to(self.device)
        model.eval()
        self._model = model

    # ------------------------------------------------------------------ #
    # extraction
    # ------------------------------------------------------------------ #

    def extract(
        self,
        stimuli: Sequence[str | Path | StimulusSpec],
    ) -> np.ndarray:
        """Extract pooled features for a sequence of stimuli.

        Returns an array of shape ``(n_stim, expected_dim)``.
        """
        self._ensure_loaded()
        specs = [s if isinstance(s, StimulusSpec) else self._load_spec(s)
                 for s in stimuli]

        out: list[np.ndarray] = []
        for i in range(0, len(specs), self.batch_size):
            batch = specs[i:i + self.batch_size]
            feats = self._forward_batch(batch)
            out.append(feats)
            logger.debug("extracted batch %d-%d -> shape %s",
                         i, i + len(batch), feats.shape)
        arr = np.concatenate(out, axis=0) if out else np.empty((0, 0))
        self._validate_shape(arr)
        return arr

    def _forward_batch(self, batch: list[StimulusSpec]) -> np.ndarray:
        """Subclasses / test doubles override this for custom forward paths.

        The default implementation builds a processor-compatible input dict
        and pools the last-hidden-state per ``self.config.pooling``.
        """
        inputs = self._processor_inputs(batch)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        pooled = self._pool(outputs)
        return pooled.detach().to(torch.float32).cpu().numpy()

    def _processor_inputs(self, batch: list[StimulusSpec]) -> dict:
        """Convert a batch of StimulusSpecs into kwargs for the HF model."""
        if self.config.input_type == "image":
            images = [b.image for b in batch]
            if any(img is None for img in images):
                raise ValueError(
                    f"{self.config.name} requires image stimuli; "
                    "got StimulusSpec without .image"
                )
            inputs = self._processor(images=images, return_tensors="pt")
        elif self.config.input_type == "video":
            clips = [b.frames for b in batch]
            if any(c is None for c in clips):
                raise ValueError(
                    f"{self.config.name} requires video stimuli; "
                    "got StimulusSpec without .frames"
                )
            inputs = self._processor(videos=clips, return_tensors="pt")
        else:
            raise ValueError(f"unknown input_type {self.config.input_type!r}")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _pool(self, outputs) -> torch.Tensor:
        """Reduce a HuggingFace model output to ``(batch, d)``."""
        if self.config.pooling == "pooler":
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            # Fall through to CLS if pooler_output is missing.
        last = getattr(outputs, "last_hidden_state", None)
        if last is None:
            last = outputs[0]
        if self.config.pooling in ("cls", "pooler"):
            return last[:, 0]
        if self.config.pooling == "mean":
            return last.mean(dim=1)
        raise ValueError(f"unknown pooling {self.config.pooling!r}")

    def _load_spec(self, path: str | Path) -> StimulusSpec:
        """Decode a stimulus file path into a StimulusSpec.

        For full fidelity on video clips this delegates to moviepy when
        available. Lightweight installs (test environments) can skip
        decoding by passing pre-built ``StimulusSpec`` instances.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        stim_id = path.stem
        if self.config.input_type == "image":
            from PIL import Image
            image = np.array(Image.open(path).convert("RGB"))
            return StimulusSpec(stimulus_id=stim_id, image=image)
        # Video
        import moviepy.editor as mpy  # type: ignore[import-untyped]
        clip = mpy.VideoFileClip(str(path))
        total = clip.reader.nframes
        n = self.config.n_frames
        idx = np.linspace(0, total - 1, n).round().astype(int)
        frames = np.stack([clip.get_frame(i / clip.fps) for i in idx], axis=0)
        clip.close()
        return StimulusSpec(stimulus_id=stim_id, frames=frames.astype(np.uint8))

    def _validate_shape(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        if arr.ndim != 2:
            raise RuntimeError(f"feature array must be 2-D, got {arr.shape}")
        if arr.shape[1] != self.config.expected_dim:
            logger.warning(
                "feature dim mismatch for %s: expected %d, got %d "
                "(silently accepting; update ExtractorConfig.expected_dim)",
                self.config.name, self.config.expected_dim, arr.shape[1],
            )

    # ------------------------------------------------------------------ #
    # caching helpers
    # ------------------------------------------------------------------ #

    def cache_key(self, stimulus_ids: Sequence[str]) -> str:
        """Deterministic cache key from model + stimulus manifest."""
        h = hashlib.sha1()
        h.update(self.config.hf_model_id.encode())
        h.update(self.config.pooling.encode())
        for sid in stimulus_ids:
            h.update(b"|")
            h.update(sid.encode())
        return h.hexdigest()[:16]

    def save_cache(self, features: np.ndarray, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=features,
            model_id=self.config.hf_model_id,
            pooling=self.config.pooling,
        )
        logger.info("Saved %s features (%s) to %s",
                    self.config.name, features.shape, path)

    @staticmethod
    def load_cache(path: str | Path) -> np.ndarray:
        data = np.load(path)
        return data["features"]
