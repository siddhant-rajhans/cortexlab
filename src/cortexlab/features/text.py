"""Text-encoder feature extractor for representational-alignment baselines.

Mirrors :class:`cortexlab.features.extractors.FoundationFeatureExtractor`
but for text inputs. Produces a pooled ``(n_texts, d)`` array that slots
directly into the per-modality feature cache consumed by
:func:`cortexlab.data.studies.lahner2024bold.load_subject`.

Design notes
------------

* **Reuses the vision-side pattern** (``TextExtractorConfig`` +
  presets). One class, many presets, lazy HuggingFace import.
* **CLIP / SigLIP** expose a dedicated ``get_text_features`` method that
  returns a projected embedding already aligned with their image
  embeddings in a joint space. That is the right surface when text
  features will be regressed against brain responses alongside matching
  vision features.
* **Generic language models** fall back to pooling the
  ``last_hidden_state``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class TextExtractorConfig:
    """Configuration for a single text-encoder preset.

    Attributes
    ----------
    name
        Short identifier used in cache paths and logs.
    hf_model_id
        HuggingFace Hub checkpoint.
    expected_dim
        Pooled-feature dimensionality. Used as a cheap correctness
        check after the first batch.
    model_class
        Name of the HF model class. ``CLIPModel`` and ``SiglipModel``
        expose ``get_text_features``; everything else is pooled from
        ``last_hidden_state``.
    tokenizer_class
        Name of the HF tokenizer class. Defaults to ``AutoTokenizer``.
    max_length
        Tokenizer truncation length. None keeps the model default.
    pooling
        ``"projection"`` (use ``get_text_features`` when available),
        ``"cls"``, or ``"mean"``. ``projection`` is the right default
        for CLIP/SigLIP; it falls back to CLS when the model lacks a
        projection head.
    """

    name: str
    hf_model_id: str
    expected_dim: int
    model_class: str = "AutoModel"
    tokenizer_class: str = "AutoTokenizer"
    max_length: int | None = None
    pooling: str = "projection"


TEXT_PRESETS: dict[str, TextExtractorConfig] = {
    "clip-text-vit-l-14": TextExtractorConfig(
        name="clip-text-vit-l-14",
        hf_model_id="openai/clip-vit-large-patch14",
        expected_dim=768,
        model_class="CLIPModel",
        tokenizer_class="CLIPTokenizer",
        max_length=77,
        pooling="projection",
    ),
    "siglip2-text-vit-l": TextExtractorConfig(
        name="siglip2-text-vit-l",
        hf_model_id="google/siglip2-large-patch16-384",
        expected_dim=1152,
        model_class="AutoModel",
        tokenizer_class="AutoTokenizer",
        max_length=64,
        pooling="projection",
    ),
}


class TextFeatureExtractor:
    """Text-side counterpart to :class:`FoundationFeatureExtractor`.

    Parameters
    ----------
    config
        A :class:`TextExtractorConfig`.
    device
        Torch device string.
    dtype
        Forward-pass dtype. Defaults to ``float16`` on CUDA,
        ``float32`` on CPU.
    batch_size
        Texts per forward pass.
    model_factory
        Test hook. When provided, returns ``(tokenizer, model)``
        directly and skips the HuggingFace loader. Required to test
        without network access.
    """

    def __init__(
        self,
        config: TextExtractorConfig,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        batch_size: int = 32,
        model_factory: Callable[..., tuple[object, torch.nn.Module]] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.batch_size = int(batch_size)
        self._tokenizer: object | None = None
        self._model: torch.nn.Module | None = None
        self._model_factory = model_factory

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "TextFeatureExtractor":
        if preset not in TEXT_PRESETS:
            raise KeyError(
                f"unknown text preset {preset!r}; available: {sorted(TEXT_PRESETS)}"
            )
        return cls(TEXT_PRESETS[preset], **kwargs)

    # ------------------------------------------------------------------ #
    # model loading                                                      #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._model_factory is not None:
            self._tokenizer, self._model = self._model_factory(
                self.config, self.device, self.dtype,
            )
            return

        try:
            import transformers
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "transformers is required for real text-feature extraction. "
                "Install it or pass model_factory=... for tests."
            ) from e

        logger.info("Loading text encoder %s (%s) on %s [%s]",
                    self.config.name, self.config.hf_model_id,
                    self.device, self.dtype)
        tok_cls = getattr(transformers, self.config.tokenizer_class)
        model_cls = getattr(transformers, self.config.model_class)
        self._tokenizer = tok_cls.from_pretrained(self.config.hf_model_id)
        model = model_cls.from_pretrained(
            self.config.hf_model_id, torch_dtype=self.dtype,
        ).to(self.device)
        model.eval()
        self._model = model

    # ------------------------------------------------------------------ #
    # extraction                                                         #
    # ------------------------------------------------------------------ #

    def extract(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``(n_texts, expected_dim)`` pooled embeddings."""
        self._ensure_loaded()
        texts = list(texts)
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            feats = self._forward_batch(batch)
            out.append(feats)
            logger.debug("text batch %d-%d -> %s", i, i + len(batch), feats.shape)
        arr = np.concatenate(out, axis=0) if out else np.empty((0, 0))
        self._validate_shape(arr)
        return arr

    def _forward_batch(self, batch: list[str]) -> np.ndarray:
        enc = self._tokenize(batch)
        with torch.inference_mode():
            feats = self._pool(enc)
        return feats.detach().to(torch.float32).cpu().numpy()

    def _tokenize(self, batch: list[str]) -> dict[str, torch.Tensor]:
        kwargs = {"padding": True, "truncation": True, "return_tensors": "pt"}
        if self.config.max_length is not None:
            kwargs["max_length"] = self.config.max_length
        inputs = self._tokenizer(batch, **kwargs)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _pool(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        pooling = self.config.pooling
        if pooling == "projection":
            # CLIP / SigLIP expose a projected text embedding.
            getter = getattr(self._model, "get_text_features", None)
            if getter is not None:
                return getter(**inputs)
            # Fall through to CLS.
            pooling = "cls"
        outputs = self._model(**inputs)
        last = getattr(outputs, "last_hidden_state", None)
        if last is None:
            last = outputs[0]
        if pooling == "cls":
            return last[:, 0]
        if pooling == "mean":
            mask = inputs.get("attention_mask")
            if mask is not None:
                mask_f = mask.unsqueeze(-1).to(last.dtype)
                return (last * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            return last.mean(dim=1)
        raise ValueError(f"unknown pooling {pooling!r}")

    def _validate_shape(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        if arr.ndim != 2:
            raise RuntimeError(f"text features must be 2-D, got {arr.shape}")
        if arr.shape[1] != self.config.expected_dim:
            logger.warning(
                "text feature dim mismatch for %s: expected %d, got %d "
                "(silently accepting)",
                self.config.name, self.config.expected_dim, arr.shape[1],
            )

    # ------------------------------------------------------------------ #
    # caching                                                            #
    # ------------------------------------------------------------------ #

    def cache_key(self, texts: Sequence[str]) -> str:
        h = hashlib.sha1()
        h.update(self.config.hf_model_id.encode())
        h.update(self.config.pooling.encode())
        for t in texts:
            h.update(b"|")
            h.update(t.encode())
        return h.hexdigest()[:16]

    def save_cache(self, features: np.ndarray, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, features=features,
            model_id=self.config.hf_model_id,
            pooling=self.config.pooling,
        )
        logger.info("saved text features %s -> %s", features.shape, path)

    @staticmethod
    def load_cache(path: str | Path) -> np.ndarray:
        data = np.load(path)
        return data["features"]
