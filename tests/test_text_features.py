"""Tests for :class:`cortexlab.features.text.TextFeatureExtractor`.

Uses a stub tokenizer + model so the tests run without network access
or the HuggingFace ``transformers`` dependency's heavy weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cortexlab.features.text import (
    TEXT_PRESETS,
    TextExtractorConfig,
    TextFeatureExtractor,
)


# --------------------------------------------------------------------------- #
# stub tokenizer + model                                                      #
# --------------------------------------------------------------------------- #

class _FakeTokenizer:
    """Deterministic tokenizer: maps each character to its ord modulo V."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def __call__(self, batch, padding: bool = True, truncation: bool = True,
                 return_tensors: str = "pt", max_length: int | None = None):
        max_len = max_length or max(len(t) for t in batch)
        max_len = min(max_len, 32)
        ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, text in enumerate(batch):
            tokens = [ord(c) % self.vocab_size for c in text[:max_len]]
            ids[i, : len(tokens)] = torch.tensor(tokens)
            mask[i, : len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _FakeProjectionModel(torch.nn.Module):
    """Exposes ``get_text_features`` like CLIP / SigLIP."""

    def __init__(self, dim: int = 768, vocab_size: int = 256):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.proj = torch.nn.Linear(dim, dim)

    def get_text_features(self, input_ids, attention_mask=None):
        emb = self.embed(input_ids)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (emb * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            pooled = emb.mean(1)
        return self.proj(pooled)


class _FakeEncoderOutputs:
    def __init__(self, last):
        self.last_hidden_state = last


class _FakeEncoderModel(torch.nn.Module):
    """Model without ``get_text_features``; exercises CLS/mean fallback."""

    def __init__(self, dim: int = 128, vocab_size: int = 256):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)

    def forward(self, input_ids, attention_mask=None):
        return _FakeEncoderOutputs(last=self.embed(input_ids))


def _factory(model: torch.nn.Module, tokenizer=None):
    def make(config, device, dtype):
        return (tokenizer or _FakeTokenizer(), model.to(device))
    return make


@pytest.fixture
def texts():
    return [
        "a duck swimming in water",
        "two people dancing on a stage",
        "a car driving down a road at sunset",
    ]


# --------------------------------------------------------------------------- #
# preset registry                                                             #
# --------------------------------------------------------------------------- #

def test_text_presets_registered():
    assert "clip-text-vit-l-14" in TEXT_PRESETS
    assert "siglip2-text-vit-l" in TEXT_PRESETS


def test_from_preset_rejects_unknown():
    with pytest.raises(KeyError, match="unknown text preset"):
        TextFeatureExtractor.from_preset("not-a-real-preset")


def test_preset_configs_look_sane():
    for name, cfg in TEXT_PRESETS.items():
        assert cfg.name == name
        assert cfg.hf_model_id.count("/") == 1
        assert cfg.expected_dim > 0
        assert cfg.pooling in {"projection", "cls", "mean"}


# --------------------------------------------------------------------------- #
# extraction                                                                  #
# --------------------------------------------------------------------------- #

def test_extraction_shape_projection(texts):
    ext = TextFeatureExtractor.from_preset(
        "clip-text-vit-l-14", device="cpu",
        model_factory=_factory(_FakeProjectionModel(dim=768)),
    )
    feats = ext.extract(texts)
    assert feats.shape == (3, 768)
    assert feats.dtype == np.float32


def test_extraction_is_deterministic(texts):
    ext = TextFeatureExtractor.from_preset(
        "clip-text-vit-l-14", device="cpu",
        model_factory=_factory(_FakeProjectionModel(dim=768)),
    )
    a = ext.extract(texts)
    b = ext.extract(texts)
    np.testing.assert_array_equal(a, b)


def test_different_texts_yield_different_embeddings(texts):
    ext = TextFeatureExtractor.from_preset(
        "clip-text-vit-l-14", device="cpu",
        model_factory=_factory(_FakeProjectionModel(dim=768)),
    )
    feats = ext.extract(texts)
    # Each row should be distinct (tokens differ → embeddings differ).
    norms = np.linalg.norm(feats[:, None] - feats[None, :], axis=-1)
    assert np.all(norms[np.triu_indices(3, k=1)] > 0)


def test_fallback_to_cls_when_no_projection(texts):
    """Model without get_text_features should use CLS token path."""
    cfg = TextExtractorConfig(
        name="fake-encoder",
        hf_model_id="fake/encoder",
        expected_dim=128,
        pooling="cls",
    )
    ext = TextFeatureExtractor(
        cfg, device="cpu",
        model_factory=_factory(_FakeEncoderModel(dim=128)),
    )
    feats = ext.extract(texts)
    assert feats.shape == (3, 128)


def test_mean_pooling_respects_attention_mask(texts):
    cfg = TextExtractorConfig(
        name="mean-encoder",
        hf_model_id="fake/encoder",
        expected_dim=128,
        pooling="mean",
    )
    ext = TextFeatureExtractor(
        cfg, device="cpu",
        model_factory=_factory(_FakeEncoderModel(dim=128)),
    )
    feats = ext.extract(texts)
    assert feats.shape == (3, 128)
    # With variable-length inputs, mean should not equal sum/max_len (no padding leak).
    long_text = texts[2]      # longest
    short_text = texts[0]     # shortest
    emb_long = ext.extract([long_text])[0]
    emb_short = ext.extract([short_text])[0]
    # The short-text row should not equal the long-text row, even with padding.
    assert not np.allclose(emb_short, emb_long)


def test_batching_preserves_rows(texts):
    # Share a single model instance across both extractors so the only
    # difference is batch_size; otherwise random init changes the weights.
    shared = _FakeProjectionModel(dim=768)
    ext = TextFeatureExtractor.from_preset(
        "clip-text-vit-l-14", device="cpu", batch_size=2,
        model_factory=_factory(shared),
    )
    all_at_once = TextFeatureExtractor.from_preset(
        "clip-text-vit-l-14", device="cpu", batch_size=16,
        model_factory=_factory(shared),
    )
    a = ext.extract(texts)
    b = all_at_once.extract(texts)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_extract_empty_input_returns_empty():
    ext = TextFeatureExtractor.from_preset(
        "clip-text-vit-l-14", device="cpu",
        model_factory=_factory(_FakeProjectionModel(dim=768)),
    )
    feats = ext.extract([])
    assert feats.size == 0


# --------------------------------------------------------------------------- #
# cache round-trip                                                            #
# --------------------------------------------------------------------------- #

def test_save_and_load_cache(tmp_path: Path, texts):
    ext = TextFeatureExtractor.from_preset(
        "clip-text-vit-l-14", device="cpu",
        model_factory=_factory(_FakeProjectionModel(dim=768)),
    )
    feats = ext.extract(texts)
    out = tmp_path / "text.npz"
    ext.save_cache(feats, out)
    restored = TextFeatureExtractor.load_cache(out)
    np.testing.assert_array_equal(feats, restored)


def test_cache_key_is_deterministic_and_sensitive():
    ext = TextFeatureExtractor.from_preset(
        "clip-text-vit-l-14", device="cpu",
        model_factory=_factory(_FakeProjectionModel(dim=768)),
    )
    a = ext.cache_key(["one", "two", "three"])
    b = ext.cache_key(["one", "two", "three"])
    c = ext.cache_key(["one", "two", "four"])
    assert a == b
    assert a != c
    assert len(a) == 16
