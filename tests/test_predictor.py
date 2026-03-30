"""Tests for the predictor module validation logic."""

from pathlib import Path

import pytest

lightning = pytest.importorskip("lightning", reason="lightning not installed")


class TestPredictorValidation:
    def test_valid_suffixes_keys(self):
        from cortexlab.inference.predictor import VALID_SUFFIXES

        assert "text_path" in VALID_SUFFIXES
        assert "audio_path" in VALID_SUFFIXES
        assert "video_path" in VALID_SUFFIXES

    def test_valid_suffixes_text(self):
        from cortexlab.inference.predictor import VALID_SUFFIXES

        assert ".txt" in VALID_SUFFIXES["text_path"]

    def test_valid_suffixes_audio(self):
        from cortexlab.inference.predictor import VALID_SUFFIXES

        assert ".wav" in VALID_SUFFIXES["audio_path"]
        assert ".mp3" in VALID_SUFFIXES["audio_path"]

    def test_valid_suffixes_video(self):
        from cortexlab.inference.predictor import VALID_SUFFIXES

        assert ".mp4" in VALID_SUFFIXES["video_path"]
        assert ".avi" in VALID_SUFFIXES["video_path"]


class TestDownloadFile:
    def test_download_creates_file(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from cortexlab.inference.predictor import download_file

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test data"]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("cortexlab.inference.predictor.requests.get", return_value=mock_response):
            path = download_file("http://example.com/file.bin", tmp_path / "out.bin")
            assert path.exists()
