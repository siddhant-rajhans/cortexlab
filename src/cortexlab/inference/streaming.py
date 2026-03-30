"""Real-time sliding-window fMRI prediction from live feature streams.

:class:`StreamingPredictor` buffers pre-extracted feature tensors one TR
at a time and emits cortical-surface predictions once the context window
is full.  Designed for BCI pipelines where features arrive continuously
from upstream extractor models.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np
import torch
from neuralset.dataloader import SegmentData

logger = logging.getLogger(__name__)


class StreamingPredictor:
    """Sliding-window predictor for real-time fMRI inference.

    Operates at the *feature level* -- the caller must provide
    pre-extracted tensors from running extractor models (e.g. Wav2Vec2,
    V-JEPA2, LLaMA).

    Parameters
    ----------
    model : torch.nn.Module
        A :class:`FmriEncoderModel` instance in eval mode.
    window_trs : int
        Number of TRs that form the context window.
    step_trs : int
        Emit a prediction every *step_trs* frames.
    tr_seconds : float
        Duration of one TR in seconds.
    modalities : list[str]
        Expected modality keys (e.g. ``["text", "audio", "video"]``).
    device : str or torch.device
        Device for inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        window_trs: int = 40,
        step_trs: int = 1,
        tr_seconds: float = 1.0,
        modalities: list[str] | None = None,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.window_trs = window_trs
        self.step_trs = step_trs
        self.tr_seconds = tr_seconds
        self.modalities = modalities or list(model.feature_dims.keys())
        self.device = torch.device(device)
        self._buffer: deque[dict[str, torch.Tensor]] = deque(maxlen=window_trs)
        self._frames_since_emit = 0
        self._lock = threading.Lock()

    @classmethod
    def from_cortexlab_model(
        cls,
        cortexlab_model,
        window_trs: int = 40,
        step_trs: int = 1,
        tr_seconds: float = 1.0,
        device: str = "cuda",
    ) -> StreamingPredictor:
        """Create from a loaded CortexLab/TribeModel inference wrapper."""
        return cls(
            model=cortexlab_model._model,
            window_trs=window_trs,
            step_trs=step_trs,
            tr_seconds=tr_seconds,
            device=device,
        )

    def push_frame(
        self, features: dict[str, torch.Tensor]
    ) -> np.ndarray | None:
        """Push one TR's worth of features and maybe get a prediction.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            Mapping from modality name to feature tensor.  Each tensor
            should have shape ``(n_layers, D)`` or ``(D,)``.

        Returns
        -------
        np.ndarray or None
            Prediction of shape ``(n_vertices,)`` if a prediction was
            emitted, otherwise ``None``.
        """
        with self._lock:
            normalised: dict[str, torch.Tensor] = {}
            for mod in self.modalities:
                t = features.get(mod)
                if t is None:
                    # Zero-fill missing modality
                    dims = self.model.feature_dims.get(mod)
                    if dims is not None:
                        num_layers, feat_dim = dims
                        t = torch.zeros(num_layers, feat_dim)
                    else:
                        continue
                if t.ndim == 1:
                    t = t.unsqueeze(0)  # (D,) -> (1, D)
                normalised[mod] = t
            self._buffer.append(normalised)
            self._frames_since_emit += 1

            if len(self._buffer) < self.window_trs:
                return None
            if self._frames_since_emit < self.step_trs:
                return None

            self._frames_since_emit = 0
            return self._predict()

    def _predict(self) -> np.ndarray:
        """Run inference on the current buffer contents."""
        batch_data: dict[str, torch.Tensor] = {}
        for mod in self.modalities:
            frames = []
            for frame in self._buffer:
                if mod in frame:
                    frames.append(frame[mod])
            if frames:
                # Stack: (T, L, D) -> (L, D, T) -> (1, L, D, T)
                stacked = torch.stack(frames, dim=0)  # (T, L, D)
                stacked = stacked.permute(1, 2, 0).unsqueeze(0)  # (1, L, D, T)
                batch_data[mod] = stacked.to(self.device)

        import neuralset.segments as seg
        dummy_segments = [seg.Segment(start=0.0, duration=float(self.window_trs * self.tr_seconds), timeline="stream")]
        batch = SegmentData(data=batch_data, segments=dummy_segments)
        with torch.inference_mode():
            pred = self.model(batch, pool_outputs=True)  # (1, V, T')
        return pred[0, :, -1].cpu().numpy()  # (V,) -- last timestep

    def flush(self) -> list[np.ndarray]:
        """Force-emit predictions for any remaining buffered frames."""
        with self._lock:
            results = []
            if len(self._buffer) >= self.window_trs:
                results.append(self._predict())
            return results

    def reset(self) -> None:
        """Clear the buffer and reset the step counter."""
        with self._lock:
            self._buffer.clear()
            self._frames_since_emit = 0
