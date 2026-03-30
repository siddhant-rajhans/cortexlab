"""ONNX export for cross-platform deployment.

Wraps the FmriEncoderModel in a thin module that accepts flat tensor
inputs (required by ONNX tracing) instead of the SegmentData object.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from neuralset.dataloader import SegmentData
from torch import nn

logger = logging.getLogger(__name__)


class _OnnxWrapper(nn.Module):
    """Wraps FmriEncoderModel to accept flat tensor inputs for ONNX export."""

    def __init__(self, model: nn.Module, modality_keys: list[str]):
        super().__init__()
        self.model = model
        self.modality_keys = modality_keys

    def forward(self, *tensors) -> torch.Tensor:
        import neuralset.segments as seg

        data = {}
        for key, tensor in zip(self.modality_keys, tensors):
            data[key] = tensor
        # Add dummy subject_id (average subject = 0)
        B = tensors[0].shape[0]
        data["subject_id"] = torch.zeros(B, dtype=torch.long, device=tensors[0].device)
        segments = [seg.Segment(start=0.0, duration=1.0, timeline="export") for _ in range(B)]
        batch = SegmentData(data=data, segments=segments)
        return self.model(batch)


def export_to_onnx(
    model: nn.Module,
    sample_batch: SegmentData,
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export a FmriEncoderModel to ONNX format.

    Parameters
    ----------
    model : nn.Module
        The :class:`FmriEncoderModel` instance.
    sample_batch : SegmentData
        A sample batch used for tracing. Determines input shapes.
    output_path : str or Path
        Path for the output ``.onnx`` file.
    opset_version : int
        ONNX opset version.

    Returns
    -------
    Path
        The path to the saved ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    modality_keys = [k for k in model.feature_dims if k in sample_batch.data]
    wrapper = _OnnxWrapper(model, modality_keys)
    wrapper.eval()

    sample_tensors = tuple(sample_batch.data[k] for k in modality_keys)
    input_names = modality_keys
    output_names = ["predictions"]

    dynamic_axes = {k: {0: "batch_size"} for k in modality_keys}
    dynamic_axes["predictions"] = {0: "batch_size"}

    torch.onnx.export(
        wrapper,
        sample_tensors,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info("Exported ONNX model to %s", output_path)
    return output_path
