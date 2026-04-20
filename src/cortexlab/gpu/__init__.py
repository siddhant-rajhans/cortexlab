"""GPU-accelerated building blocks for CortexLab.

The primary entry point is :class:`cortexlab.gpu.ridge.VoxelRidgeEncoder`,
a voxelwise ridge-regression encoder with cross-validated regularization
and an optional Triton backend for NVIDIA GPUs.
"""

from cortexlab.gpu.ridge import VoxelRidgeEncoder

__all__ = ["VoxelRidgeEncoder"]
