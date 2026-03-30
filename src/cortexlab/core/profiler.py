"""Memory profiling utilities for tracking GPU memory usage.

Provides a context manager that records peak CUDA memory allocated
and reserved during a code block.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class MemoryReport:
    """GPU memory usage report."""

    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    current_allocated_mb: float = 0.0


@contextmanager
def memory_profiler(device: str | torch.device = "cuda"):
    """Context manager that tracks peak CUDA memory during a code block.

    Yields a :class:`MemoryReport` that is populated when the block exits.
    On CPU or when CUDA is unavailable, all values are zero.

    Example
    -------
    >>> with memory_profiler("cuda") as report:
    ...     model(batch)
    >>> print(f"Peak: {report.peak_allocated_mb:.1f} MB")
    """
    report = MemoryReport()

    if not torch.cuda.is_available():
        yield report
        return

    device = torch.device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    try:
        yield report
    finally:
        torch.cuda.synchronize(device)
        report.peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        report.peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        report.current_allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
