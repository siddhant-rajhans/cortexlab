"""Shared test utilities."""

import neuralset.segments as seg


def make_segments(n):
    """Create n dummy segments for SegmentData construction."""
    return [seg.Segment(start=float(i), duration=1.0, timeline="test") for i in range(n)]
