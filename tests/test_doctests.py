"""Lock in runnable docstring examples for analysis modules.

Without this, a refactor can silently break examples that users copy from
the analysis API docs. Keeping the list explicit avoids enabling doctests
repo-wide for modules whose docstrings are not curated as runnable examples.
"""

from __future__ import annotations

import doctest

import pytest

import cortexlab.analysis.brain_alignment
import cortexlab.analysis.cognitive_load
import cortexlab.analysis.connectivity
import cortexlab.analysis.lesion
import cortexlab.analysis.stats
import cortexlab.analysis.temporal_dynamics

DOCTEST_MODULES = [
    cortexlab.analysis.brain_alignment,
    cortexlab.analysis.cognitive_load,
    cortexlab.analysis.connectivity,
    cortexlab.analysis.lesion,
    cortexlab.analysis.stats,
    cortexlab.analysis.temporal_dynamics,
]


@pytest.mark.parametrize("mod", DOCTEST_MODULES, ids=lambda module: module.__name__)
def test_module_doctests_pass(mod) -> None:
    """Every curated analysis-module doctest runs without failures."""
    result = doctest.testmod(mod, verbose=False)
    assert result.failed == 0, (
        f"{mod.__name__}: {result.failed}/{result.attempted} doctest examples failed. "
        f"Re-run `python -m doctest -v {mod.__file__}` locally to inspect failures."
    )
