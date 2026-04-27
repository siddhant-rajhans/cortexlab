"""Tests for :mod:`cortexlab.viz.surface_renderer`.

The matplotlib renderer is always exercised. The plotly renderer is
gated on ``pytest.importorskip`` so the suite passes without the
``[viz]`` extras installed.

Renderers produce PNG bytes; we check that those bytes are non-empty,
start with the PNG magic number, and that the factory's auto-fallback
behaves correctly under both presence and absence of plotly.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from cortexlab.viz.surface_renderer import (
    MESH_VERTS_PER_HEMI,
    MatplotlibRenderer,
    RenderConfig,
    SurfaceRenderer,
    make_renderer,
    truncate_to_mesh,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# truncate_to_mesh                                                            #
# --------------------------------------------------------------------------- #

def test_truncate_to_mesh_identity_when_already_target_size():
    n = MESH_VERTS_PER_HEMI["fsaverage5"] * 2
    arr = np.arange(n, dtype=np.float32)
    out = truncate_to_mesh(arr, "fsaverage5")
    np.testing.assert_array_equal(out, arr)


def test_truncate_to_mesh_downsamples_fsaverage7_to_fsaverage5():
    # Build a fake fsaverage7-sized array; verify the truncated copy
    # contains the first N5 verts of LH and the first N5 verts of RH.
    n7 = MESH_VERTS_PER_HEMI["fsaverage"] * 2
    arr = np.arange(n7, dtype=np.float32)
    out = truncate_to_mesh(arr, "fsaverage5")
    n5 = MESH_VERTS_PER_HEMI["fsaverage5"]
    assert out.shape == (2 * n5,)
    np.testing.assert_array_equal(out[:n5], np.arange(n5))
    rh_offset = MESH_VERTS_PER_HEMI["fsaverage"]
    np.testing.assert_array_equal(out[n5:], np.arange(rh_offset, rh_offset + n5))


def test_truncate_to_mesh_rejects_upsampling():
    n5 = MESH_VERTS_PER_HEMI["fsaverage5"] * 2
    arr = np.zeros(n5, dtype=np.float32)
    with pytest.raises(ValueError, match="cannot upsample|lower-resolution"):
        truncate_to_mesh(arr, "fsaverage7")


# --------------------------------------------------------------------------- #
# RenderConfig                                                                #
# --------------------------------------------------------------------------- #

def test_render_config_defaults():
    c = RenderConfig()
    assert c.mesh == "fsaverage5"
    assert c.symmetric_cbar is True
    assert c.bg_color == "white"


def test_render_config_is_frozen():
    c = RenderConfig()
    with pytest.raises(Exception):
        c.mesh = "fsaverage7"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

def test_make_renderer_explicit_matplotlib_works():
    r = make_renderer(engine="matplotlib")
    assert isinstance(r, MatplotlibRenderer)
    assert r.name == "matplotlib"


def test_make_renderer_unknown_engine_raises():
    with pytest.raises(ValueError, match="unknown engine"):
        make_renderer(engine="vulkan")  # type: ignore[arg-type]


def test_make_renderer_explicit_plotly_raises_without_extras(monkeypatch):
    """If the user explicitly asks for plotly but it isn't installed, the
    factory must raise — silently falling back would surprise them."""
    # Pretend plotly is uninstallable for this test.
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "kaleido", None)
    with pytest.raises(ImportError, match="\\[viz\\] extras"):
        make_renderer(engine="plotly")


def test_make_renderer_auto_falls_back_when_plotly_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "kaleido", None)
    r = make_renderer(engine="auto")
    assert isinstance(r, MatplotlibRenderer)


# --------------------------------------------------------------------------- #
# Matplotlib renderer smoke tests                                             #
# --------------------------------------------------------------------------- #

@pytest.fixture
def synth_stat_map():
    """A small fsaverage5-sized stat map; LH gets a positive bump on
    half the vertices, RH stays at zero."""
    n_per_hemi = MESH_VERTS_PER_HEMI["fsaverage5"]
    rng = np.random.default_rng(0)
    lh = np.where(rng.random(n_per_hemi) > 0.5, 0.5, 0.0).astype(np.float32)
    rh = np.zeros(n_per_hemi, dtype=np.float32)
    return np.concatenate([lh, rh])


def test_matplotlib_renderer_produces_png_left(synth_stat_map):
    r = make_renderer(engine="matplotlib", mesh="fsaverage5")
    config = RenderConfig(mesh="fsaverage5", dpi=72)
    png = r.render_frame(synth_stat_map, view=(0.0, 180.0),
                         hemi="left", config=config)
    assert isinstance(png, (bytes, bytearray))
    assert png[:8] == PNG_MAGIC
    assert len(png) > 1000  # not a degenerate empty image


def test_matplotlib_renderer_produces_png_both_hemispheres(synth_stat_map):
    r = make_renderer(engine="matplotlib", mesh="fsaverage5")
    config = RenderConfig(mesh="fsaverage5", dpi=72)
    png = r.render_frame(synth_stat_map, view=(0.0, 180.0),
                         hemi="both", config=config)
    assert png[:8] == PNG_MAGIC
    assert len(png) > 1000


def test_matplotlib_renderer_static_panels(synth_stat_map):
    r = make_renderer(engine="matplotlib", mesh="fsaverage5")
    config = RenderConfig(mesh="fsaverage5", dpi=72)
    views = [("L lat", (0.0, 180.0)), ("R lat", (0.0, 0.0))]
    hemis = ["left", "right"]
    png = r.render_static_panels(synth_stat_map, views, hemis, config)
    assert png[:8] == PNG_MAGIC


def test_matplotlib_renderer_handles_all_nan(synth_stat_map):
    """An all-NaN stat map should not crash; renderer falls back to
    a blank brain (matplotlib will accept the NaNs as transparent)."""
    nan_map = np.full_like(synth_stat_map, np.nan)
    r = make_renderer(engine="matplotlib", mesh="fsaverage5")
    config = RenderConfig(mesh="fsaverage5", dpi=72)
    png = r.render_frame(nan_map, view=(0.0, 180.0),
                         hemi="left", config=config)
    assert png[:8] == PNG_MAGIC


# --------------------------------------------------------------------------- #
# Plotly renderer (only if extras installed)                                  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def plotly_renderer():
    pytest.importorskip("plotly")
    pytest.importorskip("kaleido")
    return make_renderer(engine="plotly", mesh="fsaverage5")


def test_plotly_renderer_factory(plotly_renderer):
    assert plotly_renderer.name == "plotly"


def test_plotly_renderer_produces_png_left(plotly_renderer, synth_stat_map):
    config = RenderConfig(mesh="fsaverage5", width=400, height=400)
    png = plotly_renderer.render_frame(
        synth_stat_map, view=(0.0, 180.0),
        hemi="left", config=config,
    )
    assert isinstance(png, (bytes, bytearray))
    assert png[:8] == PNG_MAGIC
    assert len(png) > 1000


def test_plotly_renderer_html_output(plotly_renderer, synth_stat_map):
    config = RenderConfig(mesh="fsaverage5", width=400, height=400)
    html = plotly_renderer.render_html(
        synth_stat_map, view=(0.0, 180.0),
        hemi="left", config=config,
    )
    assert isinstance(html, str)
    assert "<html" in html.lower()
    # Plotly figures embed Plotly.js or a CDN reference.
    assert "plotly" in html.lower()
