"""Pluggable cortical-surface renderer.

This module provides a thin abstraction over two rendering engines that
both produce per-vertex stat maps on inflated fsaverage surfaces:

* ``MatplotlibRenderer`` uses ``nilearn.plotting.plot_surf_stat_map``
  with matplotlib's default (3D, software rasterizer) backend. Pure
  CPU; no extra dependencies; slow on dense meshes.

* ``PlotlyRenderer`` uses the same nilearn function with
  ``engine="plotly"``. The actual render runs in WebGL, so on any
  machine with a GPU (laptop NVIDIA, HPC datacenter card, even
  integrated Intel) the cortex draws in shader code rather than
  matplotlib's Python loop. Output PNGs are produced via ``kaleido``,
  which bundles a headless Chromium. ``HTML`` output is essentially
  free since plotly already builds the figure.

Why two engines instead of "just use plotly"?

* Plotly's headless toolchain (kaleido + Chromium) occasionally fights
  with HPC system libraries; users who can't install it should still
  be able to make figures. Matplotlib is a guaranteed fallback.
* For deck-quality static figures, both engines are interchangeable
  and matplotlib produces an output style some users prefer.

The factory :func:`make_renderer` defaults to plotly when the
``[viz]`` extras are installed and silently falls back to matplotlib
otherwise. Callers who want explicit control pass ``engine="..."``.

Animation API
-------------

Both renderers expose ``render_frame(stat_map, view, ...)`` returning
PNG bytes. Animation orchestration lives in
:mod:`scripts.animate_cortical_maps` (assemble frames into GIF / MP4);
the renderer cares only about one frame at a time.

Output formats
--------------

* PNG: returned as bytes; callers write to disk or pipe into PIL.
* GIF / MP4: assembled by the orchestrator from PNG frame bytes.
* HTML: only ``PlotlyRenderer.render_html()`` produces this, free
  with the plotly engine.
"""

from __future__ import annotations

import abc
import io
import logging
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Canonical fsaverage hierarchy. fsaverageK is built from fsaverage(K-1)
# by quad-splitting each triangle, so vertex K of fsaverageN is identical
# in surface position to vertex K of fsaverage(N+1) for K < n_verts(N).
# This lets us truncate higher-resolution data down to a coarser mesh
# without interpolation.
MESH_VERTS_PER_HEMI: dict[str, int] = {
    "fsaverage": 163842,
    "fsaverage7": 163842,
    "fsaverage6": 40962,
    "fsaverage5": 10242,
    "fsaverage4": 2562,
    "fsaverage3": 642,
}


def truncate_to_mesh(stat_map: np.ndarray, mesh: str) -> np.ndarray:
    """Project a higher-res stat map down to a coarser fsaverage level.

    Pure subset truncation; no interpolation. Raises if the requested
    mesh has *more* vertices than the data (would require real upsampling).
    """
    n_target = MESH_VERTS_PER_HEMI[mesh]
    n_data_per_hemi = stat_map.shape[0] // 2
    if n_data_per_hemi == n_target:
        return stat_map
    if n_target > n_data_per_hemi:
        raise ValueError(
            f"data has {n_data_per_hemi} verts/hemi but {mesh!r} expects "
            f"{n_target}; pick a lower-resolution mesh."
        )
    lh = stat_map[:n_target]
    rh = stat_map[n_data_per_hemi : n_data_per_hemi + n_target]
    return np.concatenate([lh, rh])


@dataclass(frozen=True)
class RenderConfig:
    """Renderer-agnostic options.

    Per-engine implementations can ignore options that don't translate.
    Defaults are tuned for slide-quality output.
    """

    mesh: str = "fsaverage5"
    cmap: str = "cold_hot"
    threshold: float | None = None
    vmin: float | None = None
    vmax: float | None = None
    symmetric_cbar: bool = True
    dpi: int = 120          # matplotlib only; plotly uses width/height
    width: int = 1200       # plotly only
    height: int = 600       # plotly only
    bg_color: str = "white"


class SurfaceRenderer(abc.ABC):
    """Abstract base. Implementations render single frames."""

    name: str

    def __init__(self, mesh: str = "fsaverage5"):
        self.mesh = mesh
        self._fs = None  # cache fetch_surf_fsaverage result

    def _fsaverage(self):
        if self._fs is None:
            from nilearn.datasets import fetch_surf_fsaverage  # lazy
            self._fs = fetch_surf_fsaverage(mesh=self.mesh)
        return self._fs

    @abc.abstractmethod
    def render_frame(
        self,
        stat_map: np.ndarray,
        view: tuple[float, float],
        hemi: Literal["left", "right", "both"],
        config: RenderConfig,
    ) -> bytes:
        """Render one frame at the given (elev, azim) and return PNG bytes."""
        ...

    def render_static_panels(
        self,
        stat_map: np.ndarray,
        views: Sequence[tuple[str, tuple[float, float]]],
        hemis: Sequence[Literal["left", "right"]],
        config: RenderConfig,
    ) -> bytes:
        """Default multi-panel renderer: place each (hemi, view) in a row.

        Subclasses may override for engine-specific multi-panel layouts.
        """
        # Default falls back to N independent frames combined with PIL.
        from PIL import Image  # lazy
        if len(views) != len(hemis):
            raise ValueError("views and hemis must have the same length")
        frames = [
            Image.open(io.BytesIO(self.render_frame(stat_map, ang, h, config)))
            for (_label, ang), h in zip(views, hemis)
        ]
        # Horizontal strip.
        widths = [f.width for f in frames]
        max_h = max(f.height for f in frames)
        canvas = Image.new("RGB", (sum(widths), max_h), config.bg_color)
        x = 0
        for f in frames:
            canvas.paste(f, (x, 0))
            x += f.width
        buf = io.BytesIO()
        canvas.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def _vmin_vmax(self, stat_map: np.ndarray, config: RenderConfig) -> tuple[float, float]:
        if config.vmin is not None and config.vmax is not None:
            return float(config.vmin), float(config.vmax)
        finite = np.isfinite(stat_map)
        if not finite.any():
            return -1.0, 1.0
        vmax = float(np.nanmax(np.abs(stat_map))) if config.symmetric_cbar else float(np.nanmax(stat_map))
        vmin = -vmax if config.symmetric_cbar and vmax > 0 else float(np.nanmin(stat_map))
        return vmin, vmax


class MatplotlibRenderer(SurfaceRenderer):
    """matplotlib 3D backend. Pure CPU, slow but always available."""

    name = "matplotlib"

    def render_frame(self, stat_map, view, hemi, config):
        import matplotlib.pyplot as plt  # lazy
        from nilearn.plotting import plot_surf_stat_map  # lazy

        truncated = truncate_to_mesh(stat_map, config.mesh)
        n_per_hemi = truncated.shape[0] // 2
        lh, rh = truncated[:n_per_hemi], truncated[n_per_hemi:]

        fs = self._fsaverage()
        vmin, vmax = self._vmin_vmax(truncated, config)

        if hemi == "both":
            fig, axarr = plt.subplots(
                1, 2, figsize=(10, 5),
                subplot_kw={"projection": "3d"},
                gridspec_kw={"wspace": 0, "hspace": 0},
            )
            for ax, h, data, mesh_geom, sulc in [
                (axarr[0], "left",  lh, fs.infl_left,  fs.sulc_left),
                (axarr[1], "right", rh, fs.infl_right, fs.sulc_right),
            ]:
                plot_surf_stat_map(
                    mesh_geom, data, bg_map=sulc,
                    hemi=h, view=view,
                    cmap=config.cmap, vmin=vmin, vmax=vmax,
                    threshold=config.threshold, colorbar=False,
                    axes=ax, engine="matplotlib",
                )
        else:
            data = lh if hemi == "left" else rh
            mesh_geom = fs.infl_left if hemi == "left" else fs.infl_right
            sulc = fs.sulc_left if hemi == "left" else fs.sulc_right
            fig, ax = plt.subplots(
                figsize=(6, 6), subplot_kw={"projection": "3d"},
            )
            plot_surf_stat_map(
                mesh_geom, data, bg_map=sulc,
                hemi=hemi, view=view,
                cmap=config.cmap, vmin=vmin, vmax=vmax,
                threshold=config.threshold, colorbar=False,
                axes=ax, engine="matplotlib",
            )

        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=config.dpi,
            bbox_inches="tight", facecolor=config.bg_color,
        )
        plt.close(fig)
        buf.seek(0)
        return buf.read()


class PlotlyRenderer(SurfaceRenderer):
    """plotly + kaleido backend. WebGL-rendered, GPU-accelerated.

    Requires the ``[viz]`` extras (``plotly`` and ``kaleido``). Falls
    back gracefully via :func:`make_renderer` when those aren't
    installed.
    """

    name = "plotly"

    def render_frame(self, stat_map, view, hemi, config):
        import plotly.io as pio  # lazy
        truncated = truncate_to_mesh(stat_map, config.mesh)
        fig = self._build_figure(truncated, view, hemi, config)
        # kaleido produces a static PNG of the WebGL scene.
        return pio.to_image(fig, format="png", width=config.width, height=config.height)

    def render_html(self, stat_map, view, hemi, config) -> str:
        """Plotly-only convenience: return a self-contained HTML string
        containing an interactive 3D figure the user can rotate / zoom."""
        truncated = truncate_to_mesh(stat_map, config.mesh)
        fig = self._build_figure(truncated, view, hemi, config)
        return fig.to_html(include_plotlyjs="cdn", full_html=True)

    def _build_figure(self, truncated, view, hemi, config):
        from nilearn.plotting import plot_surf_stat_map  # lazy
        n_per_hemi = truncated.shape[0] // 2
        lh, rh = truncated[:n_per_hemi], truncated[n_per_hemi:]
        fs = self._fsaverage()
        vmin, vmax = self._vmin_vmax(truncated, config)

        # nilearn's plotly engine returns a plotly Figure already; we
        # lift the camera to (elev, azim) and tweak background.
        if hemi == "both":
            # nilearn doesn't natively render both hemispheres in one
            # plotly figure; build LH and RH separately and combine
            # via plotly subplot grid.
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "scene"}, {"type": "scene"}]],
                horizontal_spacing=0,
            )
            for col, (h, data, mesh_geom, sulc) in enumerate([
                ("left",  lh, fs.infl_left,  fs.sulc_left),
                ("right", rh, fs.infl_right, fs.sulc_right),
            ], start=1):
                sub = plot_surf_stat_map(
                    mesh_geom, data, bg_map=sulc,
                    hemi=h, view=view,
                    cmap=config.cmap, vmin=vmin, vmax=vmax,
                    threshold=config.threshold, colorbar=False,
                    engine="plotly",
                )
                # nilearn returns a PlotlySurfaceFigure; pull its underlying figure.
                inner = getattr(sub, "figure", sub)
                for trace in inner.data:
                    fig.add_trace(trace, row=1, col=col)
        else:
            data = lh if hemi == "left" else rh
            mesh_geom = fs.infl_left if hemi == "left" else fs.infl_right
            sulc = fs.sulc_left if hemi == "left" else fs.sulc_right
            sub = plot_surf_stat_map(
                mesh_geom, data, bg_map=sulc,
                hemi=hemi, view=view,
                cmap=config.cmap, vmin=vmin, vmax=vmax,
                threshold=config.threshold, colorbar=False,
                engine="plotly",
            )
            fig = getattr(sub, "figure", sub)

        # Quality defaults across all scenes.
        fig.update_layout(
            paper_bgcolor=config.bg_color,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
        return fig


def make_renderer(
    engine: Literal["auto", "matplotlib", "plotly"] = "auto",
    mesh: str = "fsaverage5",
) -> SurfaceRenderer:
    """Factory. ``engine='auto'`` picks plotly when available, else matplotlib.

    Explicit ``engine='plotly'`` will raise ``ImportError`` if the
    ``[viz]`` extras aren't installed; that's the right behavior for
    callers who specifically asked for it.
    """
    if engine == "matplotlib":
        return MatplotlibRenderer(mesh=mesh)
    if engine == "plotly":
        try:
            import plotly  # noqa: F401
            import kaleido  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "engine='plotly' requires the [viz] extras. "
                "Install them with `pip install cortexlab[viz]` or "
                "`pip install plotly kaleido`."
            ) from e
        return PlotlyRenderer(mesh=mesh)
    if engine == "auto":
        try:
            import plotly  # noqa: F401
            import kaleido  # noqa: F401
            logger.info("auto-selected plotly renderer (GPU/WebGL)")
            return PlotlyRenderer(mesh=mesh)
        except ImportError:
            logger.info(
                "plotly + kaleido not installed; falling back to matplotlib renderer. "
                "Install with `pip install cortexlab[viz]` for GPU acceleration."
            )
            return MatplotlibRenderer(mesh=mesh)
    raise ValueError(f"unknown engine {engine!r}; pick auto|matplotlib|plotly")


__all__ = [
    "MESH_VERTS_PER_HEMI",
    "truncate_to_mesh",
    "RenderConfig",
    "SurfaceRenderer",
    "MatplotlibRenderer",
    "PlotlyRenderer",
    "make_renderer",
]
