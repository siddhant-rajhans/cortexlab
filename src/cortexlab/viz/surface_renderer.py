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
        # For one-sided perceptual colormaps (hot, viridis, plasma, ...) the
        # symmetric-cbar convention wastes half the cmap below zero where
        # there's no data, dimming everything. Detect and override.
        one_sided_cmaps = {"hot", "viridis", "plasma", "magma", "inferno",
                            "Reds", "Oranges", "YlOrRd", "afmhot", "gist_heat"}
        symmetric = config.symmetric_cbar and config.cmap not in one_sided_cmaps
        if symmetric:
            vmax = float(np.nanmax(np.abs(stat_map)))
            vmin = -vmax if vmax > 0 else float(np.nanmin(stat_map))
        else:
            vmax = float(np.nanmax(stat_map))
            data_min = float(np.nanmin(stat_map))
            vmin = max(0.0, data_min) if config.cmap in one_sided_cmaps else data_min
        return vmin, vmax


class MatplotlibRenderer(SurfaceRenderer):
    """matplotlib 3D backend. Pure CPU, slow but always available."""

    name = "matplotlib"

    def render_frame(self, stat_map, view, hemi, config):
        # Force a headless backend before importing pyplot. Necessary
        # in test suites and pipelines where another renderer (e.g.
        # PyVista) may have left matplotlib's interactive backend in a
        # half-initialized state.
        import matplotlib  # lazy
        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)
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


class PyVistaRenderer(SurfaceRenderer):
    """PyVista + VTK + OpenGL backend. The TRIBE-quality path.

    Mirrors the recipe published by Meta's TRIBE v2 cortical plotting
    code: smooth-shaded mesh on a sulcal-depth background, rendered
    off-screen at dpi-supersampled resolution, then alpha-cropped.
    Real OpenGL via VTK uses any available GPU (NVIDIA / AMD / Intel
    integrated / Apple Metal); falls back to software-OSMesa on a
    GPU-less host but stays correct.

    Why this exists when matplotlib + plotly already cover the static
    panel cases: matplotlib's 3D backend is flat-shaded with no
    proper specular highlights, so the inflated cortex looks washed
    out. Plotly's WebGL is correct but the camera setup we use
    flattens the brain into a 2D silhouette in our subplot layout.
    PyVista lets us set ``view_vector`` directly, so the camera
    behaves the way published brain figures expect.

    Requires the ``[plotting]`` extras (which ship pyvista already)
    or an explicit ``pip install pyvista``.
    """

    name = "pyvista"

    # TRIBE-quality defaults. The dpi=3000 supersample is the single
    # biggest visual upgrade over nilearn's default rendering.
    dpi: int = 3000
    bg_darkness: float = 0.0
    ambient: float = 0.3
    w_pad: float = 0.03
    h_pad: float = 0.03

    # Camera vectors (toward, viewup) in VTK's right-handed coords.
    # Match TRIBE's VIEW_DICT exactly so output orientation is identical.
    VIEW_DICT: dict[str, tuple[list[int], list[int]]] = {
        "lateral_left":   ([-1, 0, 0], [0, 0, 1]),
        "lateral_right":  ([1, 0, 0],  [0, 0, 1]),
        "medial_left":    ([1, 0, 0],  [0, 0, 1]),
        "medial_right":   ([-1, 0, 0], [0, 0, 1]),
        "dorsal":         ([0, 0, 1],  [0, 1, 0]),
        "ventral":        ([0, 0, -1], [1, 0, 0]),
        "anterior":       ([0, 1, 0],  [0, 0, -1]),
        "posterior":      ([0, -1, 0], [0, 0, 1]),
    }

    def render_frame(self, stat_map, view, hemi, config):
        """Render a single frame.

        ``view`` for this renderer is interpreted as a string name
        ('lateral', 'medial', 'dorsal', etc) when given as such, OR
        as a (elev, azim) tuple in degrees that we convert to a
        view_vector. For animation, the (elev, azim) form lets the
        existing animation pipeline rotate around the vertical axis.
        """
        truncated = truncate_to_mesh(stat_map, config.mesh)
        n_per_hemi = truncated.shape[0] // 2
        lh, rh = truncated[:n_per_hemi], truncated[n_per_hemi:]

        if hemi == "both":
            from PIL import Image
            left_png = self._render_one(lh, "left", view, config)
            right_png = self._render_one(rh, "right", view, config)
            left_img = Image.open(io.BytesIO(left_png)).convert("RGB")
            right_img = Image.open(io.BytesIO(right_png)).convert("RGB")
            max_h = max(left_img.height, right_img.height)
            canvas = Image.new(
                "RGB",
                (left_img.width + right_img.width, max_h),
                config.bg_color,
            )
            canvas.paste(left_img, (0, 0))
            canvas.paste(right_img, (left_img.width, 0))
            buf = io.BytesIO()
            canvas.save(buf, format="PNG", optimize=True)
            return buf.getvalue()

        data = lh if hemi == "left" else rh
        return self._render_one(data, hemi, view, config)

    def _render_one(self, hemi_data, hemi, view, config):
        import pyvista as pv  # lazy
        from PIL import Image

        fs = self._fsaverage()
        mesh_path = fs.infl_left if hemi == "left" else fs.infl_right
        sulc_path = fs.sulc_left if hemi == "left" else fs.sulc_right

        # nilearn 0.13+ ships these as GIFTI; load via nilearn.surface.
        from nilearn.surface import load_surf_data, load_surf_mesh
        m = load_surf_mesh(mesh_path)
        vertices = np.asarray(m.coordinates)
        faces = np.asarray(m.faces)
        bg_map = np.asarray(load_surf_data(sulc_path))

        # Composite RGBA stat-map onto sulcal-depth background. Suprathreshold
        # voxels show the colormap; subthreshold show gyri/sulci modulation.
        # NaN voxels (e.g. FDR-masked) become fully transparent so the
        # sulcal background shows through.
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        vmin, vmax = self._vmin_vmax(hemi_data, config)
        cmap = self._thresholded_cmap(config.cmap, threshold=config.threshold,
                                       vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        nan_mask = ~np.isfinite(hemi_data)
        data_filled = np.where(nan_mask, vmin, hemi_data)
        rgba = sm.to_rgba(data_filled).copy()  # cmap output is read-only
        # Force NaN voxels fully transparent so sulcal bg dominates them.
        rgba[nan_mask, 3] = 0.0
        # Force voxels at exactly vmin (or below threshold band) to also
        # be transparent — keeps cleanly-empty areas truly empty.
        if config.threshold is not None and not nan_mask.all():
            below = np.abs(hemi_data) < config.threshold
            rgba[below, 3] = 0.0
        bg_norm = (bg_map - bg_map.min()) / (bg_map.max() - bg_map.min() + 1e-8)
        bg_rgb = 1 - np.column_stack(
            [self.bg_darkness + bg_norm * (1 - self.bg_darkness)] * 3
        )
        colors = rgba[:, 3:4] * rgba[:, :3] + (1 - rgba[:, 3:4]) * bg_rgb

        pv_faces = np.column_stack([np.full(len(faces), 3), faces])
        surf = pv.PolyData(vertices, pv_faces)
        surf.point_data["colors"] = colors

        # Render off-screen at supersampled resolution. The window_size
        # is 1:1 pixels — Pillow's tight_crop is what tightens the result.
        w_px = max(1200, config.width or 1200)
        h_px = max(800, config.height or 800)
        pl = pv.Plotter(window_size=[w_px, h_px], off_screen=True)
        pl.add_mesh(
            surf, scalars="colors", rgb=True,
            smooth_shading=True, ambient=self.ambient,
        )
        pl.set_background(config.bg_color)
        vec, up = self._view_vector(view, hemi)
        pl.view_vector(vec, viewup=up)
        img = pl.screenshot(return_img=True)
        pl.close()

        # Tight-crop transparent / matching-bg edges for clean panels.
        cropped = self._tight_crop(img, config.bg_color, self.w_pad, self.h_pad)
        buf = io.BytesIO()
        Image.fromarray(cropped).save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def _view_vector(self, view, hemi):
        """Translate either a named view, a (camera, viewup) tuple, or a
        matplotlib-style (elev, azim) tuple into VTK camera vectors.

        The (elev, azim) lateral/medial angles used by
        :mod:`scripts.plot_cortical_maps` are hemisphere-aware in
        nilearn's matplotlib backend (the same numeric pair means
        different things for LH vs RH). PyVista's view_vector is
        absolute in 3D space, so we look up the matching named view
        for the four canonical static panels rather than running a
        generic angle-to-vector formula. For arbitrary rotations
        (animation) we fall through to the formula.
        """
        if isinstance(view, str):
            key = view if view in self.VIEW_DICT else f"lateral_{hemi}"
            return self.VIEW_DICT[key]
        if isinstance(view, tuple) and len(view) == 2 and isinstance(view[0], list):
            return view  # already (vec, up)
        elev, azim = view
        # Map nilearn's canonical static-panel angles to absolute named
        # views, hemisphere-aware. Covers the 4-panel layout used by
        # `plot_cortical_maps.py` — anything else falls through to the
        # generic angle-to-vector path below for animation.
        STATIC: dict[tuple[float, float], dict[str, str]] = {
            (0.0, 180.0):  {"left": "lateral_left",  "right": "medial_right"},
            (0.0,   0.0):  {"left": "medial_left",   "right": "lateral_right"},
            (90.0,  0.0):  {"left": "dorsal",        "right": "dorsal"},
            (-90.0, 0.0):  {"left": "ventral",       "right": "ventral"},
        }
        key = (round(float(elev), 1), round(float(azim), 1))
        named = STATIC.get(key, {}).get(hemi)
        if named is not None:
            return self.VIEW_DICT[named]
        # General (elev, azim) -> unit vector for animation rotation.
        rad_e = np.deg2rad(elev)
        rad_a = np.deg2rad(azim)
        x = np.cos(rad_e) * np.cos(rad_a)
        y = np.cos(rad_e) * np.sin(rad_a)
        z = np.sin(rad_e)
        return [x, y, z], [0, 0, 1]

    def _thresholded_cmap(self, name, threshold, vmin, vmax):
        """Build a colormap that shows neutral gray for |val| < threshold
        rather than fading to background. Matches the TRIBE recipe.
        """
        from matplotlib import colormaps
        from matplotlib.colors import LinearSegmentedColormap, to_rgba
        base = colormaps[name]
        if threshold is None or vmax <= vmin:
            return base
        n = 256
        xs = np.linspace(0, 1, n)
        colors = base(xs)
        # Neutral gray for values whose Norm-mapped position falls in
        # the |val| < threshold band.
        if vmin < 0:  # symmetric
            band = threshold / max(abs(vmin), abs(vmax))
            mask = np.abs(xs - 0.5) < band / 2
        else:
            band = threshold / (vmax - vmin)
            mask = xs < band
        colors[mask] = [0.5, 0.5, 0.5, 1.0]
        return LinearSegmentedColormap.from_list(f"{name}_thr", colors)

    def _tight_crop(self, img: np.ndarray, bg_color: str, w_pad: float, h_pad: float):
        """Trim padding by finding the bounding box of pixels that
        differ from the background color."""
        from PIL import ImageColor
        bg = np.array(ImageColor.getrgb(bg_color), dtype=np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        diff = np.any(img != bg, axis=-1)
        ys, xs = np.where(diff)
        if ys.size == 0:
            return img
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        h, w = img.shape[:2]
        py = int((y1 - y0) * h_pad)
        px = int((x1 - x0) * w_pad)
        y0, y1 = max(0, y0 - py), min(h, y1 + py + 1)
        x0, x1 = max(0, x0 - px), min(w, x1 + px + 1)
        return img[y0:y1, x0:x1]


def make_renderer(
    engine: Literal["auto", "matplotlib", "plotly", "pyvista"] = "auto",
    mesh: str = "fsaverage5",
) -> SurfaceRenderer:
    """Factory. ``engine='auto'`` picks plotly when available, else matplotlib.

    Explicit ``engine='plotly'`` will raise ``ImportError`` if the
    ``[viz]`` extras aren't installed; that's the right behavior for
    callers who specifically asked for it.
    """
    if engine == "matplotlib":
        return MatplotlibRenderer(mesh=mesh)
    if engine == "pyvista":
        try:
            import pyvista  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "engine='pyvista' requires pyvista. "
                "Install with `pip install cortexlab[plotting]` or `pip install pyvista`."
            ) from e
        return PyVistaRenderer(mesh=mesh)
    if engine == "plotly":
        try:
            import kaleido  # noqa: F401
            import plotly  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "engine='plotly' requires the [viz] extras. "
                "Install them with `pip install cortexlab[viz]` or "
                "`pip install plotly kaleido`. "
                "Kaleido v1+ also needs an external Chrome binary; "
                "fetch it with `plotly_get_chrome` after install."
            ) from e
        return PlotlyRenderer(mesh=mesh)
    if engine == "auto":
        # Prefer PyVista when available — it produces TRIBE-quality
        # smooth-shaded output via real OpenGL on any GPU.
        try:
            import pyvista  # noqa: F401
            logger.info("auto-selected pyvista renderer (OpenGL)")
            return PyVistaRenderer(mesh=mesh)
        except ImportError:
            pass
        try:
            import kaleido  # noqa: F401
            import plotly  # noqa: F401
        except ImportError:
            logger.info(
                "neither pyvista nor plotly+kaleido installed; falling back to "
                "matplotlib renderer. Install with `pip install cortexlab[plotting]` "
                "for the PyVista path."
            )
            return MatplotlibRenderer(mesh=mesh)
        # kaleido v1+ depends on an external Chrome install. Imports succeed
        # but the first render raises ``ChromeNotFoundError`` if Chrome is
        # missing. Probe with a 1x1 dummy figure so the fallback decision
        # happens up-front rather than mid-animation.
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            pio.to_image(
                go.Figure(data=[go.Bar(x=[0], y=[0])]),
                format="png", width=10, height=10,
            )
        except Exception as e:  # noqa: BLE001 — any failure -> fallback
            logger.info(
                "plotly/kaleido installed but render probe failed (%s); "
                "falling back to matplotlib renderer. Run `plotly_get_chrome` "
                "to fetch the headless Chrome binary kaleido v1+ requires.",
                type(e).__name__,
            )
            return MatplotlibRenderer(mesh=mesh)
        logger.info("auto-selected plotly renderer (GPU/WebGL)")
        return PlotlyRenderer(mesh=mesh)
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
