# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from neuralset.extractors.neuro import FSAVERAGE_SIZES
from nilearn.datasets import load_fsaverage
from nilearn.plotting import plot_surf_roi, plot_surf_stat_map

from cortexlab.data.loader import get_hcp_roi_indices

from .base import BasePlotBrain
from .utils import get_cmap, get_scalar_mappable, robust_normalize, saturate_colors

VIEW_DICT = {
    "left": (0, 180),
    "right": (0, 0),
    "medial_left": (0, 0),
    "medial_right": (0, 180),
    "dorsal": (90, 0),
    "ventral": (-90, 0),
    "anterior": (0, 90),
    "posterior": (0, -90),
    "posterior_left": (0, -135),
    "posterior_right": (0, -45),
    "posterior_ventral": (-45, -90),
    "posterior_ventral_left": (-10, -135),
}


class PlotBrainNilearn(BasePlotBrain):

    VIEW_DICT: tp.ClassVar[dict] = VIEW_DICT

    def get_fig_axes(self, views):
        if isinstance(views, str):
            views = [views]
        n_rows, n_cols = (1, len(views)) if len(views) <= 4 else (2, len(views) // 2)
        fig, axarr = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2 * n_cols, 2 * n_rows),
            subplot_kw={"projection": "3d"},
            gridspec_kw={"wspace": 0, "hspace": -0.2},
        )
        if len(views) == 1:
            axarr = [axarr]
        else:
            axarr = axarr.flatten()
        return fig, axarr

    def plot_surf(
        self,
        signals: np.ndarray,
        norm_percentile=None,
        colorbar_title: str | None = None,
        alpha_cmap: tp.Tuple[float, float] | None = None,
        axes: tp.Any | None = None,
        colorbar_kwargs: dict | None = None,
        views: str | list[str] | list[tuple[int, int]] = "left",
        annotated_rois: list[str] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric_cbar: bool = False,
        threshold: float | None = None,
        cmap: str = "hot",
        colorbar: bool = False,
    ):
        if isinstance(views, str):
            views = [views]
        if axes is None:
            fig, axarr = self.get_fig_axes(views=views)
        else:
            views, axarr = self.get_axarr_and_views(axes, views)
            fig = None

        if self.atlas_name is not None:
            signals = self.atlas_to_surf(signals)
        elif signals.ndim == 3:
            signals = self.vol_to_surf(signals)
        assert (
            signals.shape[0] // 2 in FSAVERAGE_SIZES.values()
        ), f"Incoherent number of vertices: {signals.shape[0]}"
        if norm_percentile is not None:
            signals = robust_normalize(signals, percentile=norm_percentile)
        hemis = self.get_hemis(signals)
        if str(signals.dtype).startswith("int"):
            plot_fn = plot_surf_roi
            for k in hemis:
                hemis[k]["roi_map"] = hemis[k].pop("stat_map")
            sm = None
        else:
            plot_fn = plot_surf_stat_map
            cmap = get_cmap(cmap, alpha_cmap=alpha_cmap)
            sm = get_scalar_mappable(
                signals,
                cmap,
                vmin=vmin,
                vmax=vmax,
                threshold=threshold,
                symmetric_cbar=symmetric_cbar,
            )
        for i, (view, ax) in enumerate(zip(views, axarr)):
            selected_hemi = (
                "left"
                if view in ["left", "medial_left"]
                else "right" if view in ["right", "medial_right"] else "both"
            )
            if isinstance(view, str):
                view = VIEW_DICT[view]
            plot_kwargs = {
                "axes": ax,
                "view": view,
                "figure": fig,
                "bg_on_data": (
                    False
                    if (alpha_cmap is not None or plot_fn == plot_surf_roi)
                    else True
                ),
                "cmap": cmap,
                "vmin": vmin,
                "vmax": vmax,
                "threshold": threshold,
                "colorbar": False,
            }
            if plot_fn == plot_surf_stat_map:
                plot_kwargs["symmetric_cbar"] = symmetric_cbar
            plot_fn(**hemis[selected_hemi], **plot_kwargs)
            if annotated_rois is not None:
                self.annotate_rois(ax, annotated_rois, hemi=selected_hemi)
            ax.set_box_aspect(None, zoom=1.4)

        if colorbar:
            if fig is None:
                cbar = plt.colorbar(
                    sm,
                    format="{x:0.2f}",
                    label=colorbar_title,
                    ax=axarr[-1],
                    **colorbar_kwargs if colorbar_kwargs is not None else {},
                    shrink=0.5,
                )
            else:
                cb_ax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
                cbar = fig.colorbar(
                    sm,
                    format="{x:0.2f}",
                    label=colorbar_title,
                    cax=cb_ax,
                    **colorbar_kwargs if colorbar_kwargs is not None else {},
                )
        return sm

    def plot_surf_rgb(
        self,
        signals: tp.List[np.ndarray],
        alpha_signals: np.ndarray | None = None,
        norm_percentile=95,
        alpha_bg=0,
        cmap: tp.Literal["rgb", "rgb_argmax", "tab10"] = "rgb",
        saturation_factor: None | float = None,
        save_path: str | None = None,
        axes: tp.List[matplotlib.axes.Axes] | None = None,
        views: list[str] | list[tuple[int, int]] = ["left"],
        bg_on_data=False,
    ):
        if isinstance(views, str):
            views = [views]
        if axes is None:
            fig, axarr = self.get_fig_axes(views=views)
        else:
            views, axarr = self.get_axarr_and_views(axes, views)
            fig = None

        fsaverage_meshes = load_fsaverage(mesh=self.mesh)
        if self.atlas_name is not None:
            signals = [self.atlas_to_surf(signal) for signal in signals]
        elif signals[0].ndim == 4:
            signals = [self.vol_to_surf(signal) for signal in signals]
        for signal in signals:
            assert (
                signal.shape[0] // 2 in FSAVERAGE_SIZES.values()
            ), f"Incoherent number of vertices: {signal.shape[0]//2}"
        hemis = [self.get_hemis(signal) for signal in signals]
        if alpha_signals is not None:
            alpha_hemis = self.get_hemis(alpha_signals)
        data = dict()
        for selected_hemis in ("left", "right", "both"):
            vertices, faces = hemis[0][selected_hemis]["surf_mesh"]
            colors = np.stack(
                [hemi[selected_hemis]["stat_map"] for hemi in hemis], axis=1
            )
            if cmap.startswith("rgb"):
                if len(signals) == 2:
                    colors = np.concatenate(
                        [colors, np.zeros((colors.shape[0], 1))], axis=1
                    )
                assert colors.shape[1] == 3
                if "argmax" in cmap:
                    colors = robust_normalize(colors, axis=1, percentile=100)
                    func = np.vectorize(lambda color: 0 if color < 1 else 1)
                    colors = func(colors)
                if norm_percentile is not None:
                    colors = robust_normalize(
                        colors, percentile=norm_percentile, two_sided=False
                    )
                if saturation_factor is not None:
                    colors = saturate_colors(colors, saturation_factor)
                colors = np.concatenate([colors, np.ones((colors.shape[0], 1))], axis=1)
            else:
                indices = np.argmax(colors, axis=1)
                cm = get_cmap(cmap)
                colors = cm(indices - 1)
                colors[indices == 0, :3] = np.zeros_like(colors[indices == 0, :3])
            if alpha_signals is not None:
                alpha = alpha_hemis[selected_hemis]["stat_map"]
                alpha_bg = 1 - alpha[:, None]

            bg = hemis[0][selected_hemis]["bg_map"]
            cmap_bg = plt.get_cmap("gray_r")
            bg = robust_normalize(bg, percentile=100)
            bg = cmap_bg(bg)
            if bg_on_data:
                colors[:, :3] = colors[:, :3] * bg[:, :3]
            else:
                colors[:, :3] = colors[:, :3] * (1 - alpha_bg) + bg[:, :3] * alpha_bg
            face_colors = np.mean(colors[faces], axis=1)
            data[selected_hemis] = dict(
                vertex_colors=colors,
                face_colors=face_colors,
                vertices=vertices,
                faces=faces,
            )

        for view, ax in zip(views, axarr):
            selected_hemis = (
                "left" if "left" in view else "right" if "right" in view else "both"
            )
            colors = data[selected_hemis]["face_colors"]
            vertices = data[selected_hemis]["vertices"]
            faces = data[selected_hemis]["faces"]

            p3dcollec = ax.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=faces,
                linewidth=0.1,
                antialiased=False,
                color="white",
            )
            ax.set_box_aspect(None, zoom=1.4)
            limits = [vertices.min(), vertices.max()]
            ax.set_xlim(*limits)
            ax.set_ylim(*limits)
            p3dcollec.set_facecolors(colors)
            ax.set_axis_off()
            ax.view_init(*VIEW_DICT[view])
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path.with_suffix(".npy"), colors)

        return data["both"]["vertex_colors"]

    def save_gif(self, ax, save_path: str | None = None):
        import matplotlib.animation as animation

        if save_path is None:
            save_path = "rgb_animation.gif"

        angles = np.linspace(0, 360, 100, endpoint=False)

        def animate(i):
            ax.view_init(elev=0, azim=angles[i])
            return (ax,)

        from matplotlib.animation import FuncAnimation

        ani = FuncAnimation(ax.figure, animate, frames=len(angles), interval=30)
        writer = animation.PillowWriter(fps=30, bitrate=1800)
        ani.save(save_path, writer=writer)

    def annotate_rois(
        self,
        ax,
        rois: str | list[str] | dict[str, list[str]],
        hemi: str = "left",
        **kwargs,
    ):
        if isinstance(rois, str):
            rois = [rois]
        assert hemi in ["left", "right"]
        data = np.zeros(2 * FSAVERAGE_SIZES[self.mesh])
        vertices = self.get_hemis(data)["both"]["surf_mesh"][0]
        if hemi == "left":
            vertices = vertices[: FSAVERAGE_SIZES[self.mesh]]
        else:
            vertices = vertices[FSAVERAGE_SIZES[self.mesh] :]
        for roi in rois:
            vertex_indices = get_hcp_roi_indices(roi, mesh=self.mesh, hemi=hemi)
            roi_center = vertices[vertex_indices].mean(axis=0)
            roi_name = rois[roi] if isinstance(rois, dict) else roi
            ax.text(roi_center[0], roi_center[1], roi_center[2], roi_name, **kwargs)
