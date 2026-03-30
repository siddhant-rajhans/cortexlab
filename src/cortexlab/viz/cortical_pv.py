# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from neuralset.extractors.neuro import FSAVERAGE_SIZES

from cortexlab.data.loader import get_hcp_roi_indices

from .base import BasePlotBrain
from .utils import (
    convert_ax_to_2d,
    get_cmap,
    get_scalar_mappable,
    robust_normalize,
    saturate_colors,
    tight_crop,
)

VIEW_DICT = {
    "ventral": ([0, 0, -1], [1, 0, 0]),
    "dorsal": ([0, 0, 1], [0, 1, 0]),
    "left": ([-1, 0, 0], [0, 0, 1]),
    "right": ([1, 0, 0], [0, 0, 1]),
    "anterior": ([0, 1, 0], [0, 0, -1]),
    "posterior": ([0, -1, 0], [0, 0, 1]),
    "medial_left": ([1, 0, 0], [0, 0, 1]),
    "medial_right": ([-1, 0, 0], [0, 0, 1]),
    "posterior_left": ([-1, 0, 0], [0, 0, 1]),
    "posterior_right": ([-1, 0, 0], [0, 0, 1]),
}


class PlotBrainPyvista(BasePlotBrain):

    dpi: int = 3000
    bg_darkness: float = 0
    ambient: float = 0.3
    w_pad: float = 0.03
    h_pad: float = 0.03

    VIEW_DICT: tp.ClassVar[dict] = VIEW_DICT

    def _convert_ax(self, ax):
        return convert_ax_to_2d(ax)

    def annotate_rois(
        self,
        pl: pv.Plotter,
        rois: str | list[str] | dict[str, str],
        hemi: str = "left",
        **kwargs,
    ):
        if isinstance(rois, str):
            rois = [rois]
        hemis = ["left", "right"] if hemi == "both" else [hemi]
        n = FSAVERAGE_SIZES[self.mesh]
        for h in hemis:
            verts = self._mesh[h]["coords"]
            for roi in rois:
                idx = get_hcp_roi_indices(roi, mesh=self.mesh, hemi=h)
                if h == "right":
                    idx = np.array(idx) - n
                center = verts[idx].mean(axis=0)
                name = rois[roi] if isinstance(rois, dict) else roi
                pl.add_point_labels(
                    center.reshape(1, 3),
                    [name],
                    shape_opacity=0,
                    **kwargs,
                )

    def plot_surf(
        self,
        data,
        axes,
        views="left",
        alpha_cmap=None,
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric_cbar: bool = False,
        threshold: float | None = None,
        cmap: str = "hot",
        norm_percentile: float | None = None,
        annotated_rois: str | list[str] | dict | None = None,
        annotated_rois_kwargs: dict | None = None,
    ):
        if norm_percentile is not None:
            data = robust_normalize(data, percentile=norm_percentile)
        if isinstance(views, str):
            views = [views]
        views, axes = self.get_axarr_and_views(axes, views)
        cmap = get_cmap(cmap, alpha_cmap=alpha_cmap)
        sm = get_scalar_mappable(
            data,
            cmap,
            vmin=vmin,
            vmax=vmax,
            threshold=threshold,
            symmetric_cbar=symmetric_cbar,
        )

        stat_maps = self.get_stat_map(data)

        for ax, view in zip(axes, views):
            selected_hemi = (
                "left"
                if view in ["left", "medial_left"]
                else "right" if view in ["right", "medial_right"] else "both"
            )
            mesh = self._mesh[selected_hemi]
            vertices, faces = mesh["coords"], mesh["faces"]
            stat_map = stat_maps[selected_hemi]

            rgba = sm.to_rgba(stat_map)
            bg_map = mesh["bg_map"]
            bg_norm = (bg_map - bg_map.min()) / (bg_map.max() - bg_map.min() + 1e-8)
            bg_rgb = 1 - np.column_stack(
                [self.bg_darkness + bg_norm * (1 - self.bg_darkness)] * 3
            )
            colors = rgba[:, 3:4] * rgba[:, :3] + (1 - rgba[:, 3:4]) * bg_rgb

            pv_faces = np.column_stack([np.full(len(faces), 3), faces])

            ax_size = ax.get_position()
            pl = pv.Plotter(
                window_size=[
                    int(ax_size.width * self.dpi),
                    int(ax_size.height * self.dpi),
                ],
                off_screen=True,
            )

            surf = pv.PolyData(vertices, pv_faces)
            surf.point_data["colors"] = colors
            pl.add_mesh(
                surf,
                scalars="colors",
                rgb=True,
                smooth_shading=True,
                ambient=self.ambient,
            )

            pl.set_background("white")
            vec, up = VIEW_DICT[view]
            pl.view_vector(vec, viewup=up)
            if annotated_rois is not None:
                self.annotate_rois(
                    pl,
                    annotated_rois,
                    **(annotated_rois_kwargs or {}),
                )
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                img = pl.screenshot(tmp.name, return_img=True)
            img = tight_crop(img, w_pad=self.w_pad, h_pad=self.h_pad)
            pl.clear()
            ax.axis("off")
            ax.imshow(img, aspect="equal")

        return sm

    def plot_surf_rgb(
        self,
        signals: tp.List[np.ndarray],
        alpha_signals: np.ndarray | None = None,
        norm_percentile=95,
        alpha_bg=0,
        cmap: tp.Literal["rgb", "rgb_argmax", "tab10"] = "rgb",
        saturation_factor: None | float = None,
        axes=None,
        views: list[str] = ["left"],
        bg_on_data=False,
    ):
        if isinstance(views, str):
            views = [views]
        views, axes = self.get_axarr_and_views(axes, views)

        if self.atlas_name is not None:
            signals = [self.atlas_to_surf(signal) for signal in signals]
        elif signals[0].ndim == 4:
            signals = [self.vol_to_surf(signal) for signal in signals]

        hemis = [self.get_hemis(signal) for signal in signals]
        if alpha_signals is not None:
            alpha_hemis = self.get_hemis(alpha_signals)

        data = dict()
        for selected_hemis in ("left", "right", "both"):
            stat_maps = [hemi[selected_hemis]["stat_map"] for hemi in hemis]
            colors = np.stack(stat_maps, axis=1)

            if cmap.startswith("rgb"):
                if len(signals) == 2:
                    colors = np.concatenate(
                        [colors, np.zeros((colors.shape[0], 1))], axis=1
                    )
                assert colors.shape[1] == 3
                if "argmax" in cmap:
                    colors = robust_normalize(colors, axis=1, percentile=100)
                    colors = (colors >= 1).astype(float)
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
                colors[indices == 0, :3] = 0

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

            mesh = self._mesh[selected_hemis]
            data[selected_hemis] = dict(
                vertex_colors=colors,
                vertices=mesh["coords"],
                faces=mesh["faces"],
            )

        for ax, view in zip(axes, views):
            selected_hemis = (
                "left" if "left" in view else "right" if "right" in view else "both"
            )
            d = data[selected_hemis]

            pv_faces = np.column_stack([np.full(len(d["faces"]), 3), d["faces"]])

            ax_size = ax.get_position()
            pl = pv.Plotter(
                window_size=[
                    int(ax_size.width * self.dpi),
                    int(ax_size.height * self.dpi),
                ],
                off_screen=True,
            )

            surf = pv.PolyData(d["vertices"], pv_faces)
            surf.point_data["colors"] = d["vertex_colors"][:, :3]
            pl.add_mesh(
                surf,
                color="black",
                scalars="colors",
                rgb=True,
                smooth_shading=True,
                ambient=0.3,
            )

            vec, up = VIEW_DICT[view]
            pl.view_vector(vec, viewup=up)
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                img = pl.screenshot(
                    tmp.name, return_img=True, transparent_background=True
                )
            img = tight_crop(img, w_pad=self.w_pad, h_pad=self.h_pad)
            pl.clear()
            ax.axis("off")
            ax.imshow(img, aspect="equal")

        return data["both"]["vertex_colors"]
