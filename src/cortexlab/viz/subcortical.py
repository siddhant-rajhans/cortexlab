# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import typing as tp
from functools import lru_cache

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pyvista as pv
import seaborn as sns
from nilearn import datasets
from nilearn.surface import vol_to_surf
from scipy.ndimage import gaussian_filter
from skimage import measure

from cortexlab.viz.utils import (
    get_cmap,
    get_scalar_mappable,
    robust_normalize,
    tight_crop,
)


@lru_cache()
def get_subcortical_mask():
    atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm")
    excluded = ["Cortex", "White", "Stem", "Background"]
    selected_indices = [
        i
        for i, label in enumerate(atlas.labels)
        if any([exc.lower() in label.lower() for exc in excluded])
    ]
    mask_data = atlas.maps.get_fdata()
    mask_data[np.isin(mask_data, selected_indices)] = 0
    mask = nib.Nifti1Image(mask_data, atlas.maps.affine, atlas.maps.header)
    return mask


def get_subcortical_labels(with_hemi: bool = False):
    excluded = ["Cortex", "White", "Stem", "Background"]
    labels = [
        label
        for label in cached_ho_atlas().labels
        if not any([exc.lower() in label.lower() for exc in excluded])
    ]
    if not with_hemi:
        labels = list(
            set(
                [
                    label.replace("Left ", "")
                    for label in labels
                    if label.startswith("Left ")
                ]
            )
        )
    return labels


@lru_cache
def cached_ho_atlas(resolution: tp.Literal["1mm", "2mm"] = "1mm"):
    return datasets.fetch_atlas_harvard_oxford(f"sub-maxprob-thr50-{resolution}")


def get_subcortical_roi_indices(roi: str):
    subcortical_mask = copy.deepcopy(get_subcortical_mask())
    data = subcortical_mask.get_fdata()
    data = data[data > 0]
    ho_sub = cached_ho_atlas(resolution="2mm")
    labels = ho_sub.labels
    sel_labels = [label for label in labels if roi.lower() in label.lower()]
    assert sel_labels, f"ROI {roi} not found in atlas"
    sel_indices = [labels.index(label) for label in sel_labels]
    voxel_indices = np.where(np.isin(data, sel_indices))[0]
    return voxel_indices


def voxel_to_mesh(voxel_scores, label, resolution):
    subcortical_mask = copy.deepcopy(get_subcortical_mask())
    data = subcortical_mask.get_fdata()
    data[data > 0] = voxel_scores
    nii = nib.Nifti1Image(data, subcortical_mask.affine, subcortical_mask.header)
    roi_mask = get_mask(label, resolution)
    mesh = get_mesh(label, resolution)
    return nii_to_mesh(nii, mesh, mask_img=roi_mask)


def nii_to_mesh(nii, mesh, mask_img=None):
    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    vertex_vals = vol_to_surf(
        nii,
        surf_mesh=(vertices, faces),
        mask_img=mask_img,
        kind="line",
        depth=np.linspace(-3, 0, 40),
        interpolation="linear",
    )
    return vertex_vals


@lru_cache()
def get_mask(label: str, resolution: tp.Literal["1mm", "2mm"] = "1mm"):
    # fetch Harvard-Oxford subcortical atlas
    ho_sub = cached_ho_atlas(resolution=resolution)
    img = ho_sub.maps
    if label == "Cerebellum":
        raise NotImplementedError(
            "Cerebellum atlas (Diedrichsen 2009) is not yet supported. "
            "Provide the atlas path manually."
        )
        img = nib.load(file)
        mask = img.get_fdata() > 0  # merge all lobules automatically
    elif label == "Brain-Stem":
        # subcortical, return hemisphere-specific mesh (default: right)
        idx = ho_sub.labels.index(label)
        mask = img.get_fdata() == idx
    else:
        if "Left" in label or "Right" in label:
            idx = ho_sub.labels.index(label)
            mask = img.get_fdata() == idx
        else:
            # merge left + right
            left_idx = ho_sub.labels.index("Left " + label)
            right_idx = ho_sub.labels.index("Right " + label)
            data = img.get_fdata()
            mask = (data == left_idx) | (data == right_idx)

    nii_mask = nib.Nifti1Image(mask.astype(float), img.affine, img.header)

    return nii_mask


@lru_cache()
def get_mesh(label: str, resolution: tp.Literal["1mm", "2mm"]):
    """
    Returns a PyVista mesh for a given label.
    For 'Cerebellum', 'Cerebral Cortex', and 'Brain-Stem', left and right hemispheres are joined.
    For other subcortical labels, returns separate left/right meshes.
    """

    if label == "Cerebral Cortex":
        fsaverage = datasets.fetch_surf_fsaverage("fsaverage7")
        nii = nib.load(fsaverage.pial_left)
        verts = nii.darrays[0].data
        faces = nii.darrays[1].data
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
        mesh = pv.PolyData(verts, faces_pv)
        return mesh

    nii_mask = get_mask(label, resolution)

    # smooth the mask slightly
    volume = gaussian_filter(nii_mask.get_fdata().astype(float), sigma=1)

    # marching cubes
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.9)
    # Convert voxel coordinates to world/MNI coordinates
    affine = nii_mask.affine
    verts = nib.affines.apply_affine(affine, verts)

    # convert faces to PyVista format
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)

    # create PyVista mesh
    mesh = pv.PolyData(verts, faces_pv)

    # smooth the mesh
    mesh = mesh.smooth(n_iter=50, relaxation_factor=0.01)

    return mesh


def plot_subcortical(
    ax,
    *,
    colors: dict = None,
    voxel_scores: np.ndarray = None,
    average_per_roi: bool = False,
    norm_percentile: int = None,
    show_cortex: bool = False,
    show_brain_stem: bool = False,
    show_cerebellum: bool = False,
    explode: float = 0.5,
    resolution: tp.Literal["1mm", "2mm"] = "1mm",
    show_scalar_bar: bool = False,
    zoom: float = 1.3,
    azimuth: float = 15,
    elevation: float = -10,
    intensity: float = 1.5,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric_cbar: bool = False,
    threshold: float | None = None,
    cmap: str = "hot",
    alpha_cmap: tuple[float, float] = None,
    **plot_kwargs,
):
    assert (colors is not None) ^ (
        voxel_scores is not None
    ), "Either colors voxel_scores must be provided"
    labels = get_subcortical_labels(with_hemi=True)
    if colors is not None:
        assert isinstance(colors, dict), "Colors must be a dictionary"
    if voxel_scores is not None:
        assert voxel_scores.ndim in [1, 2], "voxel_scores must be a 1D or 2D array"
        if average_per_roi:
            for label in labels:
                indices = get_subcortical_roi_indices(label)
                voxel_scores[indices] = voxel_scores[indices].mean()
        if norm_percentile:
            voxel_scores = robust_normalize(voxel_scores, percentile=norm_percentile)
    if show_cerebellum:
        labels.append("Cerebellum")
    if show_cortex:
        labels.append("Cerebral Cortex")
    if show_brain_stem:
        labels.append("Brain-Stem")
    plotter = pv.Plotter(lighting="none")
    rgb = False
    cmap = get_cmap(cmap, alpha_cmap=alpha_cmap)
    sm = get_scalar_mappable(
        voxel_scores,
        cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        symmetric_cbar=symmetric_cbar,
    )
    for label in labels:
        mesh = get_mesh(label, resolution)
        if label in ["Cerebral Cortex", "Brain-Stem"]:
            color = plt.cm.gray(0.8)
        else:
            if colors is not None:
                color = colors[label]
                scalars = None
            else:
                assert voxel_scores is not None
                color = plt.cm.gray(0.8)
                if voxel_scores.ndim == 1:
                    scalars = voxel_to_mesh(voxel_scores, label, resolution)
                    scalars = sm.to_rgba(scalars)
                    rgb = True
                elif voxel_scores.ndim == 2:
                    assert voxel_scores.shape[0] == 3
                    scalars = np.stack(
                        [
                            voxel_to_mesh(voxel_scores, label, resolution)
                            for voxel_scores in voxel_scores
                        ],
                        axis=1,
                    )
                    rgb = True
        exploded_points = copy.deepcopy(mesh.points)
        if label == "Cerebral Cortex":
            exploded_points[:, 0] = (
                exploded_points[:, 0] + explode * exploded_points[:, 0].mean()
            )
        else:
            exploded_points[:, 2] = (
                exploded_points[:, 2] + explode * exploded_points.mean(axis=0)[2]
            )
        exploded_mesh = pv.PolyData(exploded_points, mesh.faces)
        plotter.add_mesh(
            exploded_mesh,
            color=color,
            scalars=scalars,
            rgb=rgb,
            show_scalar_bar=show_scalar_bar,
        )
    plotter.window_size = [300, 300]
    plotter.camera.zoom(zoom)
    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
    light = pv.Light(intensity=intensity)
    light.set_headlight()
    plotter.add_light(light)

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        img = plotter.screenshot(tmp.name, return_img=True)
    img = tight_crop(img)
    ax.imshow(img)
    ax.axis("off")
    return sm


if __name__ == "__main__":

    labels = get_subcortical_labels(with_hemi=False)
    palette = sns.color_palette("Set1", n_colors=len(labels))
    colors = {
        f"{hemi} {label}": palette[i]
        for i, label in enumerate(labels)
        for hemi in ["Left", "Right"]
    }
    plotter = plot_subcortical(
        colors=colors,
        average_per_roi=True,
        cmap="fire",
        show_cerebellum=False,
        explode=1,
        resolution="1mm",
        zoom=1.3,
    )
    plt.show()
