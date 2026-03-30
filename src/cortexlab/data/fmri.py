# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import typing as tp
from enum import Enum

import neuralset as ns
import numpy as np
import pydantic
from neuralset.extractors.neuro import FSAVERAGE_SIZES


class _FmriTemplateSpaceSpec(tp.NamedTuple):
    id: str
    shape: tp.Tuple[int, int, int] | None


class FmriTemplateSpace(Enum):
    # MNI - TEMPLATEFLOW (partial)
    # We keep only 1mm-resolution variants as res mapping is handled by vol_to_surf
    MNI152LIN_RES_01 = _FmriTemplateSpaceSpec("tpl-MNI152Lin_res-01", (181, 217, 181))
    MNI152NLIN2009A_ASYM_RES_1 = _FmriTemplateSpaceSpec(
        "tpl-MNI152NLin2009aAsym_res-1", (197, 233, 189)
    )
    MNI152NLIN2009A_SYM_RES_1 = _FmriTemplateSpaceSpec(
        "tpl-MNI152NLin2009aSym_res-1", (197, 233, 189)
    )
    MNI152NLIN2009C_ASYM_RES_01 = _FmriTemplateSpaceSpec(
        "tpl-MNI152NLin2009cAsym_res-01", (193, 229, 193)
    )
    MNI152NLIN2009C_SYM_RES_1 = _FmriTemplateSpaceSpec(
        "tpl-MNI152NLin2009cSym_res-1", (193, 229, 193)
    )
    MNI152NLIN6_ASYM_RES_01 = _FmriTemplateSpaceSpec(
        "tpl-MNI152NLin6Asym_res-01", (182, 218, 182)
    )
    MNI152NLIN6_SYM_RES_01 = _FmriTemplateSpaceSpec(
        "tpl-MNI152NLin6Asym_res-01", (193, 229, 193)
    )
    MNI305 = _FmriTemplateSpaceSpec("tpl-MNI305", (172, 220, 156))
    MNICOLIN27 = _FmriTemplateSpaceSpec("tpl-MNIColin27", (181, 217, 181))

    # FSAVERAGE
    FSAVERAGE = _FmriTemplateSpaceSpec("fsaverage", (163842,))
    FSAVERAGE_6 = _FmriTemplateSpaceSpec("fsaverage6", (40962,))
    FSAVERAGE_5 = _FmriTemplateSpaceSpec("fsaverage5", (10242,))
    FSAVERAGE_4 = _FmriTemplateSpaceSpec("fsaverage4", (2562,))
    FSAVERAGE_3 = _FmriTemplateSpaceSpec("fsaverage3", (642,))

    # CIFTI
    CIFTI_HCP_FS_LR_32K = _FmriTemplateSpaceSpec("cifti-hcp-fs_LR_32k", (59412,))
    CIFTI_HCP_FS_LR_164K = _FmriTemplateSpaceSpec("cifti-hcp-fs_LR_164k", (170494,))

    # NATIVE
    T1W = _FmriTemplateSpaceSpec("T1w", None)

    # OTHER
    MNI_UNKNOWN = _FmriTemplateSpaceSpec("MNI_unknown", None)  # unknown MNI space
    UNKNOWN = _FmriTemplateSpaceSpec("unknown", None)  # unknown space
    CUSTOM = _FmriTemplateSpaceSpec(
        "custom", None
    )  # custom space e.g. provided by study authors


def is_mni_space(space: FmriTemplateSpace) -> bool:
    """
    Check if the given template space is an MNI space.
    """
    return space.name.startswith("MNI")


def load_mni_mesh(
    template: FmriTemplateSpace,
    target_space="fsaverage",
    base_path: str | None = None,
) -> dict:
    """
    Load MNI surface meshes for both hemispheres and white / pial surfaces.

    Parameters
    ----------
    template : FmriTemplateSpace
    target_space : str
    base_path : str or None
        Root directory containing FreeSurfer subjects. If ``None``, reads
        from the ``FREESURFER_SUBJECTS_DIR`` environment variable.

    Returns
    -------
    meshes : dict
        Dictionary with keys like 'pial_left', 'pial_right', 'white_left', 'white_right'
        and values as loaded nilearn surface meshes.
    """
    import os

    if not re.match(r"^fsaverage[3-6]?$", target_space):
        raise ValueError(
            f"target_space must be 'fsaverage' or 'fsaverage3/4/5/6', got '{target_space}'"
        )

    if not is_mni_space(template):
        raise ValueError(
            f"Template {template.value.id} is required to be an MNI space."
        )

    if base_path is None:
        base_path = os.getenv("FREESURFER_SUBJECTS_DIR")
    if base_path is None:
        raise EnvironmentError(
            "Set the FREESURFER_SUBJECTS_DIR environment variable to the "
            "directory containing FreeSurfer subjects, or pass base_path explicitly."
        )

    from nilearn.surface import load_surf_mesh

    mesh_dir = os.path.join(base_path, template.value.id, "surf", "surf_hybrid_mni_gii")
    meshes = {}
    for surf in ["pial", "white"]:
        for hemi in ["left", "right"]:
            mesh_path = os.path.join(mesh_dir, f"{hemi[0]}h.{surf}.{target_space}.gii")
            meshes[f"{surf}_{hemi}"] = load_surf_mesh(mesh_path)
    return meshes


class TribeSurfaceProjector(ns.extractors.neuro.SurfaceProjector):
    """Project data to an fsaverage surface mesh.
    For volumetric data, this uses ``nilearn.surface.vol_to_surf`` to project the data to the surface.
    For surface data, this simply downsamples the data to the target mesh resolution.

    Fields beyond ``mesh`` mirror the keyword arguments of
    ``nilearn.surface.vol_to_surf`` and are forwarded to it.

    Examples
    --------
    >>> SurfaceProjector(mesh="fsaverage5")
    >>> SurfaceProjector(mesh="fsaverage6", radius=5.0, interpolation="nearest")
    """

    mesh: str
    radius: float = 3.0
    interpolation: tp.Literal["linear", "nearest"] = "linear"
    kind: tp.Literal["auto", "line", "ball"] = "auto"
    n_samples: int | None = None
    mask_img: tp.Any | None = None
    depth: list[float] | None = None
    center_depth: float = 1
    extract_fsaverage_from_mni: bool = False

    _mesh: tp.Any | None = pydantic.PrivateAttr(default=None)

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        assert (
            self.center_depth >= 0 and self.center_depth <= 1
        ), "center_depth must be between 0 and 1"
        if self.mesh not in FSAVERAGE_SIZES:
            raise ValueError(f"mesh must be an fsaverage mesh (got {self.mesh!r})")

    def get_mesh(self) -> tp.Any:
        if self._mesh is None:
            if self.extract_fsaverage_from_mni:
                mni_template_spec = FmriTemplateSpace["MNI152NLIN2009C_ASYM_RES_01"]
                fsaverage = load_mni_mesh(mni_template_spec, self.mesh)
            else:
                from nilearn import datasets

                fsaverage = datasets.fetch_surf_fsaverage(self.mesh)
            self._mesh = fsaverage
        return self._mesh

    def get_intermediate_mesh(
        self, hemi: str, center_depth: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        meshes = self.get_mesh()
        surf_mesh, inner_mesh = meshes[f"pial_{hemi}"], meshes[f"white_{hemi}"]
        from nilearn.surface import InMemoryMesh

        if isinstance(surf_mesh, str):
            import nibabel

            surf_vertices, surf_faces = nibabel.load(surf_mesh).darrays
            inner_vertices, inner_faces = nibabel.load(inner_mesh).darrays
            surf_vertices, surf_faces = surf_vertices.data, surf_faces.data
            inner_vertices, inner_faces = inner_vertices.data, inner_faces.data
        elif isinstance(surf_mesh, InMemoryMesh):
            surf_vertices, surf_faces = surf_mesh.coordinates, surf_mesh.faces
            inner_vertices, inner_faces = inner_mesh.coordinates, inner_mesh.faces
        else:
            raise TypeError(f"Unsupported mesh type: {type(surf_mesh)}")
        half_vertices = surf_vertices * center_depth + inner_vertices * (
            1 - center_depth
        )
        half_depth_mesh = (half_vertices, surf_faces)
        return half_depth_mesh

    def apply(self, rec: tp.Any) -> np.ndarray:

        if len(rec.shape) == 4:
            # 4-D volume data → use nilearn.surface.vol_to_surf
            meshes = self.get_mesh()
            from nilearn.surface import vol_to_surf

            hemis = []
            for hemi in ("left", "right"):
                if self.center_depth == 1:
                    surf_mesh = meshes[f"pial_{hemi}"]
                else:
                    surf_mesh = self.get_intermediate_mesh(hemi, self.center_depth)
                hemis.append(
                    vol_to_surf(
                        rec,
                        surf_mesh=surf_mesh,
                        inner_mesh=meshes[f"white_{hemi}"],
                        radius=self.radius,
                        interpolation=self.interpolation,
                        kind=self.kind,
                        n_samples=self.n_samples,
                        mask_img=self.mask_img,
                        depth=self.depth,
                    )
                )
            return np.vstack(hemis)

        elif len(rec.shape) == 2:
            # 2-D surface data → downsample to target mesh resolution
            n_vertices = rec.shape[0] // 2
            if n_vertices not in list(FSAVERAGE_SIZES.values()) or rec.shape[0] % 2:
                msg = f"The detected number of vertices ({rec.shape[0]}) is not in {list(FSAVERAGE_SIZES.values())}"
                raise ValueError(msg)
            n_vertices_resampled = FSAVERAGE_SIZES.get(self.mesh)
            data = rec.get_fdata()
            if n_vertices < n_vertices_resampled:
                raise NotImplementedError(
                    f"Cannot upsample from {n_vertices} vertices to {n_vertices_resampled} vertices"
                )
            if n_vertices > n_vertices_resampled:
                left = data[:n_vertices_resampled, :]
                right = data[n_vertices : n_vertices + n_vertices_resampled, :]
                data = np.concatenate([left, right], axis=0)
            return data
        else:
            raise ValueError(
                f"Unexpected shape {rec.shape} (should have 2 or 4 dimensions)"
            )
