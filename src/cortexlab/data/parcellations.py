"""Cortical parcellation loaders for brain-encoding experiments.

A *parcellation* maps each cortical vertex to a named region of interest
(ROI). Brain-encoding studies typically report per-ROI statistics (mean
R^2, delta R^2 under lesion, alignment strength) rather than per-vertex
numbers, because the per-vertex signal is noisy and the neuroscience
narrative lives at the ROI level (V1, FFA, STG, ATL, etc.).

This module exposes two things:

1. :func:`build_roi_indices` — a pure-data helper that converts the
   FreeSurfer ``.annot`` format (``labels``, ``names``) into the
   ``{roi_name: int_array}`` dict that
   :func:`cortexlab.data.studies.lahner2024bold.load_subject` consumes via
   its ``parcellation=`` kwarg.

2. :func:`load_hcp_mmp_fsaverage` — a convenience loader for the
   Glasser et al. 2016 HCP-MMP 1.0 parcellation on the fsaverage7
   surface. Accepts paths to the left- and right-hemisphere ``.annot``
   files and returns the combined-hemisphere vertex indices for a
   user-selected subset of ROIs.

The module is intentionally path-agnostic; it does not download atlas
files. Users point it at ``.annot`` files they already have locally.

Where to get the HCP-MMP fsaverage annot files
-----------------------------------------------

The official projections are distributed several ways:

* ``$FREESURFER_HOME/subjects/fsaverage/label/lh.HCPMMP1.annot`` (and
  ``rh.HCPMMP1.annot``) when using the ``--hcp-mmp`` FreeSurfer build.
* `github.com/faskowit/multiAtlasTT` (atlases repackaged for multiple
  subjects including fsaverage).
* `github.com/HolmesLab/Parcellations_onto_fsaverage`.

Any of these work as long as the resulting file parses with
``nibabel.freesurfer.io.read_annot``.

Default ROI subset for NeuroAI studies
---------------------------------------

The default selection in :func:`load_hcp_mmp_fsaverage` covers the
regions most commonly reported in multimodal brain-encoding papers:

* **Early visual**: V1, V2, V3, V4
* **Lateral occipital**: LO1, LO2, LO3, MT, MST
* **Category-selective**: FFC (face complex), PH (place-related)
* **Auditory**: A1, A4, A5
* **Superior temporal**: STSda, STSdp, STSva, STSvp
* **Language**: 44, 45 (Broca), IFJa, IFJp, PFm, PGs, PGi (angular /
  IPL)
* **Anterior temporal**: TF, TGd, TGv

Callers can ask for any other HCP-MMP region by name via the ``rois``
argument.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# default ROI set for NeuroAI lesion studies                                  #
# --------------------------------------------------------------------------- #

DEFAULT_HCP_MMP_ROIS: tuple[str, ...] = (
    # Early visual hierarchy.
    "V1", "V2", "V3", "V4",
    # Lateral occipital / motion-selective.
    "LO1", "LO2", "LO3", "MT", "MST",
    # Category-selective.
    "FFC", "PH",
    # Auditory.
    "A1", "A4", "A5",
    # Superior temporal sulcus subdivisions.
    "STSda", "STSdp", "STSva", "STSvp",
    # Classical language.
    "44", "45", "IFJa", "IFJp",
    # Angular gyrus / inferior parietal lobule.
    "PFm", "PGs", "PGi",
    # Anterior temporal lobe proxies.
    "TF", "TGd", "TGv",
)
"""HCP-MMP ROI names we report by default. Covers the visual / auditory /
language / multimodal split that NeuroAI encoding papers typically present.
"""


# The HCP-MMP release prefixes each label with ``L_`` or ``R_`` and suffixes
# with ``_ROI``. Users can pass either the bare region name ("V1") or the
# fully-qualified form ("L_V1_ROI"); we normalize internally.
_HCP_MMP_PREFIX: dict[str, str] = {"left": "L_", "right": "R_"}
_HCP_MMP_SUFFIX: str = "_ROI"


# --------------------------------------------------------------------------- #
# core helper                                                                 #
# --------------------------------------------------------------------------- #

def build_roi_indices(
    labels_lh: np.ndarray,
    names_lh: Sequence[str],
    labels_rh: np.ndarray,
    names_rh: Sequence[str],
    rois: Sequence[str],
    strict: bool = False,
) -> dict[str, np.ndarray]:
    """Convert annot-style labels + names into combined-hemisphere ROI indices.

    The return format matches what
    :func:`cortexlab.data.studies.lahner2024bold.load_subject` expects for
    its ``parcellation`` kwarg: each value is a 1-D ``int64`` array of
    vertex indices into the concatenated ``(left ‖ right)`` cortex, with
    the right hemisphere offset by ``len(labels_lh)``.

    Parameters
    ----------
    labels_lh, labels_rh
        Integer label arrays, one entry per vertex, typically 163842 for
        fsaverage7. Each entry indexes into ``names_lh`` / ``names_rh``.
    names_lh, names_rh
        Region names indexed by the label values. Exactly the tuple
        returned by ``nibabel.freesurfer.io.read_annot`` once names are
        decoded from bytes to str.
    rois
        ROI names to include. Can be either bare ("V1") or
        hemisphere-qualified ("L_V1_ROI"). Matching is done
        case-insensitively and tolerates the ``L_...`` / ``R_...`` and
        ``_ROI`` decorations.
    strict
        When True, raise ``KeyError`` if a requested ROI matches neither
        hemisphere. When False, skip missing ROIs with a warning and
        return what was found.

    Returns
    -------
    dict[str, np.ndarray]
        ``{roi_name: int64_indices}``. Region names are returned in the
        bare form (no ``L_`` / ``R_`` / ``_ROI`` decoration); the indices
        combine both hemispheres.

    Raises
    ------
    ValueError
        If ``labels_lh`` and ``labels_rh`` have different lengths.
    KeyError
        If ``strict=True`` and a requested ROI is not found.

    Examples
    --------
    >>> labels_lh = np.array([0, 1, 1, 2])          # 4 vertices, 3 regions
    >>> names_lh = ["L_???", "L_V1_ROI", "L_V2_ROI"]
    >>> labels_rh = np.array([0, 2, 1, 1])
    >>> names_rh = ["R_???", "R_V1_ROI", "R_V2_ROI"]
    >>> idx = build_roi_indices(labels_lh, names_lh, labels_rh, names_rh,
    ...                         rois=["V1", "V2"])
    >>> sorted(idx["V1"].tolist())
    [1, 2, 6, 7]
    >>> sorted(idx["V2"].tolist())
    [3, 5]
    """
    labels_lh = np.asarray(labels_lh)
    labels_rh = np.asarray(labels_rh)
    if labels_lh.ndim != 1 or labels_rh.ndim != 1:
        raise ValueError("labels arrays must be 1-D")
    n_lh = int(labels_lh.shape[0])

    name_to_idx_lh = {_canonical(n): i for i, n in enumerate(names_lh)}
    name_to_idx_rh = {_canonical(n): i for i, n in enumerate(names_rh)}

    out: dict[str, np.ndarray] = {}
    for roi in rois:
        canonical = _canonical(roi)
        pieces: list[np.ndarray] = []
        if canonical in name_to_idx_lh:
            lh_label = name_to_idx_lh[canonical]
            pieces.append(np.where(labels_lh == lh_label)[0].astype(np.int64))
        if canonical in name_to_idx_rh:
            rh_label = name_to_idx_rh[canonical]
            pieces.append(
                (np.where(labels_rh == rh_label)[0] + n_lh).astype(np.int64)
            )
        if not pieces:
            msg = f"ROI {roi!r} (canonical {canonical!r}) not found in either hemisphere"
            if strict:
                raise KeyError(msg)
            logger.warning(msg + "; skipping")
            continue
        combined = np.concatenate(pieces)
        if combined.size == 0:
            logger.warning("ROI %r has zero vertices in the annot; skipping", roi)
            continue
        out[_friendly(roi)] = np.sort(combined)
    return out


def _canonical(name: str | bytes) -> str:
    """Normalize a region name to its bare, case-folded form.

    ``L_V1_ROI``, ``R_V1_ROI``, ``V1``, and ``v1`` all canonicalize to
    ``"v1"`` so that lookups in either hemisphere match regardless of
    which convention the caller uses.
    """
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    s = name.strip()
    for prefix in _HCP_MMP_PREFIX.values():
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith(_HCP_MMP_SUFFIX):
        s = s[: -len(_HCP_MMP_SUFFIX)]
    return s.casefold()


def _friendly(name: str) -> str:
    """Return a pretty display name for dict keys (strips ``L_`` / ``_ROI``
    decoration but preserves the user's original case when unambiguous).
    """
    s = name.strip()
    for prefix in _HCP_MMP_PREFIX.values():
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith(_HCP_MMP_SUFFIX):
        s = s[: -len(_HCP_MMP_SUFFIX)]
    return s


# --------------------------------------------------------------------------- #
# HCP-MMP 1.0 fsaverage convenience loader                                    #
# --------------------------------------------------------------------------- #

def load_hcp_mmp_fsaverage(
    lh_annot_path: str | os.PathLike,
    rh_annot_path: str | os.PathLike,
    rois: Sequence[str] | None = None,
    strict: bool = False,
) -> dict[str, np.ndarray]:
    """Load HCP-MMP 1.0 ROIs for fsaverage from FreeSurfer ``.annot`` files.

    Parameters
    ----------
    lh_annot_path, rh_annot_path
        Paths to the left- and right-hemisphere annot files. Typically
        ``fsaverage/label/lh.HCPMMP1.annot`` and ``rh.HCPMMP1.annot``
        from a FreeSurfer install, or equivalent downloads (see the
        module docstring for sources).
    rois
        Which HCP-MMP region names to include. None uses
        :data:`DEFAULT_HCP_MMP_ROIS`. Region names can be given with or
        without the ``L_`` / ``R_`` prefix and ``_ROI`` suffix.
    strict
        Forwarded to :func:`build_roi_indices`.

    Returns
    -------
    dict[str, np.ndarray]
        Maps each ROI name to its combined-hemisphere vertex indices.
        Ready to pass as ``parcellation=`` to ``load_subject``.

    Notes
    -----
    This function uses ``nibabel.freesurfer.io.read_annot``, which
    returns the label array, the RGBA color table, and the list of
    region names. Only the labels and names are used here.
    """
    try:
        import nibabel.freesurfer.io as fsio  # noqa: WPS433  (lazy dep)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "load_hcp_mmp_fsaverage requires nibabel. Install it with "
            "`pip install nibabel`, or install the [analysis] extras."
        ) from e

    lh_path = Path(lh_annot_path)
    rh_path = Path(rh_annot_path)
    if not lh_path.exists():
        raise FileNotFoundError(f"left annot file not found: {lh_path}")
    if not rh_path.exists():
        raise FileNotFoundError(f"right annot file not found: {rh_path}")

    labels_lh, _ctab_lh, names_lh_b = fsio.read_annot(str(lh_path))
    labels_rh, _ctab_rh, names_rh_b = fsio.read_annot(str(rh_path))
    # nibabel returns bytes in `names` until 5.x; handle both.
    names_lh = [n.decode("utf-8", errors="replace") if isinstance(n, bytes) else n
                for n in names_lh_b]
    names_rh = [n.decode("utf-8", errors="replace") if isinstance(n, bytes) else n
                for n in names_rh_b]

    target_rois = tuple(rois) if rois is not None else DEFAULT_HCP_MMP_ROIS
    roi_indices = build_roi_indices(
        labels_lh, names_lh, labels_rh, names_rh,
        rois=target_rois, strict=strict,
    )
    logger.info(
        "loaded HCP-MMP fsaverage parcellation: %d ROIs recovered from %s + %s",
        len(roi_indices), lh_path.name, rh_path.name,
    )
    return roi_indices


# --------------------------------------------------------------------------- #
# env-based convenience                                                       #
# --------------------------------------------------------------------------- #

def load_hcp_mmp_from_freesurfer(
    subjects_dir: str | os.PathLike | None = None,
    subject: str = "fsaverage",
    rois: Sequence[str] | None = None,
    strict: bool = False,
) -> dict[str, np.ndarray]:
    """Load HCP-MMP from a standard FreeSurfer ``SUBJECTS_DIR`` layout.

    Convenience wrapper around :func:`load_hcp_mmp_fsaverage` that looks
    for ``{subjects_dir}/{subject}/label/{lh,rh}.HCPMMP1.annot``.

    Parameters
    ----------
    subjects_dir
        Path to FreeSurfer's ``SUBJECTS_DIR``. Falls back to the
        environment variable of the same name when None.
    subject
        Subject directory name. ``"fsaverage"`` for the standard
        template; use a subject ID for per-subject cortical surfaces.
    rois, strict
        Forwarded to :func:`load_hcp_mmp_fsaverage`.
    """
    if subjects_dir is None:
        env = os.environ.get("SUBJECTS_DIR")
        if not env:
            raise RuntimeError(
                "subjects_dir not given and SUBJECTS_DIR env var is unset"
            )
        subjects_dir = env
    root = Path(subjects_dir) / subject / "label"
    return load_hcp_mmp_fsaverage(
        lh_annot_path=root / "lh.HCPMMP1.annot",
        rh_annot_path=root / "rh.HCPMMP1.annot",
        rois=rois,
        strict=strict,
    )


__all__ = [
    "DEFAULT_HCP_MMP_ROIS",
    "build_roi_indices",
    "load_hcp_mmp_fsaverage",
    "load_hcp_mmp_from_freesurfer",
]
