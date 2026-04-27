"""Render cortical surface maps from a lesion-study output directory.

Given a manifest.json + per-subject ``subject_XX_lesion.npz`` produced by
:mod:`experiments.causal_modality_ablation`, this script computes the
group-mean per-vertex statistics (full R², ΔR² per modality, q-values
per modality after BH-FDR if available) and renders four-panel
fsaverage surface figures (left lateral, left medial, right medial,
right lateral) using ``nilearn.plotting.plot_surf_stat_map``.

Outputs land alongside the manifest as PNGs:

* ``surf_full_r2.png``               — group-mean encoder R² across cortex
* ``surf_dr2_<modality>.png``        — group-mean lesion ΔR² per modality
* ``surf_dr2_<modality>_q05.png``    — same, masked to voxels with
                                       BH-FDR q < 0.05 (when q-values present)

Usage
-----

    python scripts/plot_cortical_maps.py \\
        --results-dir $CORTEXLAB_RESULTS/lesion/final_YYYYMMDD_HHMMSS \\
        --modalities vision,text

Dependencies: ``nilearn`` (lazy import; not in cortexlab core deps because
the rest of the pipeline doesn't need it).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


VIEWS = [
    ("lh", "lateral"),
    ("lh", "medial"),
    ("rh", "medial"),
    ("rh", "lateral"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Directory with manifest.json + subject_XX_lesion.npz")
    ap.add_argument("--modalities", type=str, default=None,
                    help="Comma-separated modalities to plot. Defaults to "
                         "manifest.results[0].modality_order.")
    ap.add_argument("--mesh", type=str, default="fsaverage5",
                    choices=["fsaverage", "fsaverage5", "fsaverage6"],
                    help="fsaverage variant for plotting. 'fsaverage' = the "
                         "163,842-vertex high-res mesh (slow on a login "
                         "node, allow ~10 min per figure). 'fsaverage5' "
                         "(default, 10,242 verts/hemi) renders in seconds "
                         "and is sufficient for slide-resolution maps. "
                         "Data on fsaverage7 is auto-truncated to lower-res "
                         "meshes via the standard fsaverage-subset trick.")
    ap.add_argument("--cmap", type=str, default="cold_hot",
                    help="matplotlib/nilearn colormap.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Hide vertices below this absolute value. Useful "
                         "to declutter the unsignificant background.")
    ap.add_argument("--q-threshold", type=float, default=0.05,
                    help="q-value cutoff for the masked variants.")
    ap.add_argument("--dpi", type=int, default=160)
    return ap.parse_args()


def _group_mean(arrays: list[np.ndarray]) -> np.ndarray:
    """Mean across subjects, treating NaN as missing."""
    stack = np.stack(arrays, axis=0).astype(np.float32)
    return np.nanmean(stack, axis=0)


def _load_subject_arrays(results_dir: Path, subject_ids: list[int],
                         modalities: list[str]) -> dict[str, np.ndarray]:
    """Load per-subject arrays and return group-mean dict.

    Keys returned: ``full_r2``, ``dR2_<m>`` for each modality, plus
    ``q_<m>`` when q-value arrays were saved (require ``--fdr`` at run time).
    """
    full_r2: list[np.ndarray] = []
    dR2: dict[str, list[np.ndarray]] = {m: [] for m in modalities}
    q_vals: dict[str, list[np.ndarray]] = {m: [] for m in modalities}

    for sid in subject_ids:
        npz_path = results_dir / f"subject_{sid:02d}_lesion.npz"
        if not npz_path.exists():
            logger.warning("missing %s; skipping", npz_path)
            continue
        npz = np.load(npz_path)
        full_r2.append(npz["full_r2"])
        for m in modalities:
            dR2[m].append(npz[f"delta_{m}"])
            if f"p_{m}" in npz.files:
                # Per-subject q-values aren't stored; we can derive them
                # from p-values at plot-time. Defer that branch until we
                # actually need it (the orchestrator now stores p_<m>
                # in the npz; q-values can be recomputed cheaply).
                from cortexlab.analysis.stats import bh_fdr  # lazy
                q_vals[m].append(bh_fdr(npz[f"p_{m}"]))

    out: dict[str, np.ndarray] = {"full_r2": _group_mean(full_r2)}
    for m in modalities:
        out[f"dR2_{m}"] = _group_mean(dR2[m])
        if q_vals[m]:
            out[f"q_{m}"] = _group_mean(q_vals[m])
    return out


def _plot_panel(stat_map: np.ndarray, title: str, out_path: Path,
                cmap: str, threshold: float | None,
                mesh: str, dpi: int) -> None:
    """Four-panel cortical figure: lh-lat, lh-med, rh-med, rh-lat.

    fsaverage7 (163,842 verts/hemi) is too slow to render with
    matplotlib's 3D backend on a login node (each save takes minutes).
    Lower-resolution fsaverage variants (5/6) are topologically a
    subset of fsaverage7 — vertex K of fsaverage5 has the same
    coordinate as vertex K of fsaverage7. So when the caller asks for
    a lower-res mesh we just truncate the data to the first N verts
    per hemisphere; no interpolation needed.
    """
    from nilearn.datasets import fetch_surf_fsaverage  # lazy
    from nilearn.plotting import plot_surf_stat_map  # lazy

    fs = fetch_surf_fsaverage(mesh=mesh)
    # nilearn returns surfaces as filenames (str) or InMemoryMesh; load
    # one to get the canonical vertex count for the requested mesh.
    import nibabel.freesurfer.io as fsio
    mesh_lh_verts, _ = fsio.read_geometry(fs.infl_left)
    n_mesh_verts_per_hemi = mesh_lh_verts.shape[0]
    n_data_per_hemi = stat_map.shape[0] // 2
    if n_data_per_hemi != n_mesh_verts_per_hemi:
        if n_mesh_verts_per_hemi > n_data_per_hemi:
            raise ValueError(
                f"data has {n_data_per_hemi} verts/hemi but {mesh!r} expects "
                f"{n_mesh_verts_per_hemi}; pick a lower-res mesh, not higher."
            )
        # Truncate: first N verts of fsaverage7 == fsaverage5/6 verts.
        logger.info(
            "downsampling %d -> %d verts/hemi via fsaverage subset truncation",
            n_data_per_hemi, n_mesh_verts_per_hemi,
        )
        lh_data = stat_map[:n_mesh_verts_per_hemi]
        rh_data = stat_map[n_data_per_hemi : n_data_per_hemi + n_mesh_verts_per_hemi]
    else:
        lh_data = stat_map[:n_data_per_hemi]
        rh_data = stat_map[n_data_per_hemi:]

    fig, axarr = plt.subplots(
        1, 4, figsize=(16, 4),
        subplot_kw={"projection": "3d"},
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    finite = np.isfinite(stat_map)
    if not finite.any():
        logger.warning("no finite values in stat map for %s; skipping plot", title)
        plt.close(fig)
        return
    vmax = float(np.nanmax(np.abs(stat_map)))
    vmin = -vmax if vmax > 0 else 0.0

    panels = [
        (axarr[0], lh_data, fs.infl_left, fs.sulc_left, "left",  "lateral", "L lateral"),
        (axarr[1], lh_data, fs.infl_left, fs.sulc_left, "left",  "medial",  "L medial"),
        (axarr[2], rh_data, fs.infl_right, fs.sulc_right, "right", "medial",  "R medial"),
        (axarr[3], rh_data, fs.infl_right, fs.sulc_right, "right", "lateral", "R lateral"),
    ]
    for ax, data, mesh_geom, sulc, hemi, view, label in panels:
        plot_surf_stat_map(
            mesh_geom, data,
            bg_map=sulc, hemi=hemi, view=view,
            cmap=cmap, vmax=vmax, vmin=vmin,
            threshold=threshold,
            colorbar=False, axes=ax,
        )
        ax.set_title(label, fontsize=10)

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    results_dir = args.results_dir
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}; wrong --results-dir?")
    manifest = json.loads(manifest_path.read_text())
    subject_ids = manifest["subject_ids"]
    modalities = (
        [m.strip() for m in args.modalities.split(",")]
        if args.modalities
        else manifest["results"][0]["modality_order"]
    )
    logger.info("plotting %d modalities across %d subjects: %s",
                len(modalities), len(subject_ids), modalities)

    arrays = _load_subject_arrays(results_dir, subject_ids, modalities)

    # Full R² map.
    _plot_panel(
        arrays["full_r2"],
        title=f"Group-mean full $R^2$ (n={len(subject_ids)})",
        out_path=results_dir / "surf_full_r2.png",
        cmap=args.cmap, threshold=args.threshold,
        mesh=args.mesh, dpi=args.dpi,
    )

    for m in modalities:
        # ΔR² map.
        _plot_panel(
            arrays[f"dR2_{m}"],
            title=f"Group-mean $\\Delta R^2$ when lesioning {m} (n={len(subject_ids)})",
            out_path=results_dir / f"surf_dr2_{m}.png",
            cmap=args.cmap, threshold=args.threshold,
            mesh=args.mesh, dpi=args.dpi,
        )
        # Q-masked variant.
        if f"q_{m}" in arrays:
            masked = arrays[f"dR2_{m}"].copy()
            masked[arrays[f"q_{m}"] >= args.q_threshold] = np.nan
            _plot_panel(
                masked,
                title=(
                    f"$\\Delta R^2$ for {m}, masked to q < {args.q_threshold} "
                    f"(BH-FDR, n={len(subject_ids)})"
                ),
                out_path=results_dir / f"surf_dr2_{m}_q{int(args.q_threshold*100):02d}.png",
                cmap=args.cmap, threshold=args.threshold,
                mesh=args.mesh, dpi=args.dpi,
            )


if __name__ == "__main__":
    main()
