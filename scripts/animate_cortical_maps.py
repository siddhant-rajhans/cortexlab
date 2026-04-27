"""Render rotating-brain GIFs of cortical surface maps.

Companion to ``scripts/plot_cortical_maps.py``: loads the same
group-mean per-vertex statistics from a lesion-study output directory,
but renders 36 frames around the vertical axis and combines them into
an animated GIF. Animations are the natural visualization for a 3D
inflated cortex on a 2D slide; static views always lose information
to occlusion.

Output format::

    results_dir/anim_<statmap>_<hemi>.gif

Usage
-----

    python scripts/animate_cortical_maps.py \\
        --results-dir $CORTEXLAB_RESULTS/lesion/final_YYYYMMDD_HHMMSS \\
        --statmap dr2_vision_q05 \\
        --hemi both

Dependencies: ``nilearn`` and ``Pillow``. ``Pillow`` is the standard
Python imaging library; nearly every Python install has it via
matplotlib's transitive dependency.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


MESH_VERTS_PER_HEMI = {
    "fsaverage": 163842,
    "fsaverage7": 163842,
    "fsaverage6": 40962,
    "fsaverage5": 10242,
    "fsaverage4": 2562,
    "fsaverage3": 642,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Lesion-study output directory.")
    ap.add_argument("--statmap", type=str, default="dr2_vision_q05",
                    choices=["full_r2",
                             "dr2_vision", "dr2_vision_q05",
                             "dr2_text", "dr2_text_q05"],
                    help="Which group-mean array to animate. The _q05 "
                         "variants are masked to BH-FDR q < 0.05.")
    ap.add_argument("--hemi", type=str, default="both",
                    choices=["left", "right", "both"],
                    help="Hemisphere to spin. 'both' produces a side-by-side "
                         "panel (LH + RH) per frame.")
    ap.add_argument("--mesh", type=str, default="fsaverage5",
                    choices=list(MESH_VERTS_PER_HEMI),
                    help="fsaverage variant. fsaverage5 (default) renders "
                         "in seconds per frame; fsaverage7 is much slower.")
    ap.add_argument("--cmap", type=str, default="cold_hot")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Hide vertices below this absolute value.")
    ap.add_argument("--q-threshold", type=float, default=0.05,
                    help="q-value cutoff used for the _q05 statmaps.")
    ap.add_argument("--n-frames", type=int, default=36,
                    help="Frames around the full rotation. 36 = 10° steps.")
    ap.add_argument("--frame-duration-ms", type=int, default=80,
                    help="Per-frame display time in the GIF (ms). 80ms = "
                         "12.5 fps, smooth and readable.")
    ap.add_argument("--elevation", type=float, default=0.0,
                    help="Camera elevation in degrees (0 = horizontal).")
    ap.add_argument("--dpi", type=int, default=120)
    return ap.parse_args()


def _group_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0).astype(np.float32)
    return np.nanmean(stack, axis=0)


def _resolve_statmap(results_dir: Path, statmap: str,
                     subject_ids: list[int],
                     modalities: list[str],
                     q_threshold: float) -> np.ndarray:
    """Load + group-mean the requested statmap from per-subject npz files.

    For ``_q05`` variants, computes per-subject q-values via
    :func:`cortexlab.analysis.stats.bh_fdr` over each subject's voxel
    p-values, group-means the q's, and masks the corresponding ΔR² array.
    """
    full_r2: list[np.ndarray] = []
    dR2: dict[str, list[np.ndarray]] = {m: [] for m in modalities}
    p_vals: dict[str, list[np.ndarray]] = {m: [] for m in modalities}

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
                p_vals[m].append(npz[f"p_{m}"])

    if statmap == "full_r2":
        return _group_mean(full_r2)

    name, *suffix = statmap.split("_")  # e.g. "dr2", ["vision", "q05"]
    modality = suffix[0]
    is_masked = suffix[-1] == "q05" if len(suffix) > 1 else False

    delta = _group_mean(dR2[modality])
    if not is_masked:
        return delta

    if not p_vals[modality]:
        raise SystemExit(
            f"statmap {statmap} requires p-value arrays in the npz files; "
            "rerun the orchestrator with --permutations N to populate them."
        )
    from cortexlab.analysis.stats import bh_fdr  # lazy
    q_per_subject = [bh_fdr(p) for p in p_vals[modality]]
    q_group = _group_mean(q_per_subject)
    masked = delta.copy()
    masked[q_group >= q_threshold] = np.nan
    return masked


def _truncate_to_mesh(stat_map: np.ndarray, mesh: str) -> np.ndarray:
    """fsaverage hierarchical-subset truncation when data is higher-res."""
    n_vert_per_hemi = MESH_VERTS_PER_HEMI[mesh]
    n_data_per_hemi = stat_map.shape[0] // 2
    if n_data_per_hemi == n_vert_per_hemi:
        return stat_map
    if n_vert_per_hemi > n_data_per_hemi:
        raise ValueError(
            f"data has {n_data_per_hemi} verts/hemi; cannot upsample to "
            f"{mesh}'s {n_vert_per_hemi}."
        )
    lh = stat_map[:n_vert_per_hemi]
    rh = stat_map[n_data_per_hemi : n_data_per_hemi + n_vert_per_hemi]
    return np.concatenate([lh, rh])


def _render_frame(stat_map_truncated: np.ndarray, fs, hemi: str,
                  elev: float, azim: float,
                  cmap: str, threshold: float | None,
                  vmin: float, vmax: float, dpi: int) -> bytes:
    """Render one frame and return PNG bytes (in-memory)."""
    from nilearn.plotting import plot_surf_stat_map  # lazy

    n_vert_per_hemi = stat_map_truncated.shape[0] // 2
    lh_data = stat_map_truncated[:n_vert_per_hemi]
    rh_data = stat_map_truncated[n_vert_per_hemi:]

    if hemi == "both":
        fig, axarr = plt.subplots(
            1, 2, figsize=(10, 5),
            subplot_kw={"projection": "3d"},
            gridspec_kw={"wspace": 0, "hspace": 0},
        )
        plot_surf_stat_map(
            fs.infl_left, lh_data, bg_map=fs.sulc_left,
            hemi="left", view=(elev, azim),
            cmap=cmap, vmax=vmax, vmin=vmin, threshold=threshold,
            colorbar=False, axes=axarr[0],
        )
        plot_surf_stat_map(
            fs.infl_right, rh_data, bg_map=fs.sulc_right,
            hemi="right", view=(elev, azim),
            cmap=cmap, vmax=vmax, vmin=vmin, threshold=threshold,
            colorbar=False, axes=axarr[1],
        )
    else:
        fig, ax = plt.subplots(
            figsize=(6, 6), subplot_kw={"projection": "3d"},
        )
        data = lh_data if hemi == "left" else rh_data
        mesh_geom = fs.infl_left if hemi == "left" else fs.infl_right
        sulc = fs.sulc_left if hemi == "left" else fs.sulc_right
        plot_surf_stat_map(
            mesh_geom, data, bg_map=sulc,
            hemi=hemi, view=(elev, azim),
            cmap=cmap, vmax=vmax, vmin=vmin, threshold=threshold,
            colorbar=False, axes=ax,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    from PIL import Image  # lazy
    from nilearn.datasets import fetch_surf_fsaverage  # lazy

    results_dir = args.results_dir
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}; wrong --results-dir?")
    manifest = json.loads(manifest_path.read_text())
    modalities = manifest["results"][0]["modality_order"]

    logger.info("loading %s across %d subjects", args.statmap, len(manifest["subject_ids"]))
    stat_map = _resolve_statmap(
        results_dir, args.statmap, manifest["subject_ids"],
        modalities, args.q_threshold,
    )
    stat_map = _truncate_to_mesh(stat_map, args.mesh)

    finite = np.isfinite(stat_map)
    if not finite.any():
        raise SystemExit(
            f"statmap {args.statmap} has no finite values; nothing to animate."
        )
    vmax = float(np.nanmax(np.abs(stat_map)))
    vmin = -vmax if vmax > 0 else 0.0

    logger.info("rendering %d frames at mesh=%s, hemi=%s",
                args.n_frames, args.mesh, args.hemi)
    fs = fetch_surf_fsaverage(mesh=args.mesh)

    azimuths = np.linspace(0, 360, args.n_frames, endpoint=False)
    frames: list[Image.Image] = []
    for i, azim in enumerate(azimuths):
        png_bytes = _render_frame(
            stat_map, fs, args.hemi, args.elevation, float(azim),
            args.cmap, args.threshold, vmin, vmax, args.dpi,
        )
        frames.append(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        if (i + 1) % max(1, args.n_frames // 6) == 0:
            logger.info("rendered frame %d/%d (azim=%.0f°)",
                        i + 1, args.n_frames, azim)

    out_path = results_dir / f"anim_{args.statmap}_{args.hemi}.gif"
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.frame_duration_ms,
        loop=0,
        optimize=True,
    )
    size_kb = out_path.stat().st_size / 1024
    logger.info("wrote %s (%d frames, %.0f KB)", out_path, len(frames), size_kb)


if __name__ == "__main__":
    main()
