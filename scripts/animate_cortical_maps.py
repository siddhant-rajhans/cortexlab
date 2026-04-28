"""Render rotating-brain GIFs/MP4s of cortical surface maps.

Companion to :mod:`scripts.plot_cortical_maps`. Loads the same
group-mean per-vertex statistics from a lesion-study output directory,
renders N frames around the vertical axis, and combines them into an
animated GIF (or MP4 with ``--format mp4``). Animations are the natural
visualization for 3D inflated cortex on a 2D slide; static views always
lose information to occlusion at the medial wall.

Two rendering engines:

* ``--engine plotly`` (default when ``[viz]`` extras installed): WebGL,
  GPU-accelerated, ~30s for a 36-frame fsaverage7 GIF.
* ``--engine matplotlib``: pure CPU, ~12 min per 36-frame GIF on
  fsaverage5; impractical at fsaverage7.

Usage
-----

::

    python scripts/animate_cortical_maps.py \\
        --results-dir $CORTEXLAB_RESULTS/lesion/final_YYYYMMDD_HHMMSS \\
        --statmap dr2_vision_q05 \\
        --hemi both
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--statmap", type=str, default="dr2_vision_q05",
                    choices=["full_r2",
                             "dr2_vision", "dr2_vision_q05",
                             "dr2_text", "dr2_text_q05"])
    ap.add_argument("--hemi", type=str, default="both",
                    choices=["left", "right", "both"])
    ap.add_argument("--engine", type=str, default="auto",
                    choices=["auto", "matplotlib", "plotly", "pyvista"])
    ap.add_argument("--mesh", type=str, default="fsaverage5",
                    choices=["fsaverage", "fsaverage5", "fsaverage6",
                             "fsaverage7"])
    ap.add_argument("--surface", type=str, default="inflated",
                    choices=["inflated", "pial", "white"],
                    help="Cortical surface family. 'pial' shows the real "
                         "brain shape; 'inflated' is a smooth balloon.")
    ap.add_argument("--cmap", type=str, default="cold_hot")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--q-threshold", type=float, default=0.05)
    ap.add_argument("--n-frames", type=int, default=36,
                    help="36 = 10° steps for a 360° rotation.")
    ap.add_argument("--frame-duration-ms", type=int, default=80,
                    help="Per-frame display time. 80ms = 12.5 fps.")
    ap.add_argument("--elevation", type=float, default=0.0)
    ap.add_argument("--dpi", type=int, default=120,
                    help="matplotlib only.")
    ap.add_argument("--width", type=int, default=1200,
                    help="plotly only.")
    ap.add_argument("--height", type=int, default=600,
                    help="plotly only.")
    ap.add_argument("--format", type=str, default="gif",
                    choices=["gif", "mp4"],
                    help="Output container. mp4 needs imageio-ffmpeg.")
    return ap.parse_args()


def _group_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0).astype(np.float32)
    return np.nanmean(stack, axis=0)


def _resolve_statmap(results_dir: Path, statmap: str,
                     subject_ids: list[int],
                     modalities: list[str],
                     q_threshold: float) -> np.ndarray:
    """Load + group-mean the requested statmap from per-subject npz files."""
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

    name, *suffix = statmap.split("_")  # "dr2", ["vision", "q05"]
    modality = suffix[0]
    is_masked = len(suffix) > 1 and suffix[-1] == "q05"

    delta = _group_mean(dR2[modality])
    if not is_masked:
        return delta

    if not p_vals[modality]:
        raise SystemExit(
            f"statmap {statmap} requires per-subject p_{modality} arrays "
            "in the npz; rerun the orchestrator with --permutations N."
        )
    from cortexlab.analysis.stats import bh_fdr  # lazy
    q_per_subject = [bh_fdr(p) for p in p_vals[modality]]
    q_group = _group_mean(q_per_subject)
    masked = delta.copy()
    masked[q_group >= q_threshold] = np.nan
    return masked


def _write_gif(frames_png: list[bytes], out_path: Path, duration_ms: int) -> None:
    from PIL import Image  # lazy
    pil_frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in frames_png]
    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )


def _write_mp4(frames_png: list[bytes], out_path: Path, duration_ms: int) -> None:
    """MP4 export via imageio-ffmpeg. Smaller than GIF, plays in PowerPoint."""
    try:
        import imageio.v3 as iio
    except ImportError as e:
        raise SystemExit(
            "MP4 output requires imageio. Install with `pip install imageio imageio-ffmpeg`."
        ) from e
    from PIL import Image  # lazy
    fps = 1000.0 / duration_ms
    pil_frames = [
        np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
        for b in frames_png
    ]
    iio.imwrite(out_path, np.stack(pil_frames), fps=fps, codec="libx264")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    results_dir = args.results_dir
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}; wrong --results-dir?")
    manifest = json.loads(manifest_path.read_text())
    modalities = manifest["results"][0]["modality_order"]

    logger.info("loading %s across %d subjects",
                args.statmap, len(manifest["subject_ids"]))
    stat_map = _resolve_statmap(
        results_dir, args.statmap, manifest["subject_ids"],
        modalities, args.q_threshold,
    )

    finite = np.isfinite(stat_map)
    if not finite.any():
        raise SystemExit(
            f"statmap {args.statmap} has no finite values; nothing to animate."
        )

    from cortexlab.viz.surface_renderer import RenderConfig, make_renderer
    renderer = make_renderer(engine=args.engine, mesh=args.mesh)
    config = RenderConfig(
        mesh=args.mesh, cmap=args.cmap,
        threshold=args.threshold, dpi=args.dpi,
        width=args.width, height=args.height,
        surface=args.surface,
    )
    logger.info("rendering %d frames with %s engine, mesh=%s, hemi=%s",
                args.n_frames, renderer.name, args.mesh, args.hemi)

    azimuths = np.linspace(0, 360, args.n_frames, endpoint=False)
    frames_png: list[bytes] = []
    for i, azim in enumerate(azimuths):
        view = (args.elevation, float(azim))
        frames_png.append(renderer.render_frame(stat_map, view, args.hemi, config))
        if (i + 1) % max(1, args.n_frames // 6) == 0:
            logger.info("rendered frame %d/%d (azim=%.0f°)",
                        i + 1, args.n_frames, azim)

    out_path = results_dir / f"anim_{args.statmap}_{args.hemi}.{args.format}"
    if args.format == "gif":
        _write_gif(frames_png, out_path, args.frame_duration_ms)
    else:
        _write_mp4(frames_png, out_path, args.frame_duration_ms)
    size_kb = out_path.stat().st_size / 1024
    logger.info("wrote %s (%d frames, %.0f KB)",
                out_path, len(frames_png), size_kb)


if __name__ == "__main__":
    main()
