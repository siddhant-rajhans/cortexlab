"""Render cortical surface maps from a lesion-study output directory.

Loads group-mean per-vertex statistics from a directory of
``subject_XX_lesion.npz`` + ``manifest.json`` and renders four-panel
fsaverage surface figures (LH lateral, LH medial, RH medial, RH lateral)
into PNGs alongside the manifest.

Two rendering engines:

* ``--engine plotly`` (default when the ``[viz]`` extras are installed):
  WebGL via plotly + kaleido. GPU-accelerated, fast on dense meshes,
  publication-quality output.
* ``--engine matplotlib``: pure-CPU 3D rasterizer. Always available,
  slow on dense meshes.

Outputs land in ``--results-dir``:

* ``surf_full_r2.png``               — group-mean encoder R²
* ``surf_dr2_<modality>.png``        — group-mean lesion ΔR²
* ``surf_dr2_<modality>_q05.png``    — same, masked to BH-FDR q < 0.05

Plotly engine additionally writes interactive HTML scenes when
``--write-html`` is set.

Usage
-----

::

    python scripts/plot_cortical_maps.py \\
        --results-dir $CORTEXLAB_RESULTS/lesion/final_YYYYMMDD_HHMMSS \\
        --modalities vision,text
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


VIEW_PANELS = [
    ("L lateral", (0.0,   180.0), "left"),
    ("L medial",  (0.0,   0.0),   "left"),
    ("R medial",  (0.0,   180.0), "right"),
    ("R lateral", (0.0,   0.0),   "right"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Directory with manifest.json + subject_XX_lesion.npz")
    ap.add_argument("--modalities", type=str, default=None,
                    help="Comma-separated modalities to plot. Defaults to "
                         "manifest.results[0].modality_order.")
    ap.add_argument("--engine", type=str, default="auto",
                    choices=["auto", "matplotlib", "plotly"],
                    help="Rendering engine. 'auto' selects plotly when the "
                         "[viz] extras are installed, else matplotlib.")
    ap.add_argument("--mesh", type=str, default="fsaverage5",
                    choices=["fsaverage", "fsaverage5", "fsaverage6",
                             "fsaverage7"],
                    help="fsaverage variant. fsaverage5 (default) renders "
                         "in seconds; fsaverage7 is publication-quality "
                         "and fast under the plotly engine, slow under "
                         "matplotlib.")
    ap.add_argument("--cmap", type=str, default="cold_hot",
                    help="matplotlib/nilearn colormap.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Hide vertices below this absolute value.")
    ap.add_argument("--q-threshold", type=float, default=0.05,
                    help="q-value cutoff for the masked variants.")
    ap.add_argument("--dpi", type=int, default=160,
                    help="DPI for matplotlib engine (ignored by plotly).")
    ap.add_argument("--width", type=int, default=1600,
                    help="Pixel width for plotly engine (per panel).")
    ap.add_argument("--height", type=int, default=400,
                    help="Pixel height for plotly engine.")
    ap.add_argument("--write-html", action="store_true",
                    help="When --engine=plotly, also write standalone "
                         "interactive HTML files alongside each PNG.")
    return ap.parse_args()


def _group_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0).astype(np.float32)
    return np.nanmean(stack, axis=0)


def _load_subject_arrays(results_dir: Path, subject_ids: list[int],
                         modalities: list[str]) -> dict[str, np.ndarray]:
    """Load per-subject arrays and return group-mean dict.

    Keys returned: ``full_r2``, ``dR2_<m>`` for each modality, plus
    ``q_<m>`` when ``p_<m>`` arrays are saved (require ``--permutations``).
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
                from cortexlab.analysis.stats import bh_fdr  # lazy
                q_vals[m].append(bh_fdr(npz[f"p_{m}"]))

    out: dict[str, np.ndarray] = {"full_r2": _group_mean(full_r2)}
    for m in modalities:
        out[f"dR2_{m}"] = _group_mean(dR2[m])
        if q_vals[m]:
            out[f"q_{m}"] = _group_mean(q_vals[m])
    return out


def _plot_static(renderer, stat_map: np.ndarray, title: str,
                 out_path: Path, args, write_html: bool) -> None:
    """Four-panel static figure with the configured renderer."""
    from cortexlab.viz.surface_renderer import RenderConfig

    config = RenderConfig(
        mesh=args.mesh, cmap=args.cmap,
        threshold=args.threshold, dpi=args.dpi,
        width=args.width, height=args.height,
    )
    finite = np.isfinite(stat_map)
    if not finite.any():
        logger.warning("no finite values in %s; skipping", title)
        return

    views = [(label, ang) for label, ang, _h in VIEW_PANELS]
    hemis = [h for _label, _ang, h in VIEW_PANELS]
    png = renderer.render_static_panels(stat_map, views, hemis, config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png)
    logger.info("wrote %s", out_path)

    if write_html and renderer.name == "plotly":
        # Interactive HTML for the most informative single view (LH lateral).
        html = renderer.render_html(stat_map, (0.0, 180.0), "left", config)
        html_path = out_path.with_suffix(".html")
        html_path.write_text(html, encoding="utf-8")
        logger.info("wrote %s", html_path)


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

    from cortexlab.viz.surface_renderer import make_renderer
    renderer = make_renderer(engine=args.engine, mesh=args.mesh)
    logger.info("using %s renderer", renderer.name)

    arrays = _load_subject_arrays(results_dir, subject_ids, modalities)

    _plot_static(
        renderer, arrays["full_r2"],
        f"Group-mean full R² (n={len(subject_ids)})",
        results_dir / "surf_full_r2.png", args, args.write_html,
    )

    for m in modalities:
        _plot_static(
            renderer, arrays[f"dR2_{m}"],
            f"Group-mean ΔR² when lesioning {m}",
            results_dir / f"surf_dr2_{m}.png", args, args.write_html,
        )
        if f"q_{m}" in arrays:
            masked = arrays[f"dR2_{m}"].copy()
            masked[arrays[f"q_{m}"] >= args.q_threshold] = np.nan
            _plot_static(
                renderer, masked,
                f"ΔR² for {m}, q < {args.q_threshold} (BH-FDR)",
                results_dir / f"surf_dr2_{m}_q{int(args.q_threshold*100):02d}.png",
                args, args.write_html,
            )


if __name__ == "__main__":
    main()
