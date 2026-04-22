"""End-to-end runner for the causal modality lesion study.

This is the script that produces the numbers in the class-project
slides (Draft 3 and Final). It orchestrates:

1. Loading the BOLD Moments dataset for one or more subjects.
2. Extracting TRIBE v2 features for each modality (text, audio, video)
   on every stimulus, with caching.
3. Fitting the voxelwise ridge encoder on the training split.
4. Running the lesion protocol per modality on the test split.
5. Computing noise ceilings from inter-subject reliability.
6. Aggregating results over ROIs and saving JSON + numpy artefacts.

Permutation testing and bootstrap CIs are delegated to
``cortexlab.analysis.brain_alignment`` which already implements them.

Usage
-----

Typical full run on Stevens Jarvis::

    python -m experiments.causal_modality_ablation \
        --config experiments/config/lesion_bold_moments.yaml \
        --subjects 1 2 3 4 5 6 7 8 9 10 \
        --output experiments/results/lesion/

Quick pilot run on one subject with a subset of stimuli::

    python -m experiments.causal_modality_ablation \
        --config experiments/config/lesion_bold_moments.yaml \
        --subjects 1 --pilot 200

The script can also run in ``--mock`` mode which synthesizes tiny
deterministic data, useful for CI and for validating the pipeline
shape before kicking off a real Jarvis run.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from cortexlab.analysis.lesion import LesionResult, roi_summary, run_modality_lesion
from cortexlab.analysis.noise_ceiling import (
    inter_subject_ceiling,
    normalize_by_ceiling,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=None,
                    help="YAML config file; CLI flags override individual keys")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--pilot", type=int, default=None,
                    help="Use only N stimuli for a quick pilot run")
    ap.add_argument("--output", type=str,
                    default="experiments/results/lesion")
    ap.add_argument("--alphas", type=str,
                    default="0.01,1,100,10000,1000000")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--mask", type=str, default="zero",
                    choices=["zero", "learned"])
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", type=str, default="auto")
    ap.add_argument("--mock", action="store_true",
                    help="Run on synthetic data for smoke-testing.")
    ap.add_argument("--data-root", type=str, default=None,
                    help="BOLD Moments dataset root. Overrides yaml/env.")
    ap.add_argument("--feature-cache", type=str, default=None,
                    help="Directory containing <modality>.npz files "
                         "produced by experiments.build_feature_cache.")
    ap.add_argument("--modalities", type=str, default="vision,text",
                    help="Comma-separated modality names whose .npz files "
                         "live in --feature-cache.")
    ap.add_argument("--parcellation", type=str, default="none",
                    choices=["none", "hcp-mmp"],
                    help="Parcellation to report ROI-level results against. "
                         "'none' keeps a single 'all_cortex' bucket. "
                         "'hcp-mmp' requires --lh-annot and --rh-annot.")
    ap.add_argument("--lh-annot", type=str, default=None,
                    help="Left-hemisphere FreeSurfer .annot file for the "
                         "HCP-MMP (or compatible) parcellation.")
    ap.add_argument("--rh-annot", type=str, default=None,
                    help="Right-hemisphere FreeSurfer .annot file.")
    ap.add_argument("--parcellation-rois", type=str, default=None,
                    help="Comma-separated ROI names to include. None uses the "
                         "DEFAULT_HCP_MMP_ROIS set from cortexlab.data.parcellations.")
    return ap.parse_args()


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------- #
# data loading                                                                #
# --------------------------------------------------------------------------- #

def _resolve_parcellation(cfg: dict) -> dict[str, np.ndarray] | None:
    """Build the ``{roi_name: indices}`` dict from CLI/yaml config, or None.

    Separated from ``_load_subject_data`` so the parcellation is loaded
    exactly once and shared across subjects (the annot files are identical
    for every subject on fsaverage).
    """
    kind = cfg.get("parcellation") or "none"
    if kind == "none":
        return None
    if kind == "hcp-mmp":
        from cortexlab.data.parcellations import load_hcp_mmp_fsaverage  # lazy
        lh = cfg.get("lh_annot")
        rh = cfg.get("rh_annot")
        if not lh or not rh:
            raise ValueError(
                "parcellation=hcp-mmp requires --lh-annot and --rh-annot "
                "(or the equivalent yaml keys lh_annot / rh_annot)."
            )
        rois = cfg.get("parcellation_rois")
        return load_hcp_mmp_fsaverage(lh, rh, rois=rois)
    raise ValueError(f"unknown parcellation {kind!r}")


def _load_subject_data(
    subject_id: int, cfg: dict, pilot: int | None,
    parcellation: dict[str, np.ndarray] | None = None,
) -> dict:
    """Load features and responses for one subject.

    The heavy lifting (BOLD Moments loader) lives in
    :mod:`cortexlab.data.studies.lahner2024bold`; this wrapper exists so
    the orchestrator can return a uniform dict regardless of whether the
    data came from disk or from the mock generator.

    ``cfg`` is the merged configuration (YAML plus CLI overrides), so the
    helper does not need to know which source set each field. ``parcellation``
    is threaded through to ``load_subject`` so per-ROI aggregation is done
    downstream in the orchestrator.
    """
    from cortexlab.data.studies.lahner2024bold import load_subject  # lazy

    modalities = cfg.get("modalities") or ["vision", "text"]
    rec = load_subject(
        subject_id=subject_id,
        root=cfg.get("data_root"),
        feature_cache=cfg.get("feature_cache"),
        modalities=tuple(modalities),
        parcellation=parcellation,
        n_trimmed_stimuli=pilot,
    )
    return rec


def _mock_subject_data(subject_id: int, n_train: int = 200, n_test: int = 40,
                      p: int = 32, n_vox: int = 300) -> dict:
    """Synthetic drop-in replacement for ``_load_subject_data``.

    Each voxel block depends on exactly one modality so the lesion
    protocol has a recoverable signal. Reproducible per subject_id.
    """
    rng = np.random.default_rng(subject_id)
    n = n_train + n_test
    mods = {
        "text":  rng.standard_normal((n, p)).astype(np.float32),
        "audio": rng.standard_normal((n, p)).astype(np.float32),
        "video": rng.standard_normal((n, p)).astype(np.float32),
    }
    per = n_vox // 3
    Y = np.zeros((n, n_vox), dtype=np.float32)
    for i, m in enumerate(mods):
        W = rng.standard_normal((p, per)).astype(np.float32)
        Y[:, i * per:(i + 1) * per] = mods[m] @ W
    Y += 0.2 * rng.standard_normal(Y.shape).astype(np.float32)

    return {
        "subject_id": subject_id,
        "features_train": {m: mods[m][:n_train] for m in mods},
        "features_test":  {m: mods[m][n_train:] for m in mods},
        "y_train": Y[:n_train],
        "y_test": Y[n_train:],
        "roi_indices": {
            "text_roi":  np.arange(0, per),
            "audio_roi": np.arange(per, 2 * per),
            "video_roi": np.arange(2 * per, 3 * per),
        },
    }


def _subset(rec: dict, n: int) -> dict:
    """Truncate a subject record to the first ``n`` stimuli (preserving test)."""
    out = dict(rec)
    out["features_train"] = {m: v[:n] for m, v in rec["features_train"].items()}
    out["y_train"] = rec["y_train"][:n]
    return out


# --------------------------------------------------------------------------- #
# orchestration                                                               #
# --------------------------------------------------------------------------- #

def run_one_subject(
    rec: dict,
    alphas: list[float],
    cv: int,
    mask: str,
    device: str,
    backend: str,
) -> dict:
    """Fit encoder, run lesion, summarize over ROIs."""
    t0 = time.perf_counter()
    result = run_modality_lesion(
        rec["features_train"], rec["features_test"],
        rec["y_train"], rec["y_test"],
        alphas=alphas, cv=cv, mask_strategy=mask,
        device=device, backend=backend,
    )
    elapsed = time.perf_counter() - t0
    logger.info("subject %s: lesion done in %.1fs", rec["subject_id"], elapsed)

    summary = roi_summary(result, rec["roi_indices"])
    return {
        "subject_id": rec["subject_id"],
        "elapsed_sec": elapsed,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "mask_strategy": result.mask_strategy,
        "roi_summary": summary,
        "full_r2_mean": float(result.full_r2.mean().item()),
        "full_r2_median": float(result.full_r2.median().item()),
        "delta_r2_means": {
            m: float(result.delta_r2[m].mean().item())
            for m in result.modality_order
        },
        "modality_order": result.modality_order,
    }, result


def run_study(
    subject_ids: list[int],
    cfg: dict,
    pilot: int | None,
    alphas: list[float],
    cv: int,
    mask: str,
    device: str,
    backend: str,
    mock: bool,
    output_dir: Path,
) -> dict:
    """Run the lesion study across subjects and write results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    per_subject = []
    lesion_objs: dict[int, LesionResult] = {}
    responses_for_ceiling = []

    # Parcellation is shared across subjects; load once. Mock mode keeps
    # its synthetic per-modality ROI bucket regardless.
    parcellation = None if mock else _resolve_parcellation(cfg)
    if parcellation is not None:
        logger.info("parcellation loaded: %d ROIs", len(parcellation))

    for sid in subject_ids:
        logger.info("loading subject %d", sid)
        rec = (
            _mock_subject_data(sid)
            if mock
            else _load_subject_data(sid, cfg, pilot, parcellation=parcellation)
        )
        summary, lesion = run_one_subject(
            rec, alphas=alphas, cv=cv, mask=mask,
            device=device, backend=backend,
        )
        per_subject.append(summary)
        lesion_objs[sid] = lesion
        responses_for_ceiling.append(rec["y_test"])

    # Group-level noise ceiling (only meaningful with multiple subjects).
    ceiling_mean = None
    if len(subject_ids) >= 2:
        stack = np.stack(responses_for_ceiling, axis=0)  # (S, n_test, n_vox)
        if stack.shape[0] >= 2:
            ceil = inter_subject_ceiling(stack)
            ceiling_mean = float(ceil.mean())
            np.save(output_dir / "noise_ceiling.npy", ceil)
            # Re-report normalized scores per subject.
            for s_summary, sid in zip(per_subject, subject_ids):
                full_r2 = lesion_objs[sid].full_r2.cpu().numpy()
                normalized = normalize_by_ceiling(full_r2, ceil)
                s_summary["full_r2_ceiling_normalized_mean"] = float(normalized.mean())

    manifest = {
        "n_subjects": len(subject_ids),
        "subject_ids": subject_ids,
        "mask_strategy": mask,
        "alphas": alphas,
        "cv": cv,
        "device": device,
        "backend": backend,
        "pilot": pilot,
        "mock": mock,
        "results": per_subject,
        "group_ceiling_mean": ceiling_mean,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Also save per-subject raw LesionResult arrays for downstream viz.
    for sid, lr in lesion_objs.items():
        np.savez_compressed(
            output_dir / f"subject_{sid:02d}_lesion.npz",
            full_r2=lr.full_r2.cpu().numpy(),
            **{f"delta_{m}": lr.delta_r2[m].cpu().numpy()
               for m in lr.modality_order},
            best_alpha=lr.best_alpha.cpu().numpy(),
        )

    logger.info("wrote %d subject result(s) to %s",
                len(subject_ids), output_dir)
    return manifest


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    cfg = _load_config(args.config)

    # CLI overrides YAML. Only apply when the user actually passed
    # something different from the defaults (`None` or the sentinel
    # "vision,text" for modalities).
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if args.feature_cache is not None:
        cfg["feature_cache"] = args.feature_cache
    if args.modalities:
        cfg["modalities"] = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if args.parcellation:
        cfg["parcellation"] = args.parcellation
    if args.lh_annot is not None:
        cfg["lh_annot"] = args.lh_annot
    if args.rh_annot is not None:
        cfg["rh_annot"] = args.rh_annot
    if args.parcellation_rois:
        cfg["parcellation_rois"] = [
            r.strip() for r in args.parcellation_rois.split(",") if r.strip()
        ]

    alphas = [float(a) for a in args.alphas.split(",")]

    run_study(
        subject_ids=args.subjects,
        cfg=cfg,
        pilot=args.pilot,
        alphas=alphas,
        cv=args.cv,
        mask=args.mask,
        device=args.device,
        backend=args.backend,
        mock=args.mock,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
