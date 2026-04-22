"""Post-process an existing whole-cortex lesion output into per-ROI summary.

Given a directory of ``subject_XX_lesion.npz`` + ``manifest.json`` produced
by :mod:`experiments.causal_modality_ablation`, this script applies an
HCP-MMP parcellation and the BOLD Moments per-subject noise ceiling to
produce a per-ROI JSON + a group-mean CSV without refitting any model.

Useful when:

1. A whole-cortex run completed before parcellation / ceiling support
   was added (schema upgrade without rerun).
2. GPU is unavailable for a rerun but ROI-level tables are still needed.

Permutation-based p-values are **not** produced here; those require
repredicting with the encoder and need the original fitted weights.
Rerun the orchestrator with ``--permutations N`` to obtain them.

Usage
-----

    python scripts/postprocess_roi.py \\
        --results-dir $CORTEXLAB_RESULTS/lesion/all_subjects_20260422_135020 \\
        --lh-annot $ATLAS/lh.HCPMMP1.annot \\
        --rh-annot $ATLAS/rh.HCPMMP1.annot \\
        --data-root $CORTEXLAB_DATA

Outputs ``roi_summary.json`` and ``roi_summary_group.csv`` next to the
lesion .npz files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from cortexlab.data.parcellations import load_hcp_mmp_fsaverage
from cortexlab.data.studies.lahner2024bold import load_noise_ceiling


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Directory with subject_XX_lesion.npz + manifest.json.")
    ap.add_argument("--lh-annot", type=Path, required=True)
    ap.add_argument("--rh-annot", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=None,
                    help="BOLD Moments dataset root; falls back to CORTEXLAB_DATA env.")
    ap.add_argument("--ceiling-source", type=str, default="bold-moments",
                    choices=["bold-moments", "file"],
                    help="'bold-moments' loads per-subject n-10 pickles from "
                         "the dataset; 'file' loads a single shared ceiling "
                         "from --ceiling-file (e.g. a leave-one-out ceiling "
                         "from a prior lesion run).")
    ap.add_argument("--ceiling-file", type=Path, default=None,
                    help="Path to a 1-D .npy ceiling of length 2*N_VERTICES_PER_HEMI "
                         "(required when --ceiling-source=file).")
    ap.add_argument("--ceiling-n", type=int, default=10,
                    help="n suffix of the BOLD Moments ceiling pickle.")
    ap.add_argument("--ceiling-split", type=str, default="test",
                    choices=["train", "test"])
    ap.add_argument("--delta-keys", type=str, default=None,
                    help="Comma-separated modality names to look up as "
                         "delta_<m>. Default: inferred from manifest.results[0].modality_order.")
    return ap.parse_args()


def _resolve_data_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root
    env = os.environ.get("CORTEXLAB_DATA")
    if not env:
        raise SystemExit(
            "--data-root not given and CORTEXLAB_DATA not set; cannot load ceilings."
        )
    return Path(env)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    data_root = _resolve_data_root(args.data_root)

    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}; wrong --results-dir?")
    manifest = json.loads(manifest_path.read_text())

    modality_order: list[str]
    if args.delta_keys:
        modality_order = [m.strip() for m in args.delta_keys.split(",") if m.strip()]
    else:
        modality_order = manifest["results"][0]["modality_order"]
    print(f"Modality order: {modality_order}")

    roi_indices = load_hcp_mmp_fsaverage(args.lh_annot, args.rh_annot)
    print(f"Loaded {len(roi_indices)} ROIs")

    # A single shared ceiling (from a prior --ceiling-file) is loaded
    # once; per-subject ceilings are loaded inside the loop.
    shared_ceiling: np.ndarray | None = None
    if args.ceiling_source == "file":
        if args.ceiling_file is None:
            raise SystemExit("--ceiling-source=file requires --ceiling-file")
        if not args.ceiling_file.exists():
            raise SystemExit(f"ceiling file not found: {args.ceiling_file}")
        shared_ceiling = np.load(args.ceiling_file)
        print(f"Loaded shared ceiling {shared_ceiling.shape} from {args.ceiling_file}")

    per_subject_rows: list[dict] = []
    for sid in manifest["subject_ids"]:
        npz_path = results_dir / f"subject_{sid:02d}_lesion.npz"
        if not npz_path.exists():
            print(f"skipping sub-{sid:02d}: {npz_path} missing")
            continue
        npz = np.load(npz_path)
        full = npz["full_r2"]
        deltas = {m: npz[f"delta_{m}"] for m in modality_order}
        if shared_ceiling is not None:
            ceiling = shared_ceiling
        else:
            ceiling = load_noise_ceiling(
                subject_id=sid, root=str(data_root),
                split=args.ceiling_split, n=args.ceiling_n,
            )
        for roi, idx in roi_indices.items():
            c = ceiling[idx]
            mask = c > 0.01
            normalized = (
                float((full[idx][mask] / c[mask]).mean())
                if mask.any() else float("nan")
            )
            row = {
                "subject_id": sid,
                "roi": roi,
                "n_voxels": int(idx.size),
                "full_r2": float(full[idx].mean()),
                "full_r2_normalized": normalized,
                "ceiling_mean": float(c.mean()),
            }
            for m in modality_order:
                row[f"dR2_{m}"] = float(deltas[m][idx].mean())
            per_subject_rows.append(row)

    out_json = results_dir / "roi_summary.json"
    out_json.write_text(json.dumps(per_subject_rows, indent=2))
    print(f"Wrote {len(per_subject_rows)} subject*ROI rows to {out_json}")

    # Group-mean CSV, one row per ROI.
    by_roi: dict[str, list[dict]] = {}
    for r in per_subject_rows:
        by_roi.setdefault(r["roi"], []).append(r)

    header = [
        "roi", "n_subjects", "n_voxels",
        "full_r2_normalized_mean",
    ] + [f"dR2_{m}_mean" for m in modality_order]
    # Only emit the ratio column when exactly two modalities are present
    # (it's ill-defined for >2).
    emit_ratio = len(modality_order) == 2
    if emit_ratio:
        header.append(f"ratio_{modality_order[0]}_over_{modality_order[1]}")

    out_csv = results_dir / "roi_summary_group.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for roi, rows in by_roi.items():
            line: list = [
                roi, len(rows), rows[0]["n_voxels"],
                round(float(np.mean([r["full_r2_normalized"] for r in rows])), 4),
            ]
            means = {
                m: float(np.mean([r[f"dR2_{m}"] for r in rows]))
                for m in modality_order
            }
            for m in modality_order:
                line.append(round(means[m], 4))
            if emit_ratio:
                a = means[modality_order[0]]
                b = means[modality_order[1]]
                line.append(round(a / b, 2) if b > 0 else float("inf"))
            w.writerow(line)
    print(f"Wrote group CSV to {out_csv}")


if __name__ == "__main__":
    main()
