"""Cross-Subject Generalization Evaluation.

Evaluates how well the SubjectAdapter generalizes to unseen subjects
using a leave-one-subject-out (LOSO) cross-validation scheme with
synthetic multi-subject data.

Usage:
    python -m experiments.cross_subject_eval \
        --config experiments/config/cross_subject_eval.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_synthetic_subjects(config: dict) -> dict:
    """Generate synthetic multi-subject brain data."""
    rng = np.random.default_rng(config["experiment"]["seed"])
    n_subjects = config["data"]["n_subjects"]
    n_timepoints = config["data"]["n_timepoints"]
    n_vertices = config["data"]["n_vertices"]
    hidden_dim = config["data"]["hidden_dim"]

    # Shared signal + per-subject noise
    shared_signal = rng.standard_normal((n_timepoints, n_vertices))

    subjects = {}
    for i in range(n_subjects):
        noise = rng.standard_normal((n_timepoints, n_vertices)) * 0.3
        subjects[f"sub-{i:02d}"] = {
            "fmri": shared_signal + noise,
            "hidden_states": rng.standard_normal((n_timepoints, hidden_dim)),
        }

    return subjects


def pearson_correlation(pred: np.ndarray, actual: np.ndarray) -> float:
    """Compute mean Pearson correlation across vertices."""
    correlations = []
    for v in range(pred.shape[1]):
        p, a = pred[:, v], actual[:, v]
        if p.std() < 1e-10 or a.std() < 1e-10:
            continue
        corr = np.corrcoef(p, a)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    return float(np.mean(correlations)) if correlations else 0.0


def run_loso_evaluation(config: dict) -> dict:
    """Run leave-one-subject-out evaluation."""
    subjects = generate_synthetic_subjects(config)
    subject_names = list(subjects.keys())
    calibration_ratios = config["data"]["calibration_ratios"]
    reg = config.get("ridge", {}).get("regularization", 1e-3)

    results = {
        "subjects": {},
        "summary": {},
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    for held_out in subject_names:
        logger.info("Held-out subject: %s", held_out)
        held_out_data = subjects[held_out]
        train_subjects = {k: v for k, v in subjects.items() if k != held_out}

        # Baseline: average of training subjects
        train_fmri = np.stack([s["fmri"] for s in train_subjects.values()])
        avg_pred = train_fmri.mean(axis=0)
        baseline_corr = pearson_correlation(avg_pred, held_out_data["fmri"])

        subject_result = {"baseline_correlation": baseline_corr}

        for ratio in calibration_ratios:
            n_cal = max(1, int(held_out_data["fmri"].shape[0] * ratio))
            cal_hidden = held_out_data["hidden_states"][:n_cal]
            cal_fmri = held_out_data["fmri"][:n_cal]
            test_hidden = held_out_data["hidden_states"][n_cal:]
            test_fmri = held_out_data["fmri"][n_cal:]

            if test_fmri.shape[0] < 2:
                continue

            # Ridge regression adaptation
            X = cal_hidden.astype(np.float64)
            Y = cal_fmri.astype(np.float64)
            XtX = X.T @ X
            lam = reg * np.eye(XtX.shape[0])
            W = np.linalg.solve(XtX + lam, X.T @ Y)
            ridge_pred = test_hidden @ W
            ridge_corr = pearson_correlation(ridge_pred, test_fmri)

            # Nearest-neighbour: use closest training subject's fMRI
            best_corr = -1.0
            for train_name, train_data in train_subjects.items():
                nn_corr = pearson_correlation(
                    train_data["fmri"][n_cal: n_cal + test_fmri.shape[0]], test_fmri
                )
                if nn_corr > best_corr:
                    best_corr = nn_corr

            subject_result[f"ridge_ratio_{ratio}"] = ridge_corr
            subject_result[f"nn_ratio_{ratio}"] = best_corr
            logger.info(
                "  ratio=%.1f: ridge=%.4f, nn=%.4f, baseline=%.4f",
                ratio, ridge_corr, best_corr, baseline_corr,
            )

        results["subjects"][held_out] = subject_result

    # Summary statistics
    for key in ["baseline_correlation"] + [f"ridge_ratio_{r}" for r in calibration_ratios]:
        values = [s.get(key, float("nan")) for s in results["subjects"].values()]
        values = [v for v in values if not np.isnan(v)]
        if values:
            results["summary"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    return results


def save_results(results: dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"cross_subject_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Cross-Subject Generalization Eval")
    parser.add_argument(
        "--config",
        default="experiments/config/cross_subject_eval.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_loso_evaluation(config)
    save_results(results, config["experiment"]["output_dir"])

    print("\n=== Cross-Subject Evaluation Summary ===\n")
    for key, stats in results["summary"].items():
        print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    print()


if __name__ == "__main__":
    main()
