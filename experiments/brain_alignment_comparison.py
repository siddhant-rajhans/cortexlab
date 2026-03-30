"""Brain Alignment Comparison Experiment.

Compares how well different AI model representations align with
predicted brain responses using RSA, CKA, and Procrustes analysis.
Includes permutation tests and bootstrap confidence intervals.

Usage:
    python -m experiments.brain_alignment_comparison \
        --config experiments/config/brain_alignment_comparison.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_config(path: str | Path) -> dict:
    """Load experiment configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def generate_synthetic_data(config: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate synthetic brain predictions and model features."""
    rng = np.random.default_rng(config["experiment"]["seed"])
    n_stimuli = config["data"]["n_stimuli"]
    n_vertices = config["data"]["n_vertices"]

    brain_predictions = rng.standard_normal((n_stimuli, n_vertices))

    model_features = {}
    for model_spec in config["models"]:
        name = model_spec["name"]
        dim = model_spec["feature_dim"]
        model_features[name] = rng.standard_normal((n_stimuli, dim))

    return brain_predictions, model_features


def run_comparison(config: dict) -> dict:
    """Run the brain alignment comparison experiment."""
    from cortexlab.analysis.brain_alignment import BrainAlignmentBenchmark

    brain_pred, model_features = generate_synthetic_data(config)
    bench = BrainAlignmentBenchmark(brain_pred)

    methods = config["methods"]
    stats_config = config.get("statistics", {})
    n_perm = stats_config.get("n_permutations", 200)
    n_boot = stats_config.get("n_bootstrap", 500)
    confidence = stats_config.get("confidence", 0.95)

    results = {"models": {}, "config": config, "timestamp": datetime.now().isoformat()}

    for model_name, features in model_features.items():
        logger.info("Evaluating model: %s", model_name)
        model_results = {}

        for method in methods:
            # Score
            result = bench.score_model(features, method=method)
            entry = {"score": result.aggregate_score}

            # Permutation test
            _, p_value = bench.permutation_test(
                features, method=method, n_permutations=n_perm, seed=config["experiment"]["seed"]
            )
            entry["p_value"] = p_value

            # Bootstrap CI
            _, ci_lower, ci_upper = bench.bootstrap_ci(
                features, method=method, n_bootstrap=n_boot, confidence=confidence,
                seed=config["experiment"]["seed"]
            )
            entry["ci_lower"] = ci_lower
            entry["ci_upper"] = ci_upper

            model_results[method] = entry
            logger.info(
                "  %s: score=%.4f, p=%.4f, CI=[%.4f, %.4f]",
                method, entry["score"], p_value, ci_lower, ci_upper,
            )

        results["models"][model_name] = model_results

    return results


def save_results(results: dict, output_dir: str | Path) -> Path:
    """Save results as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"brain_alignment_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Brain Alignment Comparison")
    parser.add_argument(
        "--config",
        default="experiments/config/brain_alignment_comparison.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_comparison(config)
    save_results(results, config["experiment"]["output_dir"])

    # Print summary
    print("\n=== Brain Alignment Comparison Results ===\n")
    for model_name, methods in results["models"].items():
        print(f"  {model_name}:")
        for method, entry in methods.items():
            print(
                f"    {method:>10}: {entry['score']:+.4f}  "
                f"(p={entry['p_value']:.3f}, "
                f"CI=[{entry['ci_lower']:.4f}, {entry['ci_upper']:.4f}])"
            )
    print()


if __name__ == "__main__":
    main()
